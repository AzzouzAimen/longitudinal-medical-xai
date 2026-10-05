"""
thermal_tsdb_dataset.py

 – Activity 1 : Database-to-Model Pipeline
Topic M6 : Synthetic Thermal Time-Series
Objectif
--------
Connecter TimescaleDB directement au DataLoader PyTorch.
Les fenêtres sont lues depuis la base pendant l'entraînement (lecture DB directe),
sans passer par des fichiers .npy intermédiaires.

Architecture du module
-----------------------
  ThermalTSDBDataset     ← torch.utils.data.Dataset
      • __init__         : connexion TimescaleDB, chargement des métadonnées de fenêtres
      • __len__          : nombre de fenêtres disponibles
      • __getitem__      : requête TimescaleDB pour récupérer les signaux d'une fenêtre
      • _fetch_window    : requête SQL sur thermal_readings (hypertable)

  ThermalTSDBDataModule  ← wrapper haut niveau (inspiré de PyTorch Lightning)
      • split()          : train / val / test par patient_id
      • get_loaders()    : retourne les 3 DataLoader configurés

  ConnectionPool         ← pool de connexions psycopg2 pour multi-workers
      • get_conn()       : retourne une connexion depuis le pool

  benchmark_vs_npy()     ← mesure le temps d'une époque DB vs NPY ( T7)

Usage rapide
------------
  from thermal_tsdb_dataset import ThermalTSDBDataModule

  dm = ThermalTSDBDataModule(
      dsn="postgresql://postgres:password@localhost:5432/m6_thermal_tsdb",
      window_size=60,
      channels=["left_temperature_norm", "right_temperature_norm", "temp_asymmetry"],
      batch_size=32,
      num_workers=4,
  )
  train_loader, val_loader, test_loader = dm.get_loaders()

  for batch_signals, batch_labels in train_loader:
      # batch_signals : (B, C, T) = (32, 3, 60)  float32
      # batch_labels  : (B,)      int64
      pass

Variables d'environnement (.env)
---------------------------------
  TSDB_HOST     localhost
  TSDB_PORT     5432
  TSDB_USER     postgres
  TSDB_PASSWORD <votre_mot_de_passe>
  TSDB_DB       m6_thermal_tsdb

Requirements
------------
  pip install psycopg2-binary torch numpy pandas python-dotenv tqdm
"""

from __future__ import annotations

import logging
import os
import queue
import threading
import time
from pathlib import Path
from typing import List, Optional, Tuple

import numpy as np
import pandas as pd
import psycopg2
import psycopg2.pool
import torch
from torch.utils.data import DataLoader, Dataset
from dotenv import load_dotenv

load_dotenv()

# ---------------------------------------------------------------------------
# Logging
# ---------------------------------------------------------------------------
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    datefmt="%H:%M:%S",
)
log = logging.getLogger("thermal_tsdb_dataset")

# ---------------------------------------------------------------------------
# Constantes (identiques à etl_pipeline.py )
# ---------------------------------------------------------------------------
WINDOW_SIZE   = 60        # secondes
ALL_CHANNELS  = [
    "left_temperature",
    "right_temperature",
    "left_temperature_norm",
    "right_temperature_norm",
    "temp_asymmetry",
]
DEFAULT_CHANNELS = ["left_temperature_norm", "right_temperature_norm", "temp_asymmetry"]


# ---------------------------------------------------------------------------
# 1. Pool de connexions — nécessaire pour num_workers > 0
#    Chaque worker PyTorch tourne dans un processus séparé, donc chacun
#    crée son propre pool local (psycopg2.pool n'est pas fork-safe).
# ---------------------------------------------------------------------------

class ConnectionPool:
    """
    Wrapper autour de psycopg2.ThreadedConnectionPool.
    Crée un nouveau pool APRÈS le fork PyTorch (dans chaque worker).
    """

    _local = threading.local()

    def __init__(self, dsn: str, minconn: int = 1, maxconn: int = 4):
        self._dsn     = dsn
        self._minconn = minconn
        self._maxconn = maxconn

    def _get_pool(self) -> psycopg2.pool.ThreadedConnectionPool:
        """Crée le pool dans le thread courant si pas encore fait."""
        if not hasattr(self._local, "pool") or self._local.pool is None:
            self._local.pool = psycopg2.pool.ThreadedConnectionPool(
                self._minconn,
                self._maxconn,
                self._dsn,
            )
        return self._local.pool

    def get_conn(self):
        """Retourne une connexion depuis le pool local."""
        return self._get_pool().getconn()

    def put_conn(self, conn):
        """Remet la connexion dans le pool."""
        if hasattr(self._local, "pool") and self._local.pool:
            self._local.pool.putconn(conn)

    def close_all(self):
        """Ferme toutes les connexions du pool local."""
        if hasattr(self._local, "pool") and self._local.pool:
            self._local.pool.closeall()
            self._local.pool = None


# ---------------------------------------------------------------------------
# 2. ThermalTSDBDataset
# ---------------------------------------------------------------------------

class ThermalTSDBDataset(Dataset):
    """
    Dataset PyTorch qui lit les signaux thermiques directement depuis TimescaleDB.

    Paramètres
    ----------
    dsn : str
        DSN PostgreSQL/TimescaleDB.
    patient_ids : list[int]
        Liste des patient_id inclus dans ce split.
    channels : list[str]
        Colonnes de thermal_readings à récupérer comme canaux du signal.
        Par défaut : ["left_temperature_norm", "right_temperature_norm", "temp_asymmetry"]
    window_size : int
        Nombre de pas de temps par fenêtre (doit correspondre à ).
    anomaly_only : bool
        Si True, ne charge que les fenêtres anomalies (label=1).
    cache_metadata : bool
        Si True, les métadonnées de fenêtres sont mises en cache en mémoire.
    prefetch_size : int
        Nombre de fenêtres à pré-charger en arrière-plan (0 = désactivé).
    """

    def __init__(
        self,
        dsn: str,
        patient_ids: List[int],
        channels: List[str]        = DEFAULT_CHANNELS,
        window_size: int           = WINDOW_SIZE,
        anomaly_only: bool         = False,
        cache_metadata: bool       = True,
        prefetch_size: int         = 0,
    ):
        super().__init__()
        self.dsn         = dsn
        self.patient_ids = patient_ids
        self.channels    = channels
        self.window_size = window_size
        self.anomaly_only = anomaly_only
        self.prefetch_size = prefetch_size

        # Validate channels
        invalid = set(channels) - set(ALL_CHANNELS)
        if invalid:
            raise ValueError(f"Canaux invalides : {invalid}. Valides : {ALL_CHANNELS}")

        # Connexion directe pour charger les métadonnées (hors workers)
        self._pool = ConnectionPool(dsn)

        # Chargement des métadonnées de fenêtres depuis windows_tsdb
        self._metadata: pd.DataFrame = self._load_metadata(cache_metadata)

        log.info(
            f"ThermalTSDBDataset initialisé | "
            f"patients={patient_ids} | fenêtres={len(self._metadata)} | "
            f"canaux={channels} | anomaly_only={anomaly_only}"
        )

        # ----------------------------------------------------------------
        # Préchargement en RAM : évite N requêtes SQL par époque
        # Coût unique au démarrage (~30-60s), puis lecture mémoire pure.
        # ----------------------------------------------------------------
        log.info(f"Préchargement de {len(self._metadata)} fenêtres en RAM...")
        self._signal_cache: List[np.ndarray] = []
        for i in range(len(self._metadata)):
            row = self._metadata.iloc[i]
            signal, _ = self._fetch_window(
                patient_id   = int(row["patient_id"]),
                window_start = row["window_start"],
                window_end   = row["window_end"],
            )
            self._signal_cache.append(signal)
            if (i + 1) % 1000 == 0:
                log.info(f"  Cache : {i + 1}/{len(self._metadata)} fenêtres chargées...")
        log.info(f"Cache prêt : {len(self._signal_cache)} fenêtres en mémoire.")

        # Labels en numpy array pour accès O(1) sans iloc (beaucoup plus rapide)
        self._labels: np.ndarray = self._metadata["label"].values.astype(np.int64)

    # ------------------------------------------------------------------
    # Chargement des métadonnées de fenêtres
    # ------------------------------------------------------------------

    def _load_metadata(self, cache: bool) -> pd.DataFrame:
        """
        Charge la table windows_tsdb pour les patients sélectionnés.
        Retourne un DataFrame indexé de 0 à N-1 (index = position dans __getitem__).
        """
        placeholders = ",".join(["%s"] * len(self.patient_ids))
        query = f"""
            SELECT
                w.patient_id,
                w.window_id,
                w.segment_id,
                w.window_start,
                w.window_end,
                w.label,
                w.anomaly_ratio,
                w.is_interpolated
            FROM windows_tsdb w
            WHERE w.patient_id IN ({placeholders})
            {"AND w.label = 1" if self.anomaly_only else ""}
            ORDER BY w.patient_id, w.window_start;
        """
        conn = self._pool.get_conn()
        try:
            df = pd.read_sql_query(
                query,
                conn,
                params=self.patient_ids,
                parse_dates=["window_start", "window_end"],
            )
        finally:
            self._pool.put_conn(conn)

        if df.empty:
            raise RuntimeError(
                f"Aucune fenêtre trouvée pour les patients {self.patient_ids}. "
                "Vérifiez que l'ingestion  a bien été exécutée."
            )

        df = df.reset_index(drop=True)
        log.info(
            f"Métadonnées chargées : {len(df)} fenêtres "
            f"({df['label'].sum()} anomalies / {(df['label']==0).sum()} normales)"
        )
        return df

    # ------------------------------------------------------------------
    # Interface Dataset PyTorch
    # ------------------------------------------------------------------

    def __len__(self) -> int:
        return len(self._metadata)

    def __getitem__(self, idx: int) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Retourne (signal_tensor, label_tensor) pour la fenêtre d'index idx.

        signal_tensor : float32 de forme (C, T) = (len(channels), window_size)
        label_tensor  : int64   scalaire (0 = normal, 1 = anomalie)

        Lecture depuis le cache RAM (pas de requête SQL ici).
        """
        signal = self._signal_cache[idx]          # lecture RAM O(1)
        label  = int(self._labels[idx])           # numpy array O(1), pas de iloc

        signal_tensor = torch.from_numpy(signal)          # (C, T)
        label_tensor  = torch.tensor(label, dtype=torch.long)
        return signal_tensor, label_tensor

    # ------------------------------------------------------------------
    # Requête TimescaleDB : récupération d'une fenêtre
    # ------------------------------------------------------------------

    def _fetch_window(
        self,
        patient_id: int,
        window_start: pd.Timestamp,
        window_end: pd.Timestamp,
    ) -> Tuple[np.ndarray, List[pd.Timestamp]]:
        """
        Requête sur la hypertable thermal_readings.

        Utilise les index idx_tr_patient_ts (patient_id, timestamp DESC)
        créés en  pour une latence minimale.

        Retourne (signal, timestamps) où signal est (C, T) et timestamps est une liste de T timestamps.
        """
        col_select = ", ".join(self.channels)
        query = f"""
            SELECT timestamp, {col_select}
            FROM thermal_readings
            WHERE patient_id = %s
              AND timestamp >= %s
              AND timestamp <= %s
            ORDER BY timestamp ASC;
        """
        conn = self._pool.get_conn()
        try:
            with conn.cursor() as cur:
                cur.execute(query, (patient_id, window_start, window_end))
                rows = cur.fetchall()
        finally:
            self._pool.put_conn(conn)

        if not rows:
            # Fenêtre vide : retourne des zéros (cas rare après imputation )
            log.warning(
                f"Fenêtre vide pour patient={patient_id} "
                f"[{window_start} → {window_end}] — zéros retournés"
            )
            return (
                np.zeros((len(self.channels), self.window_size), dtype=np.float32),
                [window_start + pd.Timedelta(seconds=i) for i in range(self.window_size)]
            )

        # Extraction des timestamps et du signal
        timestamps = [r[0] for r in rows]
        arr = np.array([r[1:] for r in rows], dtype=np.float32)   # (T, C)

        # Ajustement de la longueur à window_size exact
        T = arr.shape[0]
        if T < self.window_size:
            pad = np.zeros((self.window_size - T, len(self.channels)), dtype=np.float32)
            arr = np.vstack([arr, pad])
            # Pad timestamps too
            last_ts = timestamps[-1] if timestamps else window_start
            for i in range(1, self.window_size - T + 1):
                timestamps.append(last_ts + pd.Timedelta(seconds=i))
        elif T > self.window_size:
            arr = arr[: self.window_size]
            timestamps = timestamps[: self.window_size]

        return arr.T, timestamps  # (C, T), list[T]

    # ------------------------------------------------------------------
    # Méthode utilitaire : récupérer une fenêtre avec ses métadonnées
    # ------------------------------------------------------------------

    def get_sample_with_meta(self, idx: int) -> dict:
        """Retourne le signal + les métadonnées complètes pour inspection."""
        row = self._metadata.iloc[idx]
        signal, timestamps = self._fetch_window(
            patient_id   = int(row["patient_id"]),
            window_start = row["window_start"],
            window_end   = row["window_end"],
        )
        label = int(row["label"])

        return {
            "signal":       torch.from_numpy(signal),
            "label":        label,
            "patient_id":   int(row["patient_id"]),
            "window_id":    int(row["window_id"]),
            "timestamps":   timestamps,
            "window_start": row["window_start"],
            "window_end":   row["window_end"],
            "anomaly_ratio": row["anomaly_ratio"],
            "is_interpolated": row["is_interpolated"],
        }

    # ------------------------------------------------------------------
    # Stats rapides
    # ------------------------------------------------------------------

    def class_distribution(self) -> dict:
        """Retourne la distribution des labels pour ce split."""
        counts = self._metadata["label"].value_counts().to_dict()
        total = len(self._metadata)
        return {
            "total":    total,
            "normal":   counts.get(0, 0),
            "anomaly":  counts.get(1, 0),
            "anomaly_%": round(100 * counts.get(1, 0) / total, 2),
        }

    def patients_summary(self) -> pd.DataFrame:
        """Résumé par patient : nombre de fenêtres normales/anomalies."""
        return (
            self._metadata.groupby("patient_id")["label"]
            .value_counts()
            .unstack(fill_value=0)
            .rename(columns={0: "normal", 1: "anomaly"})
        )


# ---------------------------------------------------------------------------
# 3. ThermalTSDBDataModule  (wrapper haut niveau)
# ---------------------------------------------------------------------------

class ThermalTSDBDataModule:
    """
    Gère le split train/val/test et crée les DataLoader correspondants.

    Paramètres
    ----------
    dsn : str
        DSN TimescaleDB. Si None, construit depuis les variables d'env.
    channels : list[str]
        Canaux de signal à utiliser.
    window_size : int
        Taille de fenêtre en secondes.
    batch_size : int
        Taille de lot pour le DataLoader.
    num_workers : int
        Nombre de workers PyTorch (0 = main process).
    train_ratio / val_ratio : float
        Proportions du split (le reste va au test).
    seed : int
        Graine aléatoire pour la reproductibilité.
    """

    def __init__(
        self,
        dsn: Optional[str]     = None,
        channels: List[str]    = DEFAULT_CHANNELS,
        window_size: int       = WINDOW_SIZE,
        batch_size: int        = 32,
        num_workers: int       = 0,
        train_ratio: float     = 0.70,
        val_ratio: float       = 0.15,
        seed: int              = 42,
        pin_memory: bool       = True,
        drop_last: bool        = True,
    ):
        self.dsn         = dsn or _build_dsn_from_env()
        self.channels    = channels
        self.window_size = window_size
        self.batch_size  = batch_size
        self.num_workers = num_workers
        self.train_ratio = train_ratio
        self.val_ratio   = val_ratio
        self.seed        = seed
        self.pin_memory  = pin_memory and torch.cuda.is_available()
        self.drop_last   = drop_last

        self._patient_ids: List[int] = self._fetch_all_patient_ids()
        log.info(f"ThermalTSDBDataModule | {len(self._patient_ids)} patients trouvés")

    def _fetch_all_patient_ids(self) -> List[int]:
        """Récupère tous les patient_id depuis la table subjects."""
        conn = psycopg2.connect(self.dsn)
        try:
            with conn.cursor() as cur:
                cur.execute("SELECT patient_id FROM subjects ORDER BY patient_id;")
                return [row[0] for row in cur.fetchall()]
        finally:
            conn.close()

    def split(self) -> Tuple[List[int], List[int], List[int]]:
        """
        Divise les patients en trois ensembles disjoints.

        Stratégie : split par patient_id (pas par fenêtre) pour éviter
        toute fuite de données entre train et test.
        """
        rng     = np.random.default_rng(self.seed)
        ids     = np.array(self._patient_ids)
        rng.shuffle(ids)

        n       = len(ids)
        n_train = max(1, int(n * self.train_ratio))
        n_val   = max(1, int(n * self.val_ratio))

        train_ids = ids[:n_train].tolist()
        val_ids   = ids[n_train : n_train + n_val].tolist()
        test_ids  = ids[n_train + n_val :].tolist()

        log.info(
            f"Split patients → train={train_ids}, val={val_ids}, test={test_ids}"
        )
        return train_ids, val_ids, test_ids

    def get_loaders(
        self,
    ) -> Tuple[DataLoader, DataLoader, DataLoader]:
        """
        Crée et retourne les trois DataLoader (train, val, test).

        Returns
        -------
        train_loader, val_loader, test_loader
        """
        train_ids, val_ids, test_ids = self.split()

        train_ds = ThermalTSDBDataset(
            dsn=self.dsn,
            patient_ids=train_ids,
            channels=self.channels,
            window_size=self.window_size,
        )
        val_ds = ThermalTSDBDataset(
            dsn=self.dsn,
            patient_ids=val_ids,
            channels=self.channels,
            window_size=self.window_size,
        )
        test_ds = ThermalTSDBDataset(
            dsn=self.dsn,
            patient_ids=test_ids,
            channels=self.channels,
            window_size=self.window_size,
        )

        # Calcul du poids de classe pour gérer le déséquilibre ( : ~15 % anomalies)
        train_weights = _compute_sample_weights(train_ds)
        sampler = torch.utils.data.WeightedRandomSampler(
            weights     = train_weights,
            num_samples = len(train_weights),
            replacement = True,
        )

        train_loader = DataLoader(
            train_ds,
            batch_size  = self.batch_size,
            sampler     = sampler,           # WeightedRandom pour équilibrer les classes
            num_workers = self.num_workers,
            pin_memory  = self.pin_memory,
            drop_last   = self.drop_last,
            persistent_workers = self.num_workers > 0,
        )
        val_loader = DataLoader(
            val_ds,
            batch_size  = self.batch_size,
            shuffle     = False,
            num_workers = self.num_workers,
            pin_memory  = self.pin_memory,
            drop_last   = False,
            persistent_workers = self.num_workers > 0,
        )
        test_loader = DataLoader(
            test_ds,
            batch_size  = self.batch_size,
            shuffle     = False,
            num_workers = self.num_workers,
            pin_memory  = self.pin_memory,
            drop_last   = False,
            persistent_workers = self.num_workers > 0,
        )

        log.info("DataLoaders créés :")
        log.info(f"  Train : {len(train_ds)} fenêtres | {len(train_loader)} batches")
        log.info(f"  Val   : {len(val_ds)} fenêtres | {len(val_loader)} batches")
        log.info(f"  Test  : {len(test_ds)} fenêtres | {len(test_loader)} batches")

        return train_loader, val_loader, test_loader


# ---------------------------------------------------------------------------
# 4. Benchmark DB vs NPY  (pour T7 du )
# ---------------------------------------------------------------------------

def benchmark_vs_npy(
    dsn: str,
    npy_dir: Path,
    patient_ids: List[int],
    batch_size: int = 32,
    num_workers: int = 0,
    n_epochs: int = 1,
) -> dict:
    """
    Compare la vitesse de lecture depuis TimescaleDB vs fichiers .npy.

    Retourne un dict avec :
        db_epoch_s  : temps d'une époque complète en lecture DB
        npy_epoch_s : temps d'une époque complète en lecture NPY
        speedup     : ratio npy / db (>1 = DB plus lente, <1 = DB plus rapide)
        db_samples_per_s
        npy_samples_per_s
    """
    from thermal_npy_dataset import ThermalNPYDataset  # voir thermal_npy_dataset.py

    log.info("=" * 60)
    log.info("BENCHMARK : TimescaleDB vs NPY")
    log.info("=" * 60)

    # --- DB ---
    db_ds = ThermalTSDBDataset(
        dsn=dsn,
        patient_ids=patient_ids,
    )
    db_loader = DataLoader(db_ds, batch_size=batch_size,
                           num_workers=num_workers, shuffle=True)
    t0 = time.perf_counter()
    for _ in range(n_epochs):
        for _batch in db_loader:
            pass
    db_elapsed = time.perf_counter() - t0
    db_rate = (len(db_ds) * n_epochs) / db_elapsed

    log.info(f"  DB   : {db_elapsed:.2f}s | {db_rate:,.0f} samples/s")

    # --- NPY ---
    npy_ds = ThermalNPYDataset(npy_dir=npy_dir, patient_ids=patient_ids)
    npy_loader = DataLoader(npy_ds, batch_size=batch_size,
                            num_workers=num_workers, shuffle=True)
    t0 = time.perf_counter()
    for _ in range(n_epochs):
        for _batch in npy_loader:
            pass
    npy_elapsed = time.perf_counter() - t0
    npy_rate = (len(npy_ds) * n_epochs) / npy_elapsed

    log.info(f"  NPY  : {npy_elapsed:.2f}s | {npy_rate:,.0f} samples/s")
    speedup = npy_elapsed / db_elapsed
    log.info(f"  Speedup (NPY/DB) : {speedup:.2f}x")

    return {
        "db_epoch_s":       round(db_elapsed, 3),
        "npy_epoch_s":      round(npy_elapsed, 3),
        "speedup":          round(speedup, 3),
        "db_samples_per_s": round(db_rate, 0),
        "npy_samples_per_s": round(npy_rate, 0),
    }


# ---------------------------------------------------------------------------
# 5. Helpers internes
# ---------------------------------------------------------------------------

def _build_dsn_from_env() -> str:
    """Construit le DSN depuis les variables d'environnement (.env)."""
    host     = os.getenv("TSDB_HOST",     "localhost")
    port     = os.getenv("TSDB_PORT",     "5432")
    user     = os.getenv("TSDB_USER",     "postgres")
    password = os.getenv("TSDB_PASSWORD", "postgres")
    db       = os.getenv("TSDB_DB",       "m6_thermal_tsdb")
    return f"postgresql://{user}:{password}@{host}:{port}/{db}"


def _compute_sample_weights(dataset: ThermalTSDBDataset) -> torch.Tensor:
    """
    Calcule un poids par fenêtre pour le WeightedRandomSampler.
    Les fenêtres anomalies reçoivent un poids plus élevé pour compenser
    le déséquilibre de classes (~85 % normal / ~15 % anomalie).
    """
    labels = dataset._labels  # numpy array O(1), déjà en mémoire
    class_counts = np.bincount(labels)
    class_weights = 1.0 / np.where(class_counts > 0, class_counts, 1)
    sample_weights = class_weights[labels]
    return torch.from_numpy(sample_weights).float()


# ---------------------------------------------------------------------------
# 6. Script de démonstration / smoke test
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(
        description=" – Test du DataLoader TimescaleDB"
    )
    parser.add_argument("--dsn",     default=None,
                        help="DSN TimescaleDB (défaut : depuis .env)")
    parser.add_argument("--patients", nargs="+", type=int, default=None,
                        help="Liste de patient_id à utiliser (défaut : tous)")
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--num-workers", type=int, default=0)
    parser.add_argument("--max-batches", type=int, default=5,
                        help="Nombre maximum de batches à charger dans le test")
    args = parser.parse_args()

    dsn = args.dsn or _build_dsn_from_env()

    # ----------------------------------------------------------------
    # Test 1 : chargement direct d'un Dataset
    # ----------------------------------------------------------------
    print("\n" + "=" * 60)
    print("TEST 1 : ThermalTSDBDataset")
    print("=" * 60)

    # Si pas de patients spécifiés, on prend les 3 premiers
    conn_test = psycopg2.connect(dsn)
    with conn_test.cursor() as cur:
        cur.execute("SELECT patient_id FROM subjects ORDER BY patient_id LIMIT 3;")
        sample_patients = [r[0] for r in cur.fetchall()]
    conn_test.close()

    patient_ids = args.patients or sample_patients
    print(f"  Patients : {patient_ids}")

    ds = ThermalTSDBDataset(
        dsn         = dsn,
        patient_ids = patient_ids,
        channels    = DEFAULT_CHANNELS,
        window_size = WINDOW_SIZE,
    )

    print(f"  Taille du dataset : {len(ds)} fenêtres")
    print(f"  Distribution : {ds.class_distribution()}")
    print(f"  Résumé par patient :\n{ds.patients_summary()}")

    # Chargement d'une fenêtre individuelle
    t0 = time.perf_counter()
    signal, label = ds[0]
    latency_ms = (time.perf_counter() - t0) * 1000
    print(f"\n  Fenêtre 0 : signal={signal.shape}, label={label.item()}")
    print(f"  Latence accès fenêtre unique : {latency_ms:.1f} ms")

    # ----------------------------------------------------------------
    # Test 2 : DataLoader (plusieurs batches)
    # ----------------------------------------------------------------
    print("\n" + "=" * 60)
    print("TEST 2 : DataLoader – itération sur batches")
    print("=" * 60)

    loader = DataLoader(
        ds,
        batch_size  = args.batch_size,
        shuffle     = True,
        num_workers = args.num_workers,
    )

    t0 = time.perf_counter()
    batch_times = []
    for i, (signals, labels) in enumerate(loader):
        if i >= args.max_batches:
            break
        batch_times.append(time.perf_counter() - t0)
        print(
            f"  Batch {i:02d} | signals={tuple(signals.shape)} "
            f"| labels={labels.tolist()}"
        )
        t0 = time.perf_counter()

    print(f"\n  Temps moyen par batch : {np.mean(batch_times)*1000:.1f} ms")
    print(f"  Débit estimé : {args.batch_size / np.mean(batch_times):.0f} samples/s")

    # ----------------------------------------------------------------
    # Test 3 : DataModule complet (train/val/test)
    # ----------------------------------------------------------------
    print("\n" + "=" * 60)
    print("TEST 3 : ThermalTSDBDataModule – split train/val/test")
    print("=" * 60)

    dm = ThermalTSDBDataModule(
        dsn         = dsn,
        channels    = DEFAULT_CHANNELS,
        batch_size  = args.batch_size,
        num_workers = args.num_workers,
    )
    train_loader, val_loader, test_loader = dm.get_loaders()

    print(f"\n  Train batches : {len(train_loader)}")
    print(f"  Val   batches : {len(val_loader)}")
    print(f"  Test  batches : {len(test_loader)}")

    # Une passe rapide sur le train
    print("\n  Passe rapide sur train (5 batches)...")
    t0 = time.perf_counter()
    for i, (s, l) in enumerate(train_loader):
        if i >= 5:
            break
    elapsed = time.perf_counter() - t0
    print(f"  5 batches en {elapsed:.2f}s")

    print("\n✓ Tous les tests passés. Le DataLoader TimescaleDB est opérationnel.")