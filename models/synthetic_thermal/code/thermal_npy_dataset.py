"""
thermal_npy_dataset.py

 – Module de référence NPY
Topic M6 : Synthetic Thermal Time-Series
Chargement des fenêtres depuis les fichiers .npy produits par l'ETL .
Utilisé uniquement pour le benchmark DB vs NPY (benchmark_vs_npy()).

Les fichiers attendus (produits par etl_pipeline.py) :
  <npy_dir>/patient_XX_windows.npy          shape (N, 5, 60)
  <npy_dir>/patient_XX_windows_meta.csv     colonnes : window_id, label, ...

Canaux dans le .npy (ordre fixe, défini dans etl_pipeline.py) :
  0 : left_temperature
  1 : right_temperature
  2 : left_temperature_norm
  3 : right_temperature_norm
  4 : temp_asymmetry
"""

from __future__ import annotations

import logging
from pathlib import Path
from typing import List, Tuple

import numpy as np
import pandas as pd
import torch
from torch.utils.data import Dataset

log = logging.getLogger("thermal_npy_dataset")

# Mapping canal → index dans le .npy
CHANNEL_INDEX = {
    "left_temperature":       0,
    "right_temperature":      1,
    "left_temperature_norm":  2,
    "right_temperature_norm": 3,
    "temp_asymmetry":         4,
}
DEFAULT_CHANNELS = ["left_temperature_norm", "right_temperature_norm", "temp_asymmetry"]
DEFAULT_CHANNEL_INDICES = [CHANNEL_INDEX[c] for c in DEFAULT_CHANNELS]


class ThermalNPYDataset(Dataset):
    """
    Dataset PyTorch basé sur les fichiers .npy (référence offline).

    Paramètres
    ----------
    npy_dir : Path
        Répertoire contenant patient_XX_windows.npy et patient_XX_windows_meta.csv.
    patient_ids : list[int]
        Patients à inclure.
    channels : list[str]
        Canaux à extraire (sous-ensemble de CHANNEL_INDEX).
    """

    def __init__(
        self,
        npy_dir: Path,
        patient_ids: List[int],
        channels: List[str] = DEFAULT_CHANNELS,
    ):
        self.npy_dir     = Path(npy_dir)
        self.patient_ids = patient_ids
        self.channels    = channels
        self.ch_indices  = [CHANNEL_INDEX[c] for c in channels]

        self._signals: np.ndarray  # (N_total, len(channels), 60)
        self._labels:  np.ndarray  # (N_total,)
        self._signals, self._labels = self._load_all()

        log.info(
            f"ThermalNPYDataset | patients={patient_ids} | "
            f"fenêtres={len(self._labels)} | canaux={channels}"
        )

    def _load_all(self) -> Tuple[np.ndarray, np.ndarray]:
        all_signals = []
        all_labels  = []

        for pid in self.patient_ids:
            npy_path  = self.npy_dir / f"patient_{pid:02d}_windows.npy"
            meta_path = self.npy_dir / f"patient_{pid:02d}_windows_meta.csv"

            if not npy_path.exists():
                log.warning(f"Fichier manquant : {npy_path}")
                continue

            arr  = np.load(npy_path)              # (N, 5, 60)
            meta = pd.read_csv(meta_path)

            # Sélection des canaux voulus
            arr = arr[:, self.ch_indices, :]      # (N, C, 60)

            all_signals.append(arr)
            all_labels.append(meta["label"].values)

        if not all_signals:
            raise RuntimeError(
                f"Aucun fichier .npy trouvé dans {self.npy_dir} "
                f"pour les patients {self.patient_ids}."
            )

        return (
            np.concatenate(all_signals, axis=0).astype(np.float32),
            np.concatenate(all_labels,  axis=0).astype(np.int64),
        )

    def __len__(self) -> int:
        return len(self._labels)

    def __getitem__(self, idx: int) -> Tuple[torch.Tensor, torch.Tensor]:
        signal = torch.from_numpy(self._signals[idx])      # (C, T)
        label  = torch.tensor(self._labels[idx], dtype=torch.long)
        return signal, label
