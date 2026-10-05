"""
baselines.py
 – Task 1 : Baseline Algorithms
Topic M6 : Synthetic Thermal Time-Series
Implements three baseline algorithms for comparison against TAAE (MedAttnAID):
  - Algorithm 6 : Supervised BiLSTM           (alg6)
  - Algorithm 7 : Feature-based K-Means       (alg7)
  - Algorithm 8 : TS-KMeans with DTW          (alg8)

Each baseline produces window-level anomaly predictions compatible with Table I:
  F1 (%), Precision (%), Recall (%), Individual Accuracy (%)

"""
from __future__ import annotations

import argparse
import csv
import logging
import os
import time
import warnings
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, Dataset, WeightedRandomSampler
from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler
from tqdm import tqdm

warnings.filterwarnings("ignore", category=UserWarning)

# ---------------------------------------------------------------------------
# Logging
# ---------------------------------------------------------------------------
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    datefmt="%H:%M:%S",
)
log = logging.getLogger("baselines")

# ---------------------------------------------------------------------------
# Constants 
# ---------------------------------------------------------------------------
WINDOW_SIZE = 60       # timesteps per window
N_CHANNELS  = 3        # left_norm, right_norm, asymmetry
CHANNEL_INDEX = {
    "left_temperature":       0,
    "right_temperature":      1,
    "left_temperature_norm":  2,
    "right_temperature_norm": 3,
    "temp_asymmetry":         4,
}
DEFAULT_CHANNELS        = ["left_temperature_norm", "right_temperature_norm", "temp_asymmetry"]
DEFAULT_CHANNEL_INDICES = [CHANNEL_INDEX[c] for c in DEFAULT_CHANNELS]

SEED = 42


# ===========================================================================
# Data Loading  (mirrors thermal_npy_dataset.py)
# ===========================================================================

class ThermalNPYDataset(Dataset):
    """
    Loads windows from .npy files produced by  ETL.

    Returns (signal_tensor [C, T], label_tensor [scalar]) per item,
    exactly matching the format expected by train.py DataLoaders.
    """

    def __init__(
        self,
        npy_dir: Path,
        patient_ids: List[int],
        channels: List[str] = DEFAULT_CHANNELS,
    ):
        self.npy_dir    = Path(npy_dir)
        self.ch_indices = [CHANNEL_INDEX[c] for c in channels]

        signals_list, labels_list = [], []
        for pid in patient_ids:
            npy_path  = self.npy_dir / f"patient_{pid:02d}_windows.npy"
            meta_path = self.npy_dir / f"patient_{pid:02d}_windows_meta.csv"

            if not npy_path.exists():
                log.warning(f"Missing: {npy_path} — skipping patient {pid}")
                continue

            arr  = np.load(npy_path)              # (N, 5, 60)
            meta = pd.read_csv(meta_path)
            arr  = arr[:, self.ch_indices, :]     # (N, C, 60)

            signals_list.append(arr)
            labels_list.append(meta["label"].values)

        if not signals_list:
            raise RuntimeError(
                f"No .npy files found in {npy_dir} for patients {patient_ids}.\n"
                "Check that  ETL has been run and --npy-dir is correct."
            )

        self.signals = np.concatenate(signals_list, axis=0).astype(np.float32)  # (N, C, T)
        self.labels  = np.concatenate(labels_list,  axis=0).astype(np.int64)    # (N,)

        log.info(
            f"Loaded {len(self.labels)} windows from patients {patient_ids} | "
            f"anomalies: {self.labels.sum()} / {len(self.labels)} "
            f"({100*self.labels.mean():.1f}%)"
        )

    def __len__(self) -> int:
        return len(self.labels)

    def __getitem__(self, idx: int) -> Tuple[torch.Tensor, torch.Tensor]:
        return (
            torch.from_numpy(self.signals[idx]),
            torch.tensor(self.labels[idx], dtype=torch.long),
        )


def make_splits(
    npy_dir: Path,
    patient_ids: List[int],
    train_ratio: float = 0.70,
    val_ratio:   float = 0.15,
    seed: int = SEED,
) -> Tuple[ThermalNPYDataset, ThermalNPYDataset, ThermalNPYDataset]:
    """
    Patient-level train/val/test split — same logic as ThermalTSDBDataModule.split().
    Split is by patient to avoid data leakage.
    """
    rng = np.random.default_rng(seed)
    ids = np.array(patient_ids)
    rng.shuffle(ids)

    n       = len(ids)
    n_train = max(1, int(n * train_ratio))
    n_val   = max(1, int(n * val_ratio))

    train_ids = ids[:n_train].tolist()
    val_ids   = ids[n_train : n_train + n_val].tolist()
    test_ids  = ids[n_train + n_val :].tolist()

    log.info(f"Patient split → train={train_ids} | val={val_ids} | test={test_ids}")

    return (
        ThermalNPYDataset(npy_dir, train_ids),
        ThermalNPYDataset(npy_dir, val_ids),
        ThermalNPYDataset(npy_dir, test_ids),
    )


def _compute_sample_weights(labels: np.ndarray) -> torch.Tensor:
    """WeightedRandomSampler weights to balance class imbalance (~85/15)."""
    counts  = np.bincount(labels)
    weights = 1.0 / np.where(counts > 0, counts, 1)
    return torch.from_numpy(weights[labels]).float()


# ===========================================================================
# Evaluation helpers
# ===========================================================================

def compute_metrics(
    y_true:    np.ndarray,
    y_pred:    np.ndarray,
    patient_ids_per_window: Optional[np.ndarray] = None,
    anomaly_pct_threshold:  float = 5.0,
) -> Dict[str, float]:
    """
    Computes Table I metrics:
      - F1 (%)
      - Precision (%)
      - Recall (%)
      - Individual Accuracy (%) : subject classified as pathological
        if anomaly_pct > threshold; all subjects must be classified correctly.

    Parameters
    ----------
    y_true : (N,) ground truth window labels
    y_pred : (N,) predicted window labels
    patient_ids_per_window : (N,) patient id for each window (for Individual Acc.)
    anomaly_pct_threshold  : % of anomalous windows needed to flag a patient
    """
    tp = ((y_pred == 1) & (y_true == 1)).sum()
    fp = ((y_pred == 1) & (y_true == 0)).sum()
    fn = ((y_pred == 0) & (y_true == 1)).sum()

    precision = tp / (tp + fp + 1e-8)
    recall    = tp / (tp + fn + 1e-8)
    f1        = 2 * precision * recall / (precision + recall + 1e-8)

    metrics = {
        "F1 (%)":        round(float(f1)        * 100, 2),
        "Precision (%)": round(float(precision)  * 100, 2),
        "Recall (%)":    round(float(recall)     * 100, 2),
    }

    # Individual accuracy: classify each patient
    if patient_ids_per_window is not None:
        patient_ids = np.unique(patient_ids_per_window)
        correct = 0
        for pid in patient_ids:
            mask        = (patient_ids_per_window == pid)
            true_labels = y_true[mask]
            pred_labels = y_pred[mask]

            # Ground truth: patient is pathological if ANY window is anomalous
            is_pathological_gt = int(true_labels.sum() > 0)

            # Prediction: pathological if > threshold% windows flagged
            anomaly_pct = 100.0 * pred_labels.mean()
            is_pathological_pred = int(anomaly_pct > anomaly_pct_threshold)

            if is_pathological_gt == is_pathological_pred:
                correct += 1

        metrics["Individual Acc. (%)"] = round(100 * correct / len(patient_ids), 2)
    else:
        metrics["Individual Acc. (%)"] = round(
            100 * float((y_pred == y_true).mean()), 2
        )

    return metrics


# ===========================================================================
# Algorithm 6 — Supervised BiLSTM
# ===========================================================================

class BiLSTMClassifier(nn.Module):
    """
    Supervised BiLSTM anomaly classifier.

    Architecture (Algorithm 6):
      Input  : (B, C, T)
      BiLSTM : 2 layers, hidden=64
      Pool   : last hidden state (concat fwd + bwd → 128)
      MLP    : 128 → 64 → 2 (binary classification)
    """

    def __init__(
        self,
        n_channels:  int   = N_CHANNELS,
        window_size: int   = WINDOW_SIZE,
        hidden_size: int   = 64,
        num_layers:  int   = 2,
        dropout:     float = 0.3,
    ):
        super().__init__()

        self.lstm = nn.LSTM(
            input_size   = n_channels,
            hidden_size  = hidden_size,
            num_layers   = num_layers,
            batch_first  = True,
            bidirectional= True,
            dropout      = dropout if num_layers > 1 else 0.0,
        )
        self.dropout = nn.Dropout(dropout)

        self.classifier = nn.Sequential(
            nn.Linear(hidden_size * 2, 64),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(64, 2),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """x : (B, C, T) → logits : (B, 2)"""
        # (B, C, T) → (B, T, C) for LSTM batch_first
        x = x.permute(0, 2, 1)
        out, (h_n, _) = self.lstm(x)

        # Concat forward and backward last hidden states
        # h_n shape: (num_layers * 2, B, hidden_size)
        # Take last layer: h_n[-2] = forward, h_n[-1] = backward
        h_fwd = h_n[-2]   # (B, hidden_size)
        h_bwd = h_n[-1]   # (B, hidden_size)
        h = torch.cat([h_fwd, h_bwd], dim=1)   # (B, hidden_size*2)
        h = self.dropout(h)

        return self.classifier(h)   # (B, 2)


def run_bilstm(
    train_ds: ThermalNPYDataset,
    test_ds:  ThermalNPYDataset,
    epochs:   int   = 30,
    batch_size: int = 256,
    lr:       float = 1e-3,
    device:   Optional[torch.device] = None,
    patient_ids_test: Optional[np.ndarray] = None,
) -> Dict:
    """
    Trains and evaluates the Supervised BiLSTM baseline (Algorithm 6).

    Returns dict with metrics + timing info.
    """
    if device is None:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    log.info("=" * 60)
    log.info("BASELINE alg6 — Supervised BiLSTM")
    log.info("=" * 60)

    # Weighted sampler for class imbalance
    train_weights = _compute_sample_weights(train_ds.labels)
    sampler = WeightedRandomSampler(train_weights, len(train_weights), replacement=True)

    train_loader = DataLoader(
        train_ds, batch_size=batch_size, sampler=sampler, num_workers=0
    )
    test_loader = DataLoader(
        test_ds, batch_size=batch_size, shuffle=False, num_workers=0
    )

    model     = BiLSTMClassifier().to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=1e-4)

    # Class weights for loss: compensate ~85/15 imbalance
    class_counts = np.bincount(train_ds.labels)
    class_weights = torch.tensor(
        [1.0 / c if c > 0 else 1.0 for c in class_counts],
        dtype=torch.float32
    ).to(device)
    criterion = nn.CrossEntropyLoss(weight=class_weights)

    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, mode="min", patience=5, factor=0.5
    )

    t0 = time.perf_counter()

    for epoch in range(1, epochs + 1):
        model.train()
        total_loss = 0.0
        n_batches  = 0

        for signals, labels in train_loader:
            signals = signals.to(device)
            labels  = labels.to(device)

            optimizer.zero_grad()
            logits = model(signals)
            loss   = criterion(logits, labels)
            loss.backward()
            nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()

            total_loss += loss.item()
            n_batches  += 1

        avg_loss = total_loss / max(n_batches, 1)
        scheduler.step(avg_loss)

        if epoch % 10 == 0:
            log.info(f"  BiLSTM epoch {epoch:3d}/{epochs} | loss={avg_loss:.5f}")

    train_time = time.perf_counter() - t0
    log.info(f"  Training time: {train_time:.1f}s")

    # ---- Evaluation -------------------------------------------------------
    model.eval()
    all_preds, all_labels = [], []

    with torch.no_grad():
        for signals, labels in test_loader:
            logits = model(signals.to(device))
            preds  = logits.argmax(dim=1).cpu().numpy()
            all_preds.extend(preds)
            all_labels.extend(labels.numpy())

    y_pred = np.array(all_preds)
    y_true = np.array(all_labels)

    metrics = compute_metrics(y_true, y_pred, patient_ids_test)
    metrics["Algorithm"]    = "Supervised BiLSTM"
    metrics["Train time (s)"] = round(train_time, 1)

    _log_metrics("BiLSTM (alg6)", metrics)
    return metrics


# ===========================================================================
# Algorithm 7 — Feature-based K-Means
# ===========================================================================

def extract_features(signals: np.ndarray) -> np.ndarray:
    """
    Extracts hand-crafted statistical features per window for clustering.

    Input  : (N, C, T)
    Output : (N, F)  where F = C * 8 features per channel

    Features per channel:
      mean, std, min, max, range, skewness (approx), kurtosis (approx),
      zero_crossings, mean_abs_diff (mean gradient magnitude)
    """
    N, C, T = signals.shape
    features = []

    for c in range(C):
        ch = signals[:, c, :]   # (N, T)

        mean_val  = ch.mean(axis=1)
        std_val   = ch.std(axis=1)
        min_val   = ch.min(axis=1)
        max_val   = ch.max(axis=1)
        rng_val   = max_val - min_val

        # Approximate skewness: (mean - median) / std
        median_val = np.median(ch, axis=1)
        skew_val   = (mean_val - median_val) / (std_val + 1e-8)

        # Approximate kurtosis: ratio of 4th moment to variance^2
        centered   = ch - mean_val[:, None]
        kurt_val   = (centered ** 4).mean(axis=1) / ((std_val ** 2 + 1e-8) ** 2)

        # Zero crossings around mean
        centered_signed = np.sign(centered)
        zc = (np.diff(centered_signed, axis=1) != 0).sum(axis=1).astype(float)

        # Mean absolute gradient
        grad_mag = np.abs(np.diff(ch, axis=1)).mean(axis=1)

        features.extend([
            mean_val, std_val, min_val, max_val,
            rng_val, skew_val, kurt_val, zc, grad_mag,
        ])

    return np.column_stack(features)   # (N, C*9)


def run_feature_kmeans(
    train_ds: ThermalNPYDataset,
    test_ds:  ThermalNPYDataset,
    n_clusters: int = 2,
    patient_ids_test: Optional[np.ndarray] = None,
) -> Dict:
    """
    Feature-based K-Means baseline (Algorithm 7).

    Strategy:
      1. Extract statistical features from training windows.
      2. Fit K-Means (k=2) on healthy training windows only.
      3. Assign cluster labels: the cluster with higher mean reconstruction
         distance from healthy centroid is flagged as 'anomaly'.
      4. Evaluate on test set.
    """
    log.info("=" * 60)
    log.info("BASELINE alg7 — Feature-based K-Means")
    log.info("=" * 60)

    t0 = time.perf_counter()

    # Extract features
    train_features = extract_features(train_ds.signals)   # (N_train, F)
    test_features  = extract_features(test_ds.signals)    # (N_test, F)

    # Standardise features (critical for K-Means convergence)
    scaler = StandardScaler()
    train_features_scaled = scaler.fit_transform(train_features)
    test_features_scaled  = scaler.transform(test_features)

    # Fit K-Means on ALL training windows (supervised by cluster assignment)
    kmeans = KMeans(n_clusters=n_clusters, random_state=SEED, n_init=10)
    train_cluster_labels = kmeans.fit_predict(train_features_scaled)

    # Identify which cluster corresponds to 'anomaly':
    # The cluster with a higher proportion of anomalous training windows
    cluster_anomaly_ratios = []
    for k in range(n_clusters):
        mask  = (train_cluster_labels == k)
        ratio = train_ds.labels[mask].mean() if mask.sum() > 0 else 0.0
        cluster_anomaly_ratios.append(ratio)

    anomaly_cluster = int(np.argmax(cluster_anomaly_ratios))
    log.info(
        f"  Cluster anomaly ratios: {[f'{r:.2f}' for r in cluster_anomaly_ratios]} "
        f"→ anomaly cluster = {anomaly_cluster}"
    )

    # Predict on test set
    test_cluster_labels = kmeans.predict(test_features_scaled)
    y_pred = (test_cluster_labels == anomaly_cluster).astype(int)
    y_true = test_ds.labels

    train_time = time.perf_counter() - t0
    log.info(f"  Training + inference time: {train_time:.2f}s")

    metrics = compute_metrics(y_true, y_pred, patient_ids_test)
    metrics["Algorithm"]      = "Feature K-Means"
    metrics["Train time (s)"] = round(train_time, 2)

    _log_metrics("Feature K-Means (alg7)", metrics)
    return metrics


# ===========================================================================
# Algorithm 8 — TS-KMeans with DTW
# ===========================================================================

def run_tskmeans_dtw(
    train_ds: ThermalNPYDataset,
    test_ds:  ThermalNPYDataset,
    n_clusters: int = 2,
    max_iter:   int = 10,
    n_init:     int = 2,
    patient_ids_test: Optional[np.ndarray] = None,
    subsample:  Optional[int] = 5000,
) -> Dict:
    """
    TS-KMeans with DTW baseline (Algorithm 8).

    Uses tslearn.clustering.TimeSeriesKMeans with metric='dtw'.

    Note: DTW is O(T^2) per pair → very slow on large datasets.
          subsample=5000 limits training to 5000 windows (stratified).

    Parameters
    ----------
    subsample : int or None
        Max training windows to use. None = use all (can take hours).
    """
    try:
        from tslearn.clustering import TimeSeriesKMeans
    except ImportError:
        log.error(
            "tslearn not installed. Run: pip install tslearn\n"
            "TSKMeans baseline skipped."
        )
        return {"Algorithm": "TS-KMeans DTW", "error": "tslearn not installed"}

    log.info("=" * 60)
    log.info("BASELINE alg8 — TS-KMeans DTW")
    log.info("=" * 60)

    t0 = time.perf_counter()

    # tslearn expects (N, T, C) — we have (N, C, T) so transpose
    train_signals = train_ds.signals.transpose(0, 2, 1)   # (N, T, C)
    test_signals  = test_ds.signals.transpose(0, 2, 1)    # (N, T, C)
    train_labels  = train_ds.labels
    test_labels   = test_ds.labels

    # Optional stratified subsampling for speed (keeps class ratio)
    if subsample is not None and len(train_labels) > subsample:
        rng = np.random.default_rng(SEED)
        idx_0 = np.where(train_labels == 0)[0]
        idx_1 = np.where(train_labels == 1)[0]

        # Maintain original class ratio in subsample
        n1 = max(1, int(subsample * len(idx_1) / len(train_labels)))
        n0 = subsample - n1

        chosen_0 = rng.choice(idx_0, size=min(n0, len(idx_0)), replace=False)
        chosen_1 = rng.choice(idx_1, size=min(n1, len(idx_1)), replace=False)
        chosen   = np.concatenate([chosen_0, chosen_1])
        rng.shuffle(chosen)

        train_signals = train_signals[chosen]
        train_labels  = train_labels[chosen]
        log.info(
            f"  Subsampled training to {len(train_labels)} windows "
            f"({(train_labels==1).sum()} anomalies)"
        )

    log.info(
        f"  Fitting TimeSeriesKMeans (k={n_clusters}, DTW, "
        f"max_iter={max_iter}, n_init={n_init}) on {len(train_labels)} windows..."
    )
    log.info("  This may take several minutes. Please wait...")

    model = TimeSeriesKMeans(
        n_clusters = n_clusters,
        metric     = "dtw",
        max_iter   = max_iter,
        n_init     = n_init,
        random_state = SEED,
        verbose    = 0,
        n_jobs     = -1,   # use all CPU cores
    )
    train_cluster_labels = model.fit_predict(train_signals)

    # Identify anomaly cluster (same logic as Feature K-Means)
    cluster_anomaly_ratios = []
    for k in range(n_clusters):
        mask  = (train_cluster_labels == k)
        ratio = train_labels[mask].mean() if mask.sum() > 0 else 0.0
        cluster_anomaly_ratios.append(ratio)

    anomaly_cluster = int(np.argmax(cluster_anomaly_ratios))
    log.info(
        f"  Cluster anomaly ratios: {[f'{r:.2f}' for r in cluster_anomaly_ratios]} "
        f"→ anomaly cluster = {anomaly_cluster}"
    )

    # Predict on test set
    log.info(f"  Predicting on {len(test_labels)} test windows...")
    test_cluster_labels = model.predict(test_signals)
    y_pred = (test_cluster_labels == anomaly_cluster).astype(int)
    y_true = test_labels

    train_time = time.perf_counter() - t0
    log.info(f"  Total time: {train_time:.1f}s ({train_time/60:.1f} min)")

    metrics = compute_metrics(y_true, y_pred, patient_ids_test)
    metrics["Algorithm"]      = "TS-KMeans DTW"
    metrics["Train time (s)"] = round(train_time, 1)

    _log_metrics("TS-KMeans DTW (alg8)", metrics)
    return metrics


# ===========================================================================
# Logging helper
# ===========================================================================

def _log_metrics(name: str, metrics: Dict):
    log.info(
        f"  {name} results:\n"
        f"    F1           : {metrics.get('F1 (%)', 'N/A')}%\n"
        f"    Precision    : {metrics.get('Precision (%)', 'N/A')}%\n"
        f"    Recall       : {metrics.get('Recall (%)', 'N/A')}%\n"
        f"    Individual   : {metrics.get('Individual Acc. (%)', 'N/A')}%"
    )


# ===========================================================================
# Table I printer
# ===========================================================================

def print_table_i(results: List[Dict]):
    """Prints the 3 baseline rows in Table I format."""
    print("\n" + "=" * 80)
    print("TABLE I (partial) — Anomaly Detection Performance — Baseline Results")
    print("=" * 80)
    header = (
        f"{'Algorithm':<25} {'F1 (%)':>8} {'Precision (%)':>14} "
        f"{'Recall (%)':>11} {'Indiv. Acc. (%)':>16}"
    )
    print(header)
    print("-" * 80)
    for r in results:
        if "error" in r:
            print(f"  {r['Algorithm']:<23}  ERROR: {r['error']}")
            continue
        print(
            f"  {r['Algorithm']:<23} "
            f"{r.get('F1 (%)', '-'):>8} "
            f"{r.get('Precision (%)', '-'):>14} "
            f"{r.get('Recall (%)', '-'):>11} "
            f"{r.get('Individual Acc. (%)', '-'):>16}"
        )
    print("=" * 80)
    print("Note: MedAttnAID (TAAE) row to be added by the detector evaluation.\n")


def save_table_i_csv(results: List[Dict], out_path: Path):
    """Saves Table I baseline rows as CSV for other evaluation code to append TAAE row."""
    columns = ["Algorithm", "F1 (%)", "Precision (%)", "Recall (%)", "Individual Acc. (%)"]
    rows = []
    for r in results:
        if "error" not in r:
            rows.append({col: r.get(col, "") for col in columns})

    with open(out_path, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=columns)
        writer.writeheader()
        writer.writerows(rows)

    log.info(f"Table I (baselines) saved → {out_path}")


# ===========================================================================
# CLI entry point
# ===========================================================================

def parse_args():
    p = argparse.ArgumentParser(
        description=" Task 1 — Baseline algorithms for Table I"
    )
    p.add_argument(
        "--npy-dir",
        default="../Data-Wrangling/etl_output/npy",
        help="Path to the NPY directory from  ETL (default: %(default)s)",
    )
    p.add_argument(
        "--patients",
        nargs="+",
        type=int,
        default=None,
        help="Patient IDs to use (default: auto-detect all in npy-dir)",
    )
    p.add_argument(
        "--epochs",
        type=int,
        default=30,
        help="Training epochs for BiLSTM (default: 30)",
    )
    p.add_argument(
        "--batch-size",
        type=int,
        default=256,
        help="Batch size for BiLSTM training (default: 256)",
    )
    p.add_argument(
        "--dtw-subsample",
        type=int,
        default=5000,
        help="Max training windows for TSKMeans DTW (default: 5000, set 0 for all)",
    )
    p.add_argument(
        "--skip-tskmeans",
        action="store_true",
        help="Skip TSKMeans DTW (useful when dataset is very large)",
    )
    p.add_argument(
        "--save-csv",
        action="store_true",
        help="Save Table I rows to results/table_I_baselines.csv",
    )
    p.add_argument(
        "--out-dir",
        default="results",
        help="Output directory for CSV results (default: results)",
    )
    return p.parse_args()


def auto_detect_patients(npy_dir: Path) -> List[int]:
    """Scan npy_dir for patient_XX_windows.npy and return found patient IDs."""
    found = sorted([
        int(p.stem.split("_")[1])
        for p in npy_dir.glob("patient_*_windows.npy")
    ])
    if not found:
        raise RuntimeError(
            f"No patient_XX_windows.npy files found in {npy_dir}.\n"
            "Check --npy-dir path."
        )
    return found


def main():
    args    = parse_args()
    npy_dir = Path(args.npy_dir)
    out_dir = Path(args.out_dir)
    out_dir.mkdir(exist_ok=True)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    log.info(f"Device: {device}")

    # Auto-detect patients if not specified
    patient_ids = args.patients or auto_detect_patients(npy_dir)
    log.info(f"Using patients: {patient_ids}")

    if len(patient_ids) < 3:
        log.warning(
            f"Only {len(patient_ids)} patient(s) found. "
            "Patient-level split requires at least 3 patients for meaningful "
            "train/val/test. Results may be unreliable."
        )

    # Build datasets
    train_ds, val_ds, test_ds = make_splits(npy_dir, patient_ids)

    # Build per-window patient ID arrays for Individual Accuracy
    # Reconstruct from per-patient datasets
    def get_patient_ids_array(npy_dir: Path, patient_ids: List[int]) -> np.ndarray:
        """Returns array of shape (N,) with patient_id for each window."""
        ids = []
        for pid in patient_ids:
            meta_path = npy_dir / f"patient_{pid:02d}_windows_meta.csv"
            if meta_path.exists():
                meta = pd.read_csv(meta_path)
                ids.extend([pid] * len(meta))
        return np.array(ids)

    # Recover which patients belong to test split (from make_splits determinism)
    rng = np.random.default_rng(SEED)
    ids_shuffled = np.array(patient_ids)
    rng.shuffle(ids_shuffled)
    n_train = max(1, int(len(patient_ids) * 0.70))
    n_val   = max(1, int(len(patient_ids) * 0.15))
    test_patient_ids = ids_shuffled[n_train + n_val:].tolist()
    patient_ids_test = get_patient_ids_array(npy_dir, test_patient_ids)

    # Pad to match test_ds length if needed
    if len(patient_ids_test) != len(test_ds):
        log.warning(
            "patient_ids_test length mismatch — Individual Accuracy "
            "will use window-level accuracy instead."
        )
        patient_ids_test = None

    # Run baselines
    results = []
    dtw_subsample = args.dtw_subsample if args.dtw_subsample > 0 else None

    # alg6 — Supervised BiLSTM
    r_bilstm = run_bilstm(
        train_ds         = train_ds,
        test_ds          = test_ds,
        epochs           = args.epochs,
        batch_size       = args.batch_size,
        device           = device,
        patient_ids_test = patient_ids_test,
    )
    results.append(r_bilstm)

    # alg7 — Feature K-Means
    r_kmeans = run_feature_kmeans(
        train_ds         = train_ds,
        test_ds          = test_ds,
        patient_ids_test = patient_ids_test,
    )
    results.append(r_kmeans)

    # alg8 — TS-KMeans DTW (optional)
    if not args.skip_tskmeans:
        r_tskmeans = run_tskmeans_dtw(
            train_ds         = train_ds,
            test_ds          = test_ds,
            subsample        = dtw_subsample,
            patient_ids_test = patient_ids_test,
        )
        results.append(r_tskmeans)
    else:
        log.info("TSKMeans DTW skipped (--skip-tskmeans flag set).")

    # Print and optionally save Table I
    print_table_i(results)

    if args.save_csv:
        csv_path = out_dir / "table_I_baselines.csv"
        save_table_i_csv(results, csv_path)

    return results


if __name__ == "__main__":
    main()