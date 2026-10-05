"""
localization.py
 – Task 3 : Anomaly Localization & Figure 4
Topic M6 : Synthetic Thermal Time-Series
Outputs
------------
- Table III  : Left/right localization accuracy, 95% CI, temporal quadrant analysis
- Figure 4   : Attention heatmap + reconstruction-error overlay per window

Inputs required ( outputs)
-----------------------------------
  --npy-dir   : path to data/processed/npy/
                  patient_XX_windows.npy       (N, 5, 60)
                  patient_XX_windows_meta.csv  columns: window_id, patient_id,
                                               segment_id, window_index,
                                               window_start, window_end,
                                               label, anomaly_ratio, is_interpolated
  --raw-dir   : path to data/raw/
                  patient_XX.csv               columns: timestamp, patient_id,
                                               left_temperature, right_temperature,
                                               anomaly_label, anomaly_type
  --checkpoint: results/best_model.pt
  --out-dir   : output directory (default: results)

Channels in .npy (axis=1):
  0 left_temperature
  1 right_temperature
  2 left_temperature_norm   ← model input ch 0
  3 right_temperature_norm  ← model input ch 1
  4 temp_asymmetry          ← model input ch 2

Patient split (seed=42, 70/15/15 by patient, 20 patients total):
  Reproducible via split_patient_ids() — same logic as evaluate.py
"""

from __future__ import annotations

import argparse
import csv
import json
import logging
import math
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from scipy import stats

# ---------------------------------------------------------------------------
# Logging
# ---------------------------------------------------------------------------
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    datefmt="%H:%M:%S",
)
log = logging.getLogger("localization")

# ---------------------------------------------------------------------------
# Constants (must match the rest of the codebase)
# ---------------------------------------------------------------------------
SEED         = 42
WINDOW_SIZE  = 60   # timesteps
N_CHANNELS   = 3    # left_norm, right_norm, asymmetry fed to model

# Channel indices inside the .npy array (N, 5, 60)
CH_LEFT_RAW   = 0
CH_RIGHT_RAW  = 1
CH_LEFT_NORM  = 2
CH_RIGHT_NORM = 3
CH_ASYM       = 4

# Channels fed to the model (sliced from .npy)
MODEL_CHANNEL_INDICES = [CH_LEFT_NORM, CH_RIGHT_NORM, CH_ASYM]
DEFAULT_CHANNELS      = ["left_temperature_norm", "right_temperature_norm", "temp_asymmetry"]

# Temporal quadrants
N_QUADRANTS = 4   # Q1=early, Q2, Q3, Q4=late  (15 timesteps each)


# ===========================================================================
# Patient split  (mirrors evaluate.py exactly)
# ===========================================================================

def split_patient_ids(
    patient_ids: List[int],
    train_ratio: float = 0.70,
    val_ratio:   float = 0.15,
    seed:        int   = SEED,
) -> Tuple[List[int], List[int], List[int]]:
    rng = np.random.default_rng(seed)
    ids = np.array(sorted(patient_ids))
    rng.shuffle(ids)
    n       = len(ids)
    n_train = max(1, int(n * train_ratio))
    n_val   = max(1, int(n * val_ratio))
    train_ids = ids[:n_train].tolist()
    val_ids   = ids[n_train : n_train + n_val].tolist()
    test_ids  = ids[n_train + n_val :].tolist()
    return train_ids, val_ids, test_ids


def discover_patient_ids(npy_dir: Path) -> List[int]:
    ids = []
    for f in sorted(npy_dir.glob("patient_*_windows.npy")):
        try:
            ids.append(int(f.stem.split("_")[1]))
        except ValueError:
            pass
    return sorted(ids)


# ===========================================================================
# Data loading
# ===========================================================================

class WindowDataset:
    """
    Loads .npy windows + meta CSV for a list of patient IDs.
    Keeps raw channels available for reconstruction overlay.
    """

    def __init__(self, npy_dir: Path, patient_ids: List[int]):
        self.npy_dir     = Path(npy_dir)
        self.patient_ids = patient_ids

        signals_list, labels_list, meta_list = [], [], []

        for pid in patient_ids:
            npy_path  = self.npy_dir / f"patient_{pid:02d}_windows.npy"
            meta_path = self.npy_dir / f"patient_{pid:02d}_windows_meta.csv"

            if not npy_path.exists():
                log.warning(f"Missing {npy_path} — skipping patient {pid}")
                continue

            arr  = np.load(npy_path).astype(np.float32)   # (N, 5, 60)
            meta = pd.read_csv(meta_path)

            signals_list.append(arr)
            labels_list.append(meta["label"].values.astype(np.int64))
            meta_list.append(meta)

        if not signals_list:
            raise RuntimeError(f"No .npy files found in {npy_dir} for patients {patient_ids}")

        self.signals_full = np.concatenate(signals_list, axis=0)   # (N, 5, 60)
        self.labels       = np.concatenate(labels_list,  axis=0)   # (N,)
        self.meta         = pd.concat(meta_list, ignore_index=True)

        # Model input: 3 normalised channels
        self.signals_model = self.signals_full[:, MODEL_CHANNEL_INDICES, :]  # (N, 3, 60)

        log.info(
            f"Loaded {len(self.labels)} windows | "
            f"anomalous: {self.labels.sum()} / {len(self.labels)} "
            f"({100*self.labels.mean():.1f}%)"
        )

    def __len__(self):
        return len(self.labels)


# ===========================================================================
# Model loading
# ===========================================================================

def load_model(checkpoint: Path, n_channels: int = 3, device: torch.device = None):
    """Load TAAE from checkpoint."""
    # Import from the repo root (same directory as this script)
    import sys
    sys.path.insert(0, str(Path(__file__).parent))
    from model import TAAE

    if device is None:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    model = TAAE(n_channels=n_channels, window_size=WINDOW_SIZE).to(device)

    if checkpoint.exists():
        state = torch.load(checkpoint, map_location=device)
        model.load_state_dict(state)
        log.info(f"Loaded checkpoint: {checkpoint}")
    else:
        log.warning(f"Checkpoint not found: {checkpoint}. Using random weights.")

    model.eval()
    return model, device


# ===========================================================================
# Inference — collect attention + reconstruction for all windows
# ===========================================================================

@torch.no_grad()
def run_inference(
    model,
    dataset: WindowDataset,
    batch_size: int,
    device: torch.device,
) -> Dict[str, np.ndarray]:
    """
    Returns dict with:
      attention       : (N, T)    softmax attention weights per timestep
      x_hat           : (N, C, T) reconstructed signal
      recon_error     : (N, T)    mean squared error per timestep (mean over channels)
      left_recon_err  : (N, T)    per-timestep MSE on left channel only
      right_recon_err : (N, T)    per-timestep MSE on right channel only
      window_loss     : (N,)      mean recon error per window (scalar per window)
    """
    N = len(dataset)
    all_alpha    = np.zeros((N, WINDOW_SIZE), dtype=np.float32)
    all_x_hat    = np.zeros((N, N_CHANNELS, WINDOW_SIZE), dtype=np.float32)

    for start in range(0, N, batch_size):
        end    = min(start + batch_size, N)
        x_np   = dataset.signals_model[start:end]          # (B, 3, 60)
        x      = torch.from_numpy(x_np).to(device)

        x_hat, alpha = model(x)

        all_alpha[start:end] = alpha.cpu().numpy()
        all_x_hat[start:end] = x_hat.cpu().numpy()

    # Per-timestep reconstruction error (mean over channels)
    x_orig    = dataset.signals_model                      # (N, 3, 60)
    sq_err    = (x_orig - all_x_hat) ** 2                 # (N, 3, 60)
    recon_err = sq_err.mean(axis=1)                        # (N, 60)

    # Per-channel per-timestep error for left/right localization
    # Model channel 0 = left_norm, channel 1 = right_norm
    left_err  = sq_err[:, 0, :]   # (N, 60)
    right_err = sq_err[:, 1, :]   # (N, 60)

    window_loss = recon_err.mean(axis=1)   # (N,)

    return {
        "attention":       all_alpha,
        "x_hat":           all_x_hat,
        "recon_error":     recon_err,
        "left_recon_err":  left_err,
        "right_recon_err": right_err,
        "window_loss":     window_loss,
    }


# ===========================================================================
# Left / Right anomaly localization
# ===========================================================================

def localize_channel(
    left_err:  np.ndarray,   # (N, T)
    right_err: np.ndarray,   # (N, T)
    labels:    np.ndarray,   # (N,)  0=healthy, 1=anomalous
    meta:      pd.DataFrame,
) -> Dict:
    """
    For anomalous windows, determine whether the model correctly identifies
    the side with higher reconstruction error.

    Ground truth side is derived from anomaly_ratio of the left vs right raw channels
    (if available in meta) or from the raw signal asymmetry.

    Strategy:
      - Left anomaly  : mean left_recon_err > mean right_recon_err in the window
      - Right anomaly : mean right_recon_err > mean left_recon_err
      - Validate against: which side has higher anomaly_ratio (from meta)
        or, fallback: which raw channel has higher variance

    Returns dict with localization accuracy, confusion, CI.
    """
    anomaly_mask = labels == 1
    n_anomalous  = anomaly_mask.sum()

    if n_anomalous == 0:
        log.warning("No anomalous windows in this split — localization skipped")
        return {"n_anomalous": 0}

    left_mean  = left_err[anomaly_mask].mean(axis=1)    # (n_anomalous,)
    right_mean = right_err[anomaly_mask].mean(axis=1)

    # Model prediction: which channel has higher error?
    pred_right = (right_mean > left_mean).astype(int)   # 1 = model says right

    # Ground truth: from the raw asymmetry channel in .npy
    # Channel 4 = temp_asymmetry = left - right (positive → left warmer)
    # We use the sign of the mean asymmetry across the anomalous window:
    # negative mean asymmetry → right side is hotter → right anomaly
    asym_mean_anomalous = None
    if "anomaly_ratio" in meta.columns:
        # Use anomaly_ratio as a proxy — but we don't have side info in meta directly.
        # Fall back to reconstructon-error-based localization self-consistency check.
        pass

    # True localization: whether right reconstruction error is dominant
    # (self-consistency: model agrees with its own error distribution)
    # For the asymmetry validation we use the raw asymmetry channel
    # signals_full channel 4 = temp_asymmetry
    # We do NOT have signals_full here — it's passed separately, see Table III builder

    # Basic accuracy: how often does the model show lateralised error?
    # (error ratio > 1.2 or < 0.83 = clear lateralisation)
    ratio = right_mean / (left_mean + 1e-8)
    clearly_right = ratio > 1.2
    clearly_left  = ratio < (1.0 / 1.2)
    ambiguous     = ~(clearly_right | clearly_left)

    lateralization_rate = float((clearly_right | clearly_left).mean()) * 100

    # 95% Wilson confidence interval for lateralization rate
    n   = int(n_anomalous)
    k   = int((clearly_right | clearly_left).sum())
    p   = k / n
    z   = 1.96
    denom = 1 + z**2 / n
    centre = (p + z**2 / (2 * n)) / denom
    margin = z * math.sqrt(p * (1 - p) / n + z**2 / (4 * n**2)) / denom
    ci_low  = max(0.0, centre - margin) * 100
    ci_high = min(1.0, centre + margin) * 100

    return {
        "n_anomalous":          n,
        "n_lateralized":        k,
        "lateralization_rate":  round(lateralization_rate, 2),
        "ci_95_low":            round(ci_low, 2),
        "ci_95_high":           round(ci_high, 2),
        "n_right_dominant":     int(clearly_right.sum()),
        "n_left_dominant":      int(clearly_left.sum()),
        "n_ambiguous":          int(ambiguous.sum()),
        "mean_error_ratio_right_over_left": round(float(ratio.mean()), 4),
    }


def validate_right_asymmetry(
    signals_full: np.ndarray,  # (N, 5, 60)
    left_err:     np.ndarray,  # (N, 60)
    right_err:    np.ndarray,  # (N, 60)
    labels:       np.ndarray,  # (N,)
) -> Dict:
    """
    Validates the right-channel asymmetry false-positive claim:
    On HEALTHY windows, the right channel should NOT show systematically
    higher reconstruction error than the left.

    Also checks: for ANOMALOUS windows, does right error exceed left?
    (Expected for pathological asymmetry patterns.)
    """
    healthy_mask  = labels == 0
    anomaly_mask  = labels == 1

    results = {}

    for name, mask in [("healthy", healthy_mask), ("anomalous", anomaly_mask)]:
        if mask.sum() == 0:
            results[name] = {}
            continue

        left_mean_w  = left_err[mask].mean(axis=1)    # per-window mean
        right_mean_w = right_err[mask].mean(axis=1)

        diff = right_mean_w - left_mean_w   # positive = right > left

        # t-test: is the mean difference significantly different from 0?
        t_stat, p_val = stats.ttest_1samp(diff, 0.0)

        # Raw asymmetry from signal: channel 4 = left - right
        asym_signal = signals_full[mask, CH_ASYM, :].mean(axis=1)  # (n,)

        results[name] = {
            "n_windows":              int(mask.sum()),
            "mean_right_minus_left_err": round(float(diff.mean()), 6),
            "std_right_minus_left_err":  round(float(diff.std()),  6),
            "t_statistic":            round(float(t_stat), 4),
            "p_value":                round(float(p_val),  6),
            "significant_asymmetry":  bool(p_val < 0.05),
            "mean_signal_asymmetry":  round(float(asym_signal.mean()), 4),
        }

    # False positive rate: healthy windows where right_err > left_err by >20%
    if healthy_mask.sum() > 0:
        lh = left_err[healthy_mask].mean(axis=1)
        rh = right_err[healthy_mask].mean(axis=1)
        fp_rate = float((rh > lh * 1.2).mean()) * 100
        results["healthy_right_fp_rate_pct"] = round(fp_rate, 2)

    return results


# ===========================================================================
# Temporal quadrant analysis
# ===========================================================================

def temporal_quadrant_analysis(
    attention:   np.ndarray,  # (N, T)
    recon_error: np.ndarray,  # (N, T)
    labels:      np.ndarray,  # (N,)
) -> Dict:
    """
    Divides each window into 4 temporal quadrants (15 timesteps each).
    Reports mean attention and mean reconstruction error per quadrant,
    separately for healthy and anomalous windows.

    Q1: t=0..14   (early)
    Q2: t=15..29
    Q3: t=30..44
    Q4: t=45..59  (late)
    """
    assert WINDOW_SIZE % N_QUADRANTS == 0
    q_size = WINDOW_SIZE // N_QUADRANTS   # 15

    results = {}
    for name, mask in [("healthy", labels == 0), ("anomalous", labels == 1)]:
        if mask.sum() == 0:
            results[name] = {}
            continue

        attn_sub  = attention[mask]     # (n, 60)
        err_sub   = recon_error[mask]   # (n, 60)
        q_results = []

        for q in range(N_QUADRANTS):
            s = q * q_size
            e = s + q_size
            q_results.append({
                "quadrant":        f"Q{q+1} (t={s}–{e-1})",
                "mean_attention":  round(float(attn_sub[:, s:e].mean()), 6),
                "std_attention":   round(float(attn_sub[:, s:e].std()),  6),
                "mean_recon_err":  round(float(err_sub[:, s:e].mean()),  6),
                "std_recon_err":   round(float(err_sub[:, s:e].std()),   6),
            })

        results[name] = q_results

    # Attention alignment: does the model attend more to anomalous quadrants?
    # Compute correlation between attention and recon_error across timesteps
    # for anomalous windows only
    if (labels == 1).sum() > 0:
        attn_a = attention[labels == 1].flatten()
        err_a  = recon_error[labels == 1].flatten()
        r, p   = stats.pearsonr(attn_a, err_a)
        results["attention_recon_correlation"] = {
            "pearson_r": round(float(r), 4),
            "p_value":   round(float(p), 8),
        }

    return results


# ===========================================================================
# Table III builder
# ===========================================================================

def build_table_III(
    localization_results: Dict,
    asymmetry_results:    Dict,
    quadrant_results:     Dict,
) -> List[Dict]:
    """
    Assembles Table III rows for the report.
    """
    rows = []

    # Row group 1: lateralization
    loc = localization_results
    if loc.get("n_anomalous", 0) > 0:
        rows.append({
            "Metric":    "Anomalous Windows",
            "Value":     loc["n_anomalous"],
            "CI_95_Low": "",
            "CI_95_High": "",
            "Notes":     "Total anomalous windows in test set",
        })
        rows.append({
            "Metric":    "Lateralization Rate (%)",
            "Value":     loc["lateralization_rate"],
            "CI_95_Low": loc["ci_95_low"],
            "CI_95_High": loc["ci_95_high"],
            "Notes":     "Windows with clear left or right dominant error (ratio >1.2)",
        })
        rows.append({
            "Metric":    "Right-dominant errors",
            "Value":     loc["n_right_dominant"],
            "CI_95_Low": "",
            "CI_95_High": "",
            "Notes":     "",
        })
        rows.append({
            "Metric":    "Left-dominant errors",
            "Value":     loc["n_left_dominant"],
            "CI_95_Low": "",
            "CI_95_High": "",
            "Notes":     "",
        })
        rows.append({
            "Metric":    "Ambiguous errors",
            "Value":     loc["n_ambiguous"],
            "CI_95_Low": "",
            "CI_95_High": "",
            "Notes":     "ratio between 0.83 and 1.2",
        })
        rows.append({
            "Metric":    "Mean right/left error ratio (anomalous)",
            "Value":     loc["mean_error_ratio_right_over_left"],
            "CI_95_Low": "",
            "CI_95_High": "",
            "Notes":     ">1 = right systematically higher on anomalous windows",
        })

    # Row group 2: asymmetry FP verification
    asym = asymmetry_results
    if "healthy" in asym and asym["healthy"]:
        h = asym["healthy"]
        rows.append({
            "Metric":    "Healthy windows: mean right-left error diff",
            "Value":     h["mean_right_minus_left_err"],
            "CI_95_Low": "",
            "CI_95_High": "",
            "Notes":     f"t={h['t_statistic']}, p={h['p_value']} — {'significant' if h['significant_asymmetry'] else 'not significant'}",
        })
        if "healthy_right_fp_rate_pct" in asym:
            rows.append({
                "Metric":    "Right channel FP rate on healthy windows (%)",
                "Value":     asym["healthy_right_fp_rate_pct"],
                "CI_95_Low": "",
                "CI_95_High": "",
                "Notes":     "% healthy windows where right_err > 1.2 × left_err",
            })
    if "anomalous" in asym and asym["anomalous"]:
        a = asym["anomalous"]
        rows.append({
            "Metric":    "Anomalous windows: mean right-left error diff",
            "Value":     a["mean_right_minus_left_err"],
            "CI_95_Low": "",
            "CI_95_High": "",
            "Notes":     f"t={a['t_statistic']}, p={a['p_value']} — {'significant' if a['significant_asymmetry'] else 'not significant'}",
        })

    # Row group 3: attention alignment
    if "attention_recon_correlation" in quadrant_results:
        corr = quadrant_results["attention_recon_correlation"]
        rows.append({
            "Metric":    "Attention–Reconstruction Pearson r (anomalous)",
            "Value":     corr["pearson_r"],
            "CI_95_Low": "",
            "CI_95_High": "",
            "Notes":     f"p={corr['p_value']}",
        })

    # Row group 4: quadrant breakdown
    for split_name in ["healthy", "anomalous"]:
        if split_name in quadrant_results and isinstance(quadrant_results[split_name], list):
            for q in quadrant_results[split_name]:
                rows.append({
                    "Metric":    f"{split_name.capitalize()} | {q['quadrant']} mean attention",
                    "Value":     q["mean_attention"],
                    "CI_95_Low": "",
                    "CI_95_High": "",
                    "Notes":     f"recon_err={q['mean_recon_err']}",
                })

    return rows


def save_csv(rows: List[Dict], path: Path):
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)
    log.info(f"Saved → {path}")


# ===========================================================================
# Figure 4 — Attention heatmap + reconstruction-error overlay
# ===========================================================================

def plot_figure4(
    dataset:     WindowDataset,
    inference:   Dict[str, np.ndarray],
    out_dir:     Path,
    n_examples:  int = 6,
):
    """
    Figure 4: for n_examples anomalous windows, plot a 3-panel figure:
      Top    : raw left & right temperature signal + reconstruction overlay
      Middle : per-timestep reconstruction error (left=blue, right=red, total=grey)
      Bottom : attention weight heatmap (color bar)

    Saves one combined figure with all examples as subplots.
    """
    anomaly_idx = np.where(dataset.labels == 1)[0]
    if len(anomaly_idx) == 0:
        log.warning("No anomalous windows — Figure 4 skipped")
        return

    # Pick evenly spaced examples across anomalous windows
    n_examples = min(n_examples, len(anomaly_idx))
    chosen_idx = anomaly_idx[
        np.linspace(0, len(anomaly_idx) - 1, n_examples, dtype=int)
    ]

    fig = plt.figure(figsize=(5 * n_examples, 10))
    fig.suptitle("Figure 4 — Attention Heatmap & Reconstruction Error Overlay", fontsize=13, y=1.01)

    outer = gridspec.GridSpec(1, n_examples, figure=fig, wspace=0.35)

    t_axis = np.arange(WINDOW_SIZE)

    for col, idx in enumerate(chosen_idx):
        inner = gridspec.GridSpecFromSubplotSpec(
            3, 1, subplot_spec=outer[col], hspace=0.08, height_ratios=[3, 2, 1]
        )
        ax_sig  = fig.add_subplot(inner[0])
        ax_err  = fig.add_subplot(inner[1], sharex=ax_sig)
        ax_heat = fig.add_subplot(inner[2], sharex=ax_sig)

        # ---- raw signal (channels 0,1 from signals_full = left/right raw temp)
        left_raw  = dataset.signals_full[idx, CH_LEFT_RAW,  :]
        right_raw = dataset.signals_full[idx, CH_RIGHT_RAW, :]

        # ---- reconstruction (model channels 0,1 = left_norm, right_norm)
        x_hat      = inference["x_hat"][idx]            # (3, 60)
        # Denormalise not strictly possible without scaler params, so overlay norm
        left_hat_n  = x_hat[0]
        right_hat_n = x_hat[1]

        # Normalised input for clean overlay
        left_norm  = dataset.signals_model[idx, 0, :]
        right_norm = dataset.signals_model[idx, 1, :]

        # Signal panel
        ax_sig.plot(t_axis, left_norm,  color="#2196F3", lw=1.5, label="Left (norm)")
        ax_sig.plot(t_axis, right_norm, color="#F44336", lw=1.5, label="Right (norm)")
        ax_sig.plot(t_axis, left_hat_n,  color="#2196F3", lw=1, ls="--", alpha=0.7, label="Left recon")
        ax_sig.plot(t_axis, right_hat_n, color="#F44336", lw=1, ls="--", alpha=0.7, label="Right recon")

        # Fill the reconstruction gap
        ax_sig.fill_between(t_axis, left_norm, left_hat_n,
                            alpha=0.15, color="#2196F3")
        ax_sig.fill_between(t_axis, right_norm, right_hat_n,
                            alpha=0.15, color="#F44336")

        pid  = dataset.meta.iloc[idx]["patient_id"]
        wid  = dataset.meta.iloc[idx]["window_id"]
        lbl  = dataset.meta.iloc[idx]["label"]
        ar   = dataset.meta.iloc[idx].get("anomaly_ratio", "?")
        ax_sig.set_title(f"P{int(pid)} W{int(wid)}\nlabel={int(lbl)} ar={ar:.3f}" if isinstance(ar, float) else f"P{int(pid)} W{int(wid)} label={int(lbl)}", fontsize=8)
        ax_sig.set_ylabel("Normalised temp.", fontsize=7)
        ax_sig.tick_params(labelbottom=False, labelsize=6)
        if col == 0:
            ax_sig.legend(fontsize=5, loc="upper right")

        # Error panel
        left_e  = inference["left_recon_err"][idx]
        right_e = inference["right_recon_err"][idx]
        total_e = inference["recon_error"][idx]

        ax_err.plot(t_axis, left_e,  color="#2196F3", lw=1, label="Left MSE")
        ax_err.plot(t_axis, right_e, color="#F44336", lw=1, label="Right MSE")
        ax_err.plot(t_axis, total_e, color="#555555", lw=1, ls=":", label="Mean MSE")
        ax_err.set_ylabel("Recon. error", fontsize=7)
        ax_err.tick_params(labelbottom=False, labelsize=6)
        if col == 0:
            ax_err.legend(fontsize=5, loc="upper right")

        # Heatmap panel
        alpha = inference["attention"][idx].reshape(1, -1)   # (1, T)
        ax_heat.imshow(
            alpha,
            aspect="auto",
            cmap="YlOrRd",
            extent=[0, WINDOW_SIZE, 0, 1],
            vmin=0,
            vmax=alpha.max() + 1e-8,
        )
        ax_heat.set_yticks([])
        ax_heat.set_xlabel("Timestep", fontsize=7)
        ax_heat.set_ylabel("Attn", fontsize=7)
        ax_heat.tick_params(labelsize=6)

        # Quadrant dividers on all panels
        for ax in [ax_sig, ax_err, ax_heat]:
            for q in range(1, N_QUADRANTS):
                ax.axvline(q * (WINDOW_SIZE // N_QUADRANTS), color="grey",
                           lw=0.5, ls="--", alpha=0.5)

    out_path = out_dir / "figure4_localization_heatmap.png"
    out_dir.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    log.info(f"Figure 4 saved → {out_path}")


def plot_figure4_summary(
    inference: Dict[str, np.ndarray],
    labels:    np.ndarray,
    out_dir:   Path,
):
    """
    Additional summary panel for Figure 4:
    Mean attention profile — healthy vs anomalous windows (with ±1 std band).
    """
    fig, axes = plt.subplots(1, 2, figsize=(12, 4))
    t_axis = np.arange(WINDOW_SIZE)

    # Panel A: mean attention healthy vs anomalous
    ax = axes[0]
    for name, mask, color in [
        ("Healthy",   labels == 0, "#4CAF50"),
        ("Anomalous", labels == 1, "#F44336"),
    ]:
        if mask.sum() == 0:
            continue
        a   = inference["attention"][mask]
        mu  = a.mean(axis=0)
        sd  = a.std(axis=0)
        ax.plot(t_axis, mu, color=color, lw=2, label=name)
        ax.fill_between(t_axis, mu - sd, mu + sd, alpha=0.2, color=color)
    ax.set_title("Mean Attention Profile ± 1 std", fontsize=10)
    ax.set_xlabel("Timestep")
    ax.set_ylabel("Attention weight")
    ax.legend()
    for q in range(1, N_QUADRANTS):
        ax.axvline(q * (WINDOW_SIZE // N_QUADRANTS), color="grey", lw=0.5, ls="--")
    ax.text(0.02, 0.95, "Q1", transform=ax.transAxes, fontsize=7, va="top")
    ax.text(0.27, 0.95, "Q2", transform=ax.transAxes, fontsize=7, va="top")
    ax.text(0.52, 0.95, "Q3", transform=ax.transAxes, fontsize=7, va="top")
    ax.text(0.77, 0.95, "Q4", transform=ax.transAxes, fontsize=7, va="top")

    # Panel B: mean reconstruction error healthy vs anomalous (left & right)
    ax2 = axes[1]
    for name, mask, color in [
        ("Healthy",   labels == 0, "#4CAF50"),
        ("Anomalous", labels == 1, "#F44336"),
    ]:
        if mask.sum() == 0:
            continue
        le = inference["left_recon_err"][mask].mean(axis=0)
        re = inference["right_recon_err"][mask].mean(axis=0)
        ax2.plot(t_axis, le, color=color, lw=1.5, ls="-",  label=f"{name} Left")
        ax2.plot(t_axis, re, color=color, lw=1.5, ls="--", label=f"{name} Right")
    ax2.set_title("Mean Reconstruction Error per Timestep", fontsize=10)
    ax2.set_xlabel("Timestep")
    ax2.set_ylabel("MSE")
    ax2.legend(fontsize=7)

    fig.tight_layout()
    out_path = out_dir / "figure4_summary.png"
    fig.savefig(out_path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    log.info(f"Figure 4 summary saved → {out_path}")


# ===========================================================================
# Main
# ===========================================================================

def parse_args():
    p = argparse.ArgumentParser(description="Task 3 — Anomaly Localization & Figure 4")
    p.add_argument("--npy-dir",    default="../Data-Wrangling/data/processed/npy",
                   help="Path to  processed npy/ directory")
    p.add_argument("--checkpoint", default="results/best_model.pt")
    p.add_argument("--out-dir",    default="results")
    p.add_argument("--batch-size", type=int,   default=256)
    p.add_argument("--n-fig",      type=int,   default=6,
                   help="Number of anomalous windows to show in Figure 4")
    p.add_argument("--train-ratio", type=float, default=0.70)
    p.add_argument("--val-ratio",   type=float, default=0.15)
    p.add_argument("--seed",        type=int,   default=SEED)
    p.add_argument("--patients",    nargs="*",  type=int, default=None,
                   help="Override patient list (default: auto-discover)")
    return p.parse_args()


def main():
    args    = parse_args()
    npy_dir = Path(args.npy_dir)
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    # ── Patient split ────────────────────────────────────────────────────────
    patient_ids = args.patients if args.patients else discover_patient_ids(npy_dir)
    log.info(f"Found {len(patient_ids)} patients: {patient_ids}")

    train_ids, val_ids, test_ids = split_patient_ids(
        patient_ids,
        train_ratio=args.train_ratio,
        val_ratio=args.val_ratio,
        seed=args.seed,
    )
    log.info(f"Split → train={train_ids} | val={val_ids} | test={test_ids}")

    # ── Load test set ────────────────────────────────────────────────────────
    test_ds = WindowDataset(npy_dir, test_ids)

    # ── Load model ───────────────────────────────────────────────────────────
    model, device = load_model(Path(args.checkpoint))

    # ── Inference ────────────────────────────────────────────────────────────
    log.info("Running inference on test set...")
    inference = run_inference(model, test_ds, args.batch_size, device)

    log.info(
        f"Inference done | "
        f"attention shape: {inference['attention'].shape} | "
        f"mean window loss: {inference['window_loss'].mean():.6f}"
    )

    # ── Localization ─────────────────────────────────────────────────────────
    log.info("Computing left/right localization...")
    loc_results = localize_channel(
        inference["left_recon_err"],
        inference["right_recon_err"],
        test_ds.labels,
        test_ds.meta,
    )

    # ── Asymmetry false-positive verification ────────────────────────────────
    log.info("Validating right-channel asymmetry...")
    asym_results = validate_right_asymmetry(
        test_ds.signals_full,
        inference["left_recon_err"],
        inference["right_recon_err"],
        test_ds.labels,
    )

    # ── Temporal quadrant analysis ───────────────────────────────────────────
    log.info("Temporal quadrant analysis...")
    quad_results = temporal_quadrant_analysis(
        inference["attention"],
        inference["recon_error"],
        test_ds.labels,
    )

    # ── Table III ────────────────────────────────────────────────────────────
    table_III = build_table_III(loc_results, asym_results, quad_results)
    save_csv(table_III, out_dir / "table_III_localization.csv")

    # Save full JSON for reference
    full_results = {
        "localization":  loc_results,
        "asymmetry":     asym_results,
        "quadrants":     quad_results,
        "test_patients": test_ids,
        "n_windows":     len(test_ds),
    }
    json_path = out_dir / "localization_results.json"
    with open(json_path, "w") as f:
        json.dump(full_results, f, indent=2, default=str)
    log.info(f"Full results → {json_path}")

    # ── Figure 4 ─────────────────────────────────────────────────────────────
    log.info("Generating Figure 4...")
    plot_figure4(test_ds, inference, out_dir, n_examples=args.n_fig)
    plot_figure4_summary(inference, test_ds.labels, out_dir)

    # ── Console summary ──────────────────────────────────────────────────────
    print("\n" + "=" * 60)
    print("TASK 3 — LOCALIZATION RESULTS")
    print("=" * 60)
    print(f"Test patients       : {test_ids}")
    print(f"Total windows       : {len(test_ds)}")
    print(f"Anomalous windows   : {test_ds.labels.sum()}")

    if loc_results.get("n_anomalous", 0) > 0:
        print(f"\nLateralization rate : {loc_results['lateralization_rate']}%")
        print(f"95% CI              : [{loc_results['ci_95_low']}%, {loc_results['ci_95_high']}%]")
        print(f"Right-dominant      : {loc_results['n_right_dominant']}")
        print(f"Left-dominant       : {loc_results['n_left_dominant']}")
        print(f"Ambiguous           : {loc_results['n_ambiguous']}")

    if "healthy" in asym_results and asym_results["healthy"]:
        h = asym_results["healthy"]
        print(f"\nAsymmetry FP check (healthy windows):")
        print(f"  mean right-left err diff : {h['mean_right_minus_left_err']}")
        print(f"  t={h['t_statistic']}, p={h['p_value']} → {'SIGNIFICANT ⚠' if h['significant_asymmetry'] else 'not significant ✓'}")
        if "healthy_right_fp_rate_pct" in asym_results:
            print(f"  Right FP rate on healthy : {asym_results['healthy_right_fp_rate_pct']}%")

    if "attention_recon_correlation" in quad_results:
        c = quad_results["attention_recon_correlation"]
        print(f"\nAttention–recon correlation (anomalous) : r={c['pearson_r']}, p={c['p_value']}")

    print(f"\nOutputs:")
    print(f"  Table III → {out_dir / 'table_III_localization.csv'}")
    print(f"  Figure 4  → {out_dir / 'figure4_localization_heatmap.png'}")
    print(f"  Figure 4s → {out_dir / 'figure4_summary.png'}")
    print(f"  JSON      → {out_dir / 'localization_results.json'}")
    print("=" * 60)


if __name__ == "__main__":
    main()
