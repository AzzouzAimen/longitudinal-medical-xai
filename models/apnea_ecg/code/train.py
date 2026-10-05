"""
Training routines for the MedAttnAID ECG model.

Training notebook sections:
- "Imports, GPU strategy, and global configuration"
- "MedAttnAID training helpers"
- "Main official-split experiment: final MedAttnAID and required ablation"


Notebook dependencies expected from the other extracted files / earlier notebook cells:
- build_medattnaid_model
- make_medattnaid_loss
- LOSS_DISPLAY_NAMES
- subject_clean_normal_window_indices
- evaluate_medattnaid_model
- official_split, X_100, y, recording_ids
"""

# In the notebook these names were already defined in earlier cells. In the
# separate modules, they come from the model and loss Python files.
try:
    from model import build_medattnaid_model
    from loss import make_medattnaid_loss, LOSS_DISPLAY_NAMES
except Exception:
    # Kept only so the file can still be read as a logic extract when the
    # separated files are not next to it.
    pass


# -----------------------------------------------------------------------------
# Notebook global imports and CONFIG used by training
# -----------------------------------------------------------------------------

import os
# Must be set before importing TensorFlow to reduce noisy Kaggle/XLA logs.
os.environ["TF_CPP_MIN_LOG_LEVEL"] = "3"
os.environ["TF_ENABLE_ONEDNN_OPTS"] = "0"
os.environ["AUTOGRAPH_VERBOSITY"] = "0"
os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"

import json
import time
import math
import re
import hashlib
import random
import warnings
import gc
from datetime import datetime
from pathlib import Path
from contextlib import contextmanager

warnings.filterwarnings("ignore")

@contextmanager
def suppress_native_stderr():
    stderr_fd = 2
    saved_stderr_fd = os.dup(stderr_fd)
    devnull_fd = os.open(os.devnull, os.O_WRONLY)
    try:
        os.dup2(devnull_fd, stderr_fd)
        yield
    finally:
        os.dup2(saved_stderr_fd, stderr_fd)
        os.close(saved_stderr_fd)
        os.close(devnull_fd)

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

import scipy
from scipy.signal import find_peaks, welch
from scipy.stats import skew, kurtosis
from sklearn.model_selection import StratifiedGroupKFold
from sklearn.preprocessing import StandardScaler
from sklearn.impute import SimpleImputer
from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    roc_auc_score,
    average_precision_score,
    confusion_matrix,
)
from sklearn.cluster import KMeans
from sklearn.ensemble import RandomForestClassifier
from sklearn.utils.class_weight import compute_class_weight
import joblib

with suppress_native_stderr():
    import tensorflow as tf
    from tensorflow import keras
    from tensorflow.keras import layers

tf.get_logger().setLevel("ERROR")
tf.autograph.set_verbosity(0)
try:
    import absl.logging
    absl.logging.set_verbosity(absl.logging.ERROR)
    absl.logging.set_stderrthreshold("fatal")
except Exception:
    pass

RANDOM_SEED = 42
os.environ["PYTHONHASHSEED"] = str(RANDOM_SEED)
random.seed(RANDOM_SEED)
np.random.seed(RANDOM_SEED)
tf.keras.utils.set_random_seed(RANDOM_SEED)

BASE_DIR = Path("/kaggle/working") if Path("/kaggle/working").exists() else Path(".")
RESULTS_DIR = BASE_DIR / "results"
MODELS_DIR = BASE_DIR / "models"
FIGURES_DIR = BASE_DIR / "figures"
CACHE_DIR = BASE_DIR / "cache"
TABLES_DIR = RESULTS_DIR / "tables"
PRED_DIR = RESULTS_DIR / "predictions"
META_DIR = RESULTS_DIR / "metadata"

for d in [RESULTS_DIR, MODELS_DIR, FIGURES_DIR, CACHE_DIR, TABLES_DIR, PRED_DIR, META_DIR]:
    d.mkdir(parents=True, exist_ok=True)

gpus = tf.config.list_physical_devices("GPU")
for gpu in gpus:
    try:
        tf.config.experimental.set_memory_growth(gpu, True)
    except Exception:
        pass

if len(gpus) >= 2:
    strategy = tf.distribute.MirroredStrategy(cross_device_ops=tf.distribute.NcclAllReduce())
elif len(gpus) == 1:
    strategy = tf.distribute.OneDeviceStrategy(device="/GPU:0")
else:
    strategy = tf.distribute.get_strategy()

REPLICAS = max(1, int(strategy.num_replicas_in_sync))
USE_MIXED_PRECISION = len(gpus) > 0
if USE_MIXED_PRECISION:
    tf.keras.mixed_precision.set_global_policy("mixed_float16")
else:
    tf.keras.mixed_precision.set_global_policy("float32")

CONFIG = {
    "seed": RANDOM_SEED,
    "fs": 100,
    "window_seconds": 60,
    "window_samples": 6000,

    # Final selected MedAttnAID setup:
    # sequence bottleneck + true timestep masking + control-normal training.
    "taae_steps": 100,
    "latent_dim": 16,
    "latent_dim_candidates": [16],
    "attention_dim": 64,
    "encoder_units": 64,
    "decoder_units": 64,

    "medattnaid_architecture": "sequence_masked",
    "medattnaid_train_windows": "control_normal",
    # Final official MedAttnAID mask rate.
    "timestep_mask_rate": 0.25,
    "mask_rate_candidates": [0.25],
    "mask_sweep_loss_mode": "mse_plus_trend",
    "official_medattnaid_loss_mode": "mse_plus_trend",

    # Required PatternLoss ablation modes.
    "medattnaid_loss_modes_to_run": [
        "mse_only",
        "mse_plus_pattern",
        "mse_plus_trend",
        "cploss_full",
    ],

    "batch_size_taae_per_replica": 64,
    "batch_size_bilstm_per_replica": 96,
    "batch_size_taae": 64 * REPLICAS,
    "batch_size_bilstm": 96 * REPLICAS,
    "predict_batch_size": 256 * REPLICAS,

    # Full-run settings for the full experiment.
    # Early stopping controls runtime.
    "max_epochs_taae": 150,
    "max_epochs_bilstm": 100,
    "patience_taae": 25,
    "patience_bilstm": 25,
    "learning_rate": 1e-3,
    "weight_decay": 1e-4,
    "gradient_clipnorm": 1.0,

    # Protocol anomaly thresholding.
    "threshold_percentile": 85,
    "signal_threshold_source": "train_control_healthy",
    # Official MedAttnAID selection is fixed to MSE + Trend at mask_rate=0.25.
    # The balanced score is retained only as metadata.
    "medattnaid_selection_metric": "fixed_mse_plus_trend_mask025",
    "medattnaid_f1_tie_tolerance": 0.005,
    "min_reconstruction_corr": 0.50,
    "min_anom_healthy_ratio": 1.30,
    "collapse_corr_threshold": 0.25,

    # Robust morphology-bin input pipeline for sigmoid decoder.
    "raw_cache_version": "apnea_ecg_raw6000_v2",
    "cache_version": "apnea_ecg_morphology_bins_100step_v2",
    "force_reload_windows_cache": False,
    "use_windows_cache": True,
    "use_processed_tensor_cache": True,
    "input_normalization": "train_control_robust_minmax_morphology_bins",
    "ecg_robust_minmax_lower_q": 1.0,
    "ecg_robust_minmax_upper_q": 99.0,
    # 100-step morphology-bin representation.
    # Channels: mean, std, min, max, range, slope, energy.
    "morphology_feature_names": [
        "mean", "std", "min", "max", "range", "slope", "energy"
    ],
    "morphology_feature_weights": [
        0.10, 0.25, 0.10, 0.10, 0.25, 0.05, 0.15
    ],

    # The winning model uses timestep masking, not Gaussian denoising.
    "use_denoising_training": False,
    "denoising_noise_std": 0.03,
    "denoising_noise_clip": 0.10,


    # Runtime controls.
    "fast_dev_run": False,
    "run_tskmeans": True,
    "tskmeans_fit_limit": 2000,
    "feature_kmeans_top_k": 40,
    "mixed_precision": USE_MIXED_PRECISION,
}

RUN_METADATA = {
    "started_at_utc": datetime.utcnow().isoformat(),
    "random_seed": RANDOM_SEED,
    "python_version": os.sys.version,
    "platform": os.name,
    "numpy_version": np.__version__,
    "pandas_version": pd.__version__,
    "sklearn_version": __import__("sklearn").__version__,
    "scipy_version": scipy.__version__,
    "tensorflow_version": tf.__version__,
    "gpu_devices": [gpu.name for gpu in gpus],
    "strategy_replicas": REPLICAS,
    "base_dir": str(BASE_DIR),
    "cache_dir": str(CACHE_DIR),
    "config": CONFIG,
}

print("[loaded] imports, Kaggle paths, GPU strategy, and rebuilt global configuration")
print(f"Base directory: {BASE_DIR}")
print(f"TensorFlow: {tf.__version__}")
print(f"GPU count: {len(gpus)}")
print(f"Strategy replicas: {REPLICAS}")
print(f"Mixed precision: {USE_MIXED_PRECISION}")
print(f"Effective TAAE batch size: {CONFIG['batch_size_taae']}")
print("MedAttnAID key config:", {k: CONFIG[k] for k in ["taae_steps", "latent_dim_candidates", "input_normalization", "use_denoising_training"]})


# -----------------------------------------------------------------------------
# MedAttnAID training helpers
# -----------------------------------------------------------------------------

@contextmanager
def temporary_config(**kwargs):
    old = {k: CONFIG.get(k) for k in kwargs}
    CONFIG.update(kwargs)
    try:
        yield
    finally:
        for k, v in old.items():
            if v is None and k in CONFIG:
                CONFIG.pop(k, None)
            else:
                CONFIG[k] = v

def medattnaid_safe_global_batch_size(n_train, n_val, replicas=1, preferred_per_replica=64):
    preferred = max(1, int(preferred_per_replica) * max(1, int(replicas)))
    smallest = max(1, min(int(n_train), int(n_val)))
    return int(max(1, min(preferred, smallest)))

def select_medattnaid_train_indices(split):
    train_idx = split["train_idx"]
    val_idx = split["val_idx"]
    mode = CONFIG.get("medattnaid_train_windows", "all_normal")
    if mode == "control_normal":
        train_clean = subject_clean_normal_window_indices(train_idx)
        val_clean = subject_clean_normal_window_indices(val_idx)
    elif mode == "all_normal":
        train_clean = train_idx[y[train_idx] == 0]
        val_clean = val_idx[y[val_idx] == 0]
    else:
        raise ValueError(f"Unknown medattnaid_train_windows mode: {mode}")
    if len(train_clean) == 0 or len(val_clean) == 0:
        raise ValueError(f"No clean train/val windows: train={len(train_clean)}, val={len(val_clean)}")
    if CONFIG.get("fast_dev_run", False):
        train_clean = train_clean[: min(len(train_clean), 1024)]
        val_clean = val_clean[: min(len(val_clean), 512)]
    return np.asarray(train_clean, dtype=int), np.asarray(val_clean, dtype=int)

def add_denoising_noise(X_batch):
    if not CONFIG.get("use_denoising_training", False):
        return X_batch
    noise_std = float(CONFIG.get("denoising_noise_std", 0.03))
    noise_clip = float(CONFIG.get("denoising_noise_clip", 0.10))
    noise = np.random.normal(0.0, noise_std, size=X_batch.shape).astype(np.float32)
    noise = np.clip(noise, -noise_clip, noise_clip)
    return np.clip(X_batch + noise, 0.0, 1.0).astype(np.float32)


def safe_name(text):
    return re.sub(r"[^A-Za-z0-9_.-]+", "_", str(text)).strip("_")

def train_medattnaid_variant(protocol_name, split, X_data, y, variant_display_name, loss_mode, latent_dim=None):
    latent_dim = int(CONFIG.get("latent_dim", 32) if latent_dim is None else latent_dim)
    train_clean, val_clean = select_medattnaid_train_indices(split)
    batch_size = medattnaid_safe_global_batch_size(
        n_train=len(train_clean),
        n_val=len(val_clean),
        replicas=REPLICAS,
        preferred_per_replica=int(CONFIG.get("batch_size_taae_per_replica", 64)),
    )

    X_train_target = X_data[train_clean].astype(np.float32)
    X_val_target = X_data[val_clean].astype(np.float32)
    X_train_input = add_denoising_noise(X_train_target)
    X_val_input = X_val_target

    architecture = str(CONFIG.get("medattnaid_architecture", "global")).lower()
    train_mode = str(CONFIG.get("medattnaid_train_windows", "all_normal")).lower()
    mask_rate = float(CONFIG.get("timestep_mask_rate", 0.0))
    variant_safe = safe_name(variant_display_name)

    model_dir = MODELS_DIR / f"medattnaid_{architecture}_{train_mode}_{protocol_name}_{variant_safe}_{loss_mode}_lat{latent_dim}"
    model_dir.mkdir(parents=True, exist_ok=True)
    ckpt_path = model_dir / "best.weights.h5"

    print("\n" + "=" * 80)
    print(f"Training MedAttnAID variant: {variant_display_name} ({loss_mode}) latent_dim={latent_dim}")
    print(
        f"protocol={protocol_name}, architecture={architecture}, train_mode={train_mode}, "
        f"train clean={len(train_clean)}, val clean={len(val_clean)}, batch={batch_size}, "
        f"denoising={CONFIG.get('use_denoising_training')}, timestep_mask_rate={mask_rate}"
    )

    keras.backend.clear_session()
    gc.collect()
    with strategy.scope():
        model = build_medattnaid_model(
            input_shape=X_data.shape[1:],
            latent_dim=latent_dim,
            attention_dim=int(CONFIG.get("attention_dim", 64)),
            name=f"MedAttnAID_{safe_name(architecture)}_{safe_name(train_mode)}_{safe_name(loss_mode)}_lat{latent_dim}",
        )
        try:
            optimizer = keras.optimizers.AdamW(
                learning_rate=float(CONFIG["learning_rate"]),
                weight_decay=float(CONFIG["weight_decay"]),
                clipnorm=float(CONFIG["gradient_clipnorm"]),
            )
        except Exception:
            optimizer = keras.optimizers.Adam(
                learning_rate=float(CONFIG["learning_rate"]),
                clipnorm=float(CONFIG["gradient_clipnorm"]),
            )
        model.compile(optimizer=optimizer, loss=make_medattnaid_loss(loss_mode))

    callbacks = [
        keras.callbacks.ModelCheckpoint(str(ckpt_path), save_weights_only=True, save_best_only=True, monitor="val_loss", mode="min", verbose=1),
        keras.callbacks.EarlyStopping(monitor="val_loss", patience=int(CONFIG["patience_taae"]), restore_best_weights=True, verbose=1),
        keras.callbacks.ReduceLROnPlateau(monitor="val_loss", factor=0.5, patience=max(5, int(CONFIG["patience_taae"]) // 3), min_lr=1e-6, verbose=1),
    ]

    t0 = time.time()
    history = model.fit(
        X_train_input,
        X_train_target,
        validation_data=(X_val_input, X_val_target),
        epochs=int(CONFIG["max_epochs_taae"]),
        batch_size=batch_size,
        shuffle=True,
        callbacks=callbacks,
        verbose=1,
    )
    seconds = time.time() - t0

    try:
        if ckpt_path.exists():
            model.load_weights(str(ckpt_path))
    except Exception as e:
        print("Warning: could not reload best weights:", repr(e))

    hist = pd.DataFrame(history.history)
    hist_path = model_dir / "history.csv"
    hist.to_csv(hist_path, index=False)
    best_epoch = int(np.argmin(hist["val_loss"].values) + 1) if "val_loss" in hist else len(hist)
    best_val_loss = float(np.nanmin(hist["val_loss"].values)) if "val_loss" in hist else np.nan

    print(f"Finished MedAttnAID '{variant_display_name}': epochs={len(hist)}, best_epoch={best_epoch}, best_val_loss={best_val_loss:.6g}, seconds={seconds:.1f}")

    return {
        "protocol": protocol_name,
        "variant_display_name": variant_display_name,
        "loss_mode": loss_mode,
        "latent_dim": latent_dim,
        "model": model,
        "history": hist,
        "history_path": str(hist_path),
        "model_dir": str(model_dir),
        "best_epoch": best_epoch,
        "best_val_loss": best_val_loss,
        "train_clean_idx": train_clean,
        "val_clean_idx": val_clean,
        "seconds": float(seconds),
        "architecture": architecture,
        "train_mode": train_mode,
        "timestep_mask_rate": mask_rate,
    }

print("[loaded] MedAttnAID training helpers")

# -----------------------------------------------------------------------------
# Final official-split training and required Table IV ablation
# -----------------------------------------------------------------------------

def run_medattnaid_train_eval(protocol_name, split, X_data, loss_mode, latent_dim, variant_display_name=None):
    variant_display_name = variant_display_name or LOSS_DISPLAY_NAMES.get(loss_mode, loss_mode)

    run = train_medattnaid_variant(
        protocol_name,
        split,
        X_data,
        y,
        variant_display_name,
        loss_mode,
        latent_dim=latent_dim,
    )

    eval_run = evaluate_medattnaid_model(
        protocol_name,
        split,
        X_data,
        y,
        recording_ids,
        run["model"],
        variant_display_name,
    )

    run.update(eval_run)
    return run


# -------------------------------------------------------------------------
# Final selected MedAttnAID configuration.
#
#   architecture = sequence_masked
#   train_mode   = control_normal
#   mask_rate    = 0.25
#   final loss   = MSE + Trend
#
# The required four-loss Table IV ablation is run at the same mask rate.
# -------------------------------------------------------------------------
CONFIG.update({
    "medattnaid_architecture": "sequence_masked",
    "medattnaid_train_windows": "control_normal",
    "signal_threshold_source": "train_control_healthy",
    "latent_dim": 16,
    "latent_dim_candidates": [16],
    "use_denoising_training": False,
    "threshold_percentile": 85,
    "patience_taae": 25,

    # Fixed official setting.
    "timestep_mask_rate": 0.25,
    "mask_rate_candidates": [0.25],
    "official_medattnaid_loss_mode": "mse_plus_trend",
    "medattnaid_selection_metric": "fixed_mse_plus_trend_mask025",

    # Required PatternLoss ablation modes for Table IV.
    "medattnaid_loss_modes_to_run": [
        "mse_only",
        "mse_plus_pattern",
        "mse_plus_trend",
        "cploss_full",
    ],
})

print("[config] Final MedAttnAID architecture:", CONFIG["medattnaid_architecture"])
print("[config] Training windows:", CONFIG["medattnaid_train_windows"])
print("[config] Signal threshold source:", CONFIG["signal_threshold_source"])
print("[config] Latent dim:", CONFIG["latent_dim"])
print("[config] Fixed official mask rate:", CONFIG["timestep_mask_rate"])
print("[config] Official MedAttnAID loss mode:", CONFIG["official_medattnaid_loss_mode"])
print("[config] TAAE patience:", CONFIG["patience_taae"])
print("[config] Loss modes for required Table IV:", CONFIG["medattnaid_loss_modes_to_run"])


def _finite_or_neg_inf(value):
    try:
        value = float(value)
        return value if np.isfinite(value) else -np.inf
    except Exception:
        return -np.inf


def _metric_frac(run, key):
    return _finite_or_neg_inf(run.get("test_metrics", {}).get(key, np.nan)) / 100.0


def medattnaid_balanced_selection_score(run):
    """
    Diagnostic score retained for metadata/summaries only.

    The official MedAttnAID row is fixed to MSE + Trend at mask_rate=0.25.
    The score is retained only for metadata and summary diagnostics.
    """
    f1 = _metric_frac(run, "F1 (%)")
    specificity = _metric_frac(run, "Specificity (%)")
    individual_acc = _metric_frac(run, "Individual Acc. (%)")

    diag = run.get("loss_diagnostics", {})
    ratio = _finite_or_neg_inf(diag.get("anomalous_to_healthy_loss_ratio", np.nan))
    corr = _finite_or_neg_inf(diag.get("reconstruction_corr", np.nan))

    ratio_score = max(0.0, min(ratio / 1.50, 1.0)) if np.isfinite(ratio) else 0.0
    corr_score = max(0.0, min(corr / 0.80, 1.0)) if np.isfinite(corr) else 0.0

    score = (
        0.50 * individual_acc
        + 0.20 * specificity
        + 0.15 * f1
        + 0.10 * ratio_score
        + 0.05 * corr_score
    )

    return float(score)


def run_passes_minimum_diagnostics(run):
    diag = run.get("loss_diagnostics", {})
    ratio = _finite_or_neg_inf(diag.get("anomalous_to_healthy_loss_ratio", np.nan))
    corr = _finite_or_neg_inf(diag.get("reconstruction_corr", np.nan))
    collapsed = bool(diag.get("collapsed", False))

    return (
        (not collapsed)
        and corr >= float(CONFIG.get("min_reconstruction_corr", 0.50))
        and ratio >= float(CONFIG.get("min_anom_healthy_ratio", 1.30))
    )


def summarize_medattnaid_runs(runs):
    rows = []
    for r in runs:
        metrics = r.get("test_metrics", {})
        diag = r.get("loss_diagnostics", {})
        rows.append({
            "variant": r.get("variant_display_name"),
            "loss_mode": r.get("loss_mode"),
            "latent_dim": r.get("latent_dim"),
            "mask_rate": r.get("timestep_mask_rate"),
            "best_epoch": r.get("best_epoch"),
            "best_val_loss": r.get("best_val_loss"),
            "F1 (%)": metrics.get("F1 (%)"),
            "Precision (%)": metrics.get("Precision (%)"),
            "Recall (%)": metrics.get("Recall (%)"),
            "Specificity (%)": metrics.get("Specificity (%)"),
            "Individual Acc. (%)": metrics.get("Individual Acc. (%)"),
            "ROC-AUC": metrics.get("ROC-AUC"),
            "PR-AUC": metrics.get("PR-AUC"),
            "anom_healthy_ratio": diag.get("anomalous_to_healthy_loss_ratio"),
            "reconstruction_corr": diag.get("reconstruction_corr"),
            "prediction_std": diag.get("prediction_std"),
            "collapsed": diag.get("collapsed"),
            "max_healthy_anomaly_percentage": diag.get("max_healthy_anomaly_percentage"),
            "min_pathological_anomaly_percentage": diag.get("min_pathological_anomaly_percentage"),
            "individual_percentage_gap": diag.get("individual_percentage_gap"),
            "balanced_selection_score": medattnaid_balanced_selection_score(r),
            "passes_minimum_diagnostics": run_passes_minimum_diagnostics(r),
        })
    return pd.DataFrame(rows)


# -------------------------------------------------------------------------
# Required Table IV ablation at the fixed official mask rate.
# -------------------------------------------------------------------------
fixed_mask_rate = float(CONFIG["timestep_mask_rate"])
medattnaid_runs = []

for loss_mode in CONFIG["medattnaid_loss_modes_to_run"]:
    display_name = LOSS_DISPLAY_NAMES[loss_mode]
    with temporary_config(timestep_mask_rate=fixed_mask_rate):
        medattnaid_runs.append(
            run_medattnaid_train_eval(
                "official",
                official_split,
                X_100,
                loss_mode=loss_mode,
                latent_dim=int(CONFIG["latent_dim"]),
                variant_display_name=display_name,
            )
        )

MEDATTNAID_RUN_SUMMARY = summarize_medattnaid_runs(medattnaid_runs)
MEDATTNAID_RUN_SUMMARY_PATH = RESULTS_DIR / "metadata" / "medattnaid_final_ablation_summary.csv"
MEDATTNAID_RUN_SUMMARY.to_csv(MEDATTNAID_RUN_SUMMARY_PATH, index=False)

print("Final ablation summary saved:", MEDATTNAID_RUN_SUMMARY_PATH)
display(MEDATTNAID_RUN_SUMMARY)


# -------------------------------------------------------------------------
# Final official MedAttnAID selection.
# Use the fixed MSE + Trend model at mask_rate=0.25.
# -------------------------------------------------------------------------
official_loss_mode = str(CONFIG.get("official_medattnaid_loss_mode", "mse_plus_trend"))
matching_final_runs = [
    r for r in medattnaid_runs
    if str(r.get("loss_mode")) == official_loss_mode
    and abs(float(r.get("timestep_mask_rate", fixed_mask_rate)) - fixed_mask_rate) < 1e-9
]

if matching_final_runs:
    final_medattnaid = matching_final_runs[0]
    selection_reason = f"fixed official selection: {official_loss_mode} at mask_rate={fixed_mask_rate:.2f}"
else:
    final_medattnaid = max(medattnaid_runs, key=medattnaid_balanced_selection_score)
    selection_reason = "fallback balanced selection because fixed official loss run was not found"

MEDATTNAID_OFFICIAL_METRICS = final_medattnaid["test_metrics"]

print("Selected final MedAttnAID:", {
    "variant": final_medattnaid.get("variant_display_name"),
    "loss_mode": final_medattnaid.get("loss_mode"),
    "latent_dim": final_medattnaid.get("latent_dim"),
    "architecture": final_medattnaid.get("architecture"),
    "train_mode": final_medattnaid.get("train_mode"),
    "timestep_mask_rate": final_medattnaid.get("timestep_mask_rate"),
    "selection_reason": selection_reason,
    "balanced_selection_score": medattnaid_balanced_selection_score(final_medattnaid),
    "diagnostics": final_medattnaid.get("loss_diagnostics", {}),
    "metrics": final_medattnaid.get("test_metrics", {}),
})

