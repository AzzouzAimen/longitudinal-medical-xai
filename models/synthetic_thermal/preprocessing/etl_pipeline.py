"""
etl_pipeline.py

Data wrangling ETL pipeline
Topic M6: Synthetic Thermal Time-Series Generation

Purpose : Transform raw patient CSV files into cleaned, windowed, normalized
          CSV and NumPy (.npy) files ready for dual-DB insertion (T5) and
          model training.

Pipeline stages
---------------
  Stage 1 – CSV Parsing & Schema Validation
  Stage 2 – Gap Detection & Imputation
  Stage 3 – Derived-Field Computation
  Stage 4 – Per-Patient Z-Score Normalization
  Stage 5 – Windowing & Labeling (60 s windows, 50 % overlap)
  Stage 6 – Export (CSV + NPY) + Transformation Log

Usage
-----
  python etl_pipeline.py [--data-dir PATH] [--out-dir PATH] [--patients N]

Defaults
  --data-dir  ./data/raw
  --out-dir   ./etl_output
  --patients  20  (set to a smaller number for quick testing)
"""

import os
import csv
import json
import time
import logging
import argparse
import datetime
from pathlib import Path

import numpy as np
import pandas as pd

# Constants

EXPECTED_COLUMNS   = ["timestamp", "patient_id", "left_temperature",
                       "right_temperature", "anomaly_label", "anomaly_type"]
TEMP_MIN           = 30.0      # °C : hard lower bound
TEMP_MAX           = 37.5      # °C :hard upper bound
SAMPLING_RATE_HZ   = 1
WINDOW_SIZE        = 60        # seconds
WINDOW_STEP        = 30        # seconds (50 % overlap)
GAP_INTERP_MAX     = 5         # ≤ 5 s : linear interpolation
GAP_FFILL_MAX      = 60        # 6–60 s : forward-fill + warning
VALID_LABEL_PAIRS  = {(0, "none"), (1, "spike"), (1, "drift")}
BATCH_SIZE         = 10_000

# Logging setup
def setup_logging(out_dir: Path) -> logging.Logger:
    ts = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
    log_path = out_dir / "logs" / f"etl_run_{ts}.log"
    log_path.parent.mkdir(parents=True, exist_ok=True)

    logger = logging.getLogger("etl")
    logger.setLevel(logging.DEBUG)

    fh = logging.FileHandler(log_path)
    fh.setLevel(logging.DEBUG)
    ch = logging.StreamHandler()
    ch.setLevel(logging.INFO)

    fmt = logging.Formatter("%(asctime)s [%(levelname)s] %(message)s",
                            datefmt="%H:%M:%S")
    fh.setFormatter(fmt)
    ch.setFormatter(fmt)
    logger.addHandler(fh)
    logger.addHandler(ch)
    return logger

# Stage 1 : CSV Parsing & Schema Validation

def validate_file(path: Path, logger: logging.Logger) -> bool:
    """Return True if file passes header validation; log and return False otherwise."""
    with open(path, newline="", encoding="utf-8") as f:
        reader = csv.reader(f)
        try:
            header = next(reader)
        except StopIteration:
            logger.error(f"REJECT {path.name}: empty file")
            return False

    header = [h.strip() for h in header]
    missing = set(EXPECTED_COLUMNS) - set(header)
    if missing:
        logger.error(f"REJECT {path.name}: missing columns {missing}")
        return False
    if header != EXPECTED_COLUMNS:
        logger.warning(f"WARN {path.name}: column order mismatch we use auto-mapping")
    return True


def parse_csv(path: Path, logger: logging.Logger) -> pd.DataFrame | None:
    """Load CSV, cast types, validate label coherence. Returns None on rejection."""
    try:
        df = pd.read_csv(path, parse_dates=["timestamp"],
                         dtype={
                             "patient_id": "Int32",
                             "anomaly_label": "Int8",
                             "anomaly_type": str
                         })
    except Exception as e:
        logger.error(f"REJECT {path.name}: parse error {e}")
        return None

    # Strip whitespace on string columns
    df["anomaly_type"] = df["anomaly_type"].str.strip().str.lower()

    # Cast temperatures to float64
    for col in ("left_temperature", "right_temperature"):
        df[col] = pd.to_numeric(df[col], errors="coerce").round(4)

    # Label coherence check (file-level rejection)
    invalid_mask = ~df.apply(
        lambda r: (int(r["anomaly_label"]), r["anomaly_type"]) in VALID_LABEL_PAIRS,
        axis=1
    )
    if invalid_mask.any():
        bad = df[invalid_mask][["anomaly_label", "anomaly_type"]].drop_duplicates()
        logger.error(
            f"REJECT {path.name}: invalid (label, type) combos found:\n{bad.to_string()}"
        )
        return None

    return df

# Stage 2 : Gap Detection & Imputation

def handle_gaps(df: pd.DataFrame,
                logger: logging.Logger,
                patient_id: int) -> pd.DataFrame:
    """
    Detect timestamp gaps and apply tiered gap policy:
      ≤ 5 s : linear interpolation (is_interpolated = True)
      6-60 s : forward-fill + warning (is_interpolated = True)
      > 60 s  : session split (segment_id += 1)

    Returns enriched DataFrame with is_interpolated and segment_id columns.
    """
    df = df.sort_values("timestamp").reset_index(drop=True)
    df["delta_t"] = df["timestamp"].diff().dt.total_seconds().fillna(1.0)
    df["is_interpolated"] = False
    df["segment_id"] = 0

    gap_rows = df[df["delta_t"] > 1.01]   # allow ±0.01 s jitter

    if gap_rows.empty:
        logger.debug(f"  patient_{patient_id:02d}: no gaps detected")
        df.drop(columns=["delta_t"], inplace=True)
        return df

    current_segment = 0
    interp_count    = 0
    ffill_count     = 0
    split_count     = 0

    for idx in gap_rows.index:
        gap_s = df.at[idx, "delta_t"]

        if gap_s <= GAP_INTERP_MAX:
            # Linear interpolation for temperature columns
            for col in ("left_temperature", "right_temperature"):
                df.at[idx, col] = (df.at[idx - 1, col] + df.at[idx, col]) / 2.0
            df.at[idx, "is_interpolated"] = True
            interp_count += 1

        elif gap_s <= GAP_FFILL_MAX:
            # Forward-fill
            for col in ("left_temperature", "right_temperature"):
                df.at[idx, col] = df.at[idx - 1, col]
            df.at[idx, "is_interpolated"] = True
            logger.warning(
                f"  patient_{patient_id:02d}: forward-fill gap of {gap_s:.1f}s "
                f"at {df.at[idx, 'timestamp']}"
            )
            ffill_count += 1

        else:
            # New segment
            current_segment += 1
            df.loc[idx:, "segment_id"] = current_segment
            logger.warning(
                f"  patient_{patient_id:02d}: gap {gap_s:.1f}s → "
                f"new segment {current_segment} at {df.at[idx, 'timestamp']}"
            )
            split_count += 1

    logger.info(
        f"  patient_{patient_id:02d}: gaps → interp={interp_count}, "
        f"ffill={ffill_count}, splits={split_count}"
    )
    df.drop(columns=["delta_t"], inplace=True)
    return df

# Stage 3 : Derived Fields

def add_derived_fields(df: pd.DataFrame) -> pd.DataFrame:
    """Add temp_asymmetry and session_second per segment."""
    df["temp_asymmetry"] = (
        df["right_temperature"] - df["left_temperature"]
    ).round(4)

    # session_second: 0-indexed row position within each segment
    df["session_second"] = df.groupby("segment_id").cumcount()
    return df

# Stage 4 : Per-Patient Z-Score Normalization

def normalize_patient(df: pd.DataFrame) -> tuple[pd.DataFrame, dict]:
    """
    Z-score normalization computed over all 259,200 samples (before windowing).
    Stores mean and std so predictions can be inverse-transformed.
    Returns (normalized_df, norm_params_dict).
    """
    params = {}
    for side in ("left", "right"):
        col = f"{side}_temperature"
        mu  = df[col].mean()
        std = df[col].std(ddof=0)
        std = std if std > 0 else 1.0    # guard against zero-variance
        df[f"{col}_norm"] = ((df[col] - mu) / std).round(6)
        params[f"norm_mean_{side}"] = round(float(mu), 6)
        params[f"norm_std_{side}"]  = round(float(std), 6)
    return df, params

# Stage 5 : Windowing & Labeling

def create_windows(df: pd.DataFrame,
                   patient_id: int) -> list[dict]:
    """
    Cut each segment into WINDOW_SIZE-second windows with WINDOW_STEP overlap.
    Label = 1 if ≥ 1 anomalous sample in the window (conservative rule).
    """
    windows = []
    window_index = 0

    for seg_id, seg_df in df.groupby("segment_id"):
        seg_df = seg_df.reset_index(drop=True)
        n = len(seg_df)

        start = 0
        while start + WINDOW_SIZE <= n:
            end = start + WINDOW_SIZE
            window = seg_df.iloc[start:end]

            anomaly_count = int((window["anomaly_label"] == 1).sum())
            label         = 1 if anomaly_count > 0 else 0
            anomaly_ratio = round(anomaly_count / WINDOW_SIZE, 4)

            windows.append({
                "window_id":       window_index,
                "patient_id":      patient_id,
                "segment_id":      int(seg_id),
                "window_index":    window_index,
                "window_start":    str(window["timestamp"].iloc[0]),
                "window_end":      str(window["timestamp"].iloc[-1]),
                "label":           label,
                "anomaly_ratio":   anomaly_ratio,
                "is_interpolated": bool(window["is_interpolated"].any()),
                # Raw signal arrays
                "_left":  window["left_temperature"].values.astype(np.float32),
                "_right": window["right_temperature"].values.astype(np.float32),
                "_left_norm":  window["left_temperature_norm"].values.astype(np.float32),
                "_right_norm": window["right_temperature_norm"].values.astype(np.float32),
                "_asym":  window["temp_asymmetry"].values.astype(np.float32),
                "_label_seq": window["anomaly_label"].values.astype(np.int8),
            })
            window_index += 1
            start += WINDOW_STEP

    return windows

# STAGE 6 : Export

def export_cleaned_csv(df: pd.DataFrame, out_dir: Path, patient_id: int) -> Path:
    """Export the cleaned, enriched signal CSV (one file per patient)."""
    out_path = out_dir / "cleaned_csv" / f"patient_{patient_id:02d}_cleaned.csv"
    out_path.parent.mkdir(parents=True, exist_ok=True)

    export_cols = [
        "timestamp", "patient_id",
        "left_temperature", "right_temperature",
        "left_temperature_norm", "right_temperature_norm",
        "temp_asymmetry", "anomaly_label", "anomaly_type",
        "is_interpolated", "segment_id", "session_second"
    ]
    df[export_cols].to_csv(out_path, index=False)
    return out_path


def export_npy(windows: list[dict], out_dir: Path, patient_id: int) -> Path:
    """
    Export windowed data as a .npy file.
    Shape: (num_windows, 5, 60)
      channel 0 : left_temperature (raw)
      channel 1 : right_temperature (raw)
      channel 2 : left_temperature_norm
      channel 3 : right_temperature_norm
      channel 4 : temp_asymmetry
    """
    out_dir_npy = out_dir / "npy"
    out_dir_npy.mkdir(parents=True, exist_ok=True)

    n = len(windows)
    arr = np.zeros((n, 5, WINDOW_SIZE), dtype=np.float32)
    for i, w in enumerate(windows):
        arr[i, 0] = w["_left"]
        arr[i, 1] = w["_right"]
        arr[i, 2] = w["_left_norm"]
        arr[i, 3] = w["_right_norm"]
        arr[i, 4] = w["_asym"]

    npy_path = out_dir_npy / f"patient_{patient_id:02d}_windows.npy"
    np.save(npy_path, arr)

    # Also save per-window metadata as a companion CSV
    meta_cols = ["window_id", "patient_id", "segment_id", "window_index",
                 "window_start", "window_end", "label", "anomaly_ratio",
                 "is_interpolated"]
    meta = [{k: w[k] for k in meta_cols} for w in windows]
    meta_path = out_dir_npy / f"patient_{patient_id:02d}_windows_meta.csv"
    pd.DataFrame(meta).to_csv(meta_path, index=False)

    return npy_path

# Main orchestrator

def run_pipeline(data_dir: Path, out_dir: Path,
                 max_patients: int, logger: logging.Logger) -> dict:
    start_time = time.time()
    summary = {
        "run_timestamp":   datetime.datetime.now().isoformat(),
        "data_dir":        str(data_dir),
        "out_dir":         str(out_dir),
        "patients":        [],
        "total_files":     0,
        "files_accepted":  0,
        "files_rejected":  0,
        "total_rows_in":   0,
        "total_rows_out":  0,
        "total_windows":   0,
        "rows_quarantined":0,
    }

    csv_files = sorted(data_dir.glob("patient_*.csv"))[:max_patients]
    summary["total_files"] = len(csv_files)

    # Global windows accumulator
    all_window_meta = []

    for csv_path in csv_files:
        pid = int(csv_path.stem.split("_")[1])
        logger.info(f"Processing {csv_path.name} (patient {pid:02d})")

        # Stage 1: Validate & Parse
        if not validate_file(csv_path, logger):
            summary["files_rejected"] += 1
            continue

        df = parse_csv(csv_path, logger)
        if df is None:
            summary["files_rejected"] += 1
            continue

        summary["files_accepted"] += 1
        summary["total_rows_in"] += len(df)

        # Stage 2: Gap handling
        df = handle_gaps(df, logger, pid)

        # Stage 3: Derived fields
        df = add_derived_fields(df)

        # Stage 4: Normalization
        df, norm_params = normalize_patient(df)

        # Stage 5: Windowing
        windows = create_windows(df, pid)

        # Stage 6: Export
        csv_out = export_cleaned_csv(df, out_dir, pid)
        npy_out = export_npy(windows, out_dir, pid)

        n_windows = len(windows)
        n_anomaly = sum(1 for w in windows if w["label"] == 1)
        summary["total_rows_out"] += len(df)
        summary["total_windows"]  += n_windows

        patient_record = {
            "patient_id":     pid,
            "rows_input":     len(df),
            "rows_output":    len(df),
            "windows":        n_windows,
            "anomaly_windows":n_anomaly,
            "normal_windows": n_windows - n_anomaly,
            "segments":       int(df["segment_id"].max()) + 1,
            "interpolated_rows": int(df["is_interpolated"].sum()),
            "norm_params":    norm_params,
            "cleaned_csv":    str(csv_out),
            "npy_file":       str(npy_out),
        }
        summary["patients"].append(patient_record)
        all_window_meta += [{k: w[k] for k in
                              ["window_id","patient_id","segment_id","window_index",
                               "window_start","window_end","label","anomaly_ratio",
                               "is_interpolated"]}
                             for w in windows]

        logger.info(
            f"  → rows={len(df)}, windows={n_windows} "
            f"(anomaly={n_anomaly}), segments={patient_record['segments']}"
        )

    # Export combined window metadata
    combined_meta_path = out_dir / "all_windows_meta.csv"
    pd.DataFrame(all_window_meta).to_csv(combined_meta_path, index=False)
    logger.info(f"Combined window metadata → {combined_meta_path}")

    # Write ETL summary JSON
    summary["elapsed_seconds"] = round(time.time() - start_time, 2)
    summary_path = out_dir / "logs" / "etl_summary.json"
    with open(summary_path, "w") as f:
        json.dump(summary, f, indent=2)
    logger.info(f"ETL summary → {summary_path}")
    logger.info(
        f"DONE in {summary['elapsed_seconds']}s | "
        f"accepted={summary['files_accepted']}, "
        f"rejected={summary['files_rejected']}, "
        f"windows={summary['total_windows']}"
    )
    return summary

# Entry point

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Synthetic thermal ETL pipeline")
    parser.add_argument(
        "--data-dir",
        default="data/raw",
        help="Directory containing patient_XX.csv files"
    )
    parser.add_argument(
        "--out-dir", default="etl_output",
        help="Root output directory"
    )
    parser.add_argument(
        "--patients", type=int, default=20,
        help="Maximum number of patient files to process"
    )
    args = parser.parse_args()

    data_dir = Path(args.data_dir)
    out_dir  = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    logger = setup_logging(out_dir)
    logger.info("Synthetic thermal ETL pipeline")
    logger.info(f"Data dir : {data_dir}")
    logger.info(f"Output   : {out_dir}")
    logger.info(f"Patients : {args.patients}")

    run_pipeline(data_dir, out_dir, args.patients, logger)
