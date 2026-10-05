"""Measure reproducible local Apnea/Sleep fusion-prototype latency."""

from __future__ import annotations

import argparse
import csv
import json
import statistics
import time
from datetime import datetime, timedelta, timezone
from pathlib import Path

import numpy as np
from tensorflow import keras

from pdm.db import connect
from pdm.evidence import compute_channel_evidence
from pdm.model import RuntimeMetadata, TemporalAttention  # register checkpoint layers


ROOT = Path(__file__).resolve().parents[1]
MODEL_ROOT = ROOT / "outputs" / "detector_training"
OUT = ROOT / "outputs" / "inference_latency"
CONFIG = {
    "apnea": {"fold": "a01", "budget_seconds": 60.0, "shape": (100, 7)},
    "sleep": {"fold": "SC401", "budget_seconds": 30.0, "shape": (300, 3)},
}


def _percentile(values: list[float], q: float) -> float:
    return float(np.percentile(np.asarray(values, dtype="float64"), q))


def _summary(values: list[float]) -> dict:
    return {
        "n": len(values), "mean_ms": statistics.fmean(values),
        "median_ms": statistics.median(values),
        "sd_ms": statistics.stdev(values) if len(values) > 1 else 0.0,
        "p95_ms": _percentile(values, 95),
    }


def _load(dataset: str) -> tuple[keras.Model, np.ndarray, dict, float]:
    fold = CONFIG[dataset]["fold"]
    path = MODEL_ROOT / dataset / "folds" / fold
    started = time.perf_counter()
    network = keras.models.load_model(path / "model.keras", compile=False)
    attention = network.get_layer("temporal_attention").output[1]
    inference = keras.Model(network.input, [network.output, attention])
    load_seconds = time.perf_counter() - started
    reconstruction = np.load(path / "held_out_reconstruction.npy", mmap_mode="r")
    residual = np.load(path / "held_out_residual.npy", mmap_mode="r")
    x = np.asarray(reconstruction[:100] + residual[:100], dtype="float32")
    if tuple(x.shape[1:]) != CONFIG[dataset]["shape"]:
        raise ValueError(f"{dataset} input shape differs from the frozen contract")
    thresholds = json.loads((path / "thresholds.json").read_text(encoding="utf-8"))
    return inference, x, thresholds, load_seconds


def _prepare_evidence(dataset: str, window: np.ndarray, reconstruction: np.ndarray,
                      attention: np.ndarray, thresholds: dict) -> dict:
    residual = window - reconstruction
    error = float(np.mean(np.square(residual)))
    evidence = {
        "reconstruction_error": error,
        "residual_mean": float(np.mean(residual)),
        "residual_max_abs": float(np.max(np.abs(residual))),
        "attention_peak_index": int(np.argmax(attention)),
        "attention_max": float(np.max(attention)),
    }
    if dataset == "sleep":
        times = np.asarray([
            datetime(2000, 1, 1, tzinfo=timezone.utc) + timedelta(seconds=index / 10.0)
            for index in range(window.shape[0])
        ], dtype=object)
        evidence["channels"] = compute_channel_evidence(
            window, reconstruction, thresholds["channel_thresholds"], times
        )
    else:
        evidence["channel_mean_squared_residual"] = np.mean(
            np.square(residual), axis=0
        ).astype(float).tolist()
    return evidence


def _persistence_source(cur, dataset: str) -> tuple:
    cur.execute(
        """
        SELECT ps.timestamp,ps.patient_id,ps.recording_id,ps.window_id,ps.prediction_id,
               ps.model_id,ps.reconstruction_error,ps.anomaly_threshold,ps.anomaly_margin,
               ps.normalized_margin,ps.is_anomaly,ps.predicted_class
        FROM prediction_series ps JOIN recording r ON r.recording_id=ps.recording_id
        WHERE r.modality=%s ORDER BY ps.timestamp LIMIT 1
        """,
        (dataset,),
    )
    row = cur.fetchone()
    if not row:
        raise ValueError(f"no {dataset} prediction row available for persistence timing")
    return row


def measure(dataset: str, count: int = 100, warmups: int = 10) -> tuple[list[dict], float]:
    model, windows, thresholds, load_seconds = _load(dataset)
    if count > len(windows):
        raise ValueError("requested count exceeds loaded held-out windows")
    for index in range(min(warmups, count)):
        model(windows[index:index + 1], training=False)
    rows = []
    insert_sql = """
        INSERT INTO prediction_series(
          timestamp,patient_id,recording_id,window_id,prediction_id,model_id,
          reconstruction_error,anomaly_threshold,anomaly_margin,normalized_margin,
          is_anomaly,predicted_class,probability
        ) VALUES(%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,NULL)
    """
    with connect() as conn, conn.cursor() as cur:
        source = _persistence_source(cur, dataset)
        conn.rollback()
        for index in range(count):
            window = windows[index:index + 1]
            total_start = time.perf_counter_ns()
            started = time.perf_counter_ns()
            reconstruction_tensor, attention_tensor = model(window, training=False)
            reconstruction = np.asarray(reconstruction_tensor)[0]
            attention = np.asarray(attention_tensor)[0]
            inference_ms = (time.perf_counter_ns() - started) / 1e6

            started = time.perf_counter_ns()
            evidence = _prepare_evidence(dataset, window[0], reconstruction, attention, thresholds)
            evidence_ms = (time.perf_counter_ns() - started) / 1e6

            persisted = list(source)
            persisted[6] = evidence["reconstruction_error"]
            persisted[8] = float(persisted[6]) - float(persisted[7])
            persisted[9] = float(persisted[8]) / float(persisted[7])
            persisted[10] = bool(float(persisted[6]) > float(persisted[7]))
            persisted[11] = "anomaly" if persisted[10] else "normal"
            started = time.perf_counter_ns()
            cur.execute(insert_sql, tuple(persisted))
            persistence_ms = (time.perf_counter_ns() - started) / 1e6
            total_ms = (time.perf_counter_ns() - total_start) / 1e6
            conn.rollback()
            rows.append({
                "dataset": dataset, "measurement_index": index,
                "window_shape": "x".join(str(value) for value in window.shape[1:]),
                "inference_ms": inference_ms, "evidence_xai_ms": evidence_ms,
                "pdm_persistence_ms": persistence_ms, "total_local_ms": total_ms,
                "streaming_budget_seconds": CONFIG[dataset]["budget_seconds"],
            })
    return rows, load_seconds


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--count", type=int, default=100)
    parser.add_argument("--warmups", type=int, default=10)
    args = parser.parse_args(argv)
    if args.count < 100:
        raise ValueError("the frozen latency protocol requires at least 100 windows")
    all_rows, loads = [], {}
    for dataset in ("apnea", "sleep"):
        rows, load_seconds = measure(dataset, args.count, args.warmups)
        all_rows.extend(rows)
        loads[dataset] = load_seconds
    OUT.mkdir(parents=True, exist_ok=True)
    csv_path = OUT / "raw_measurements.csv"
    with csv_path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(all_rows[0]))
        writer.writeheader(); writer.writerows(all_rows)
    summaries = {}
    for dataset in ("apnea", "sleep"):
        records = [row for row in all_rows if row["dataset"] == dataset]
        summaries[dataset] = {
            "model_load_seconds": loads[dataset],
            "inference": _summary([row["inference_ms"] for row in records]),
            "evidence_xai": _summary([row["evidence_xai_ms"] for row in records]),
            "pdm_persistence": _summary([row["pdm_persistence_ms"] for row in records]),
            "total_local": _summary([row["total_local_ms"] for row in records]),
            "streaming_budget_seconds": CONFIG[dataset]["budget_seconds"],
        }
        summaries[dataset]["p95_safety_factor"] = (
            CONFIG[dataset]["budget_seconds"] * 1000.0
            / summaries[dataset]["total_local"]["p95_ms"]
        )
    (OUT / "summary.json").write_text(json.dumps(summaries, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    lines = [
        "# Fusion prototype latency", "",
        "Local CPU measurements use 10 warm-ups followed by 100 real held-out windows per modality. Model loading is reported separately. Network/LLM latency is excluded.", "",
        "| Dataset | Input | Inference median / p95 | Evidence+XAI median / p95 | Persistence median / p95 | Total median / p95 | Budget | p95 safety factor |",
        "|---|---:|---:|---:|---:|---:|---:|---:|",
    ]
    for dataset in ("apnea", "sleep"):
        item = summaries[dataset]
        lines.append(
            f"| {dataset.title()} | {'100×7' if dataset == 'apnea' else '300×3'} | "
            f"{item['inference']['median_ms']:.3f} / {item['inference']['p95_ms']:.3f} ms | "
            f"{item['evidence_xai']['median_ms']:.3f} / {item['evidence_xai']['p95_ms']:.3f} ms | "
            f"{item['pdm_persistence']['median_ms']:.3f} / {item['pdm_persistence']['p95_ms']:.3f} ms | "
            f"{item['total_local']['median_ms']:.3f} / {item['total_local']['p95_ms']:.3f} ms | "
            f"{item['streaming_budget_seconds']:.0f} s | {item['p95_safety_factor']:.1f}× |"
        )
    lines.extend([
        "", "Persistence timing executes a valid `prediction_series` insert against the real TimescaleDB schema and rolls the transaction back after each measurement, leaving production counts unchanged.",
        "", "These measurements support only the local Apnea-ECG and Sleep-EDF prototypes. They do not replace professor-supplied results for other modalities.",
    ])
    (OUT / "LATENCY_RESULTS.md").write_text("\n".join(lines) + "\n", encoding="utf-8")
    print(json.dumps({"status": "complete", "measurements": len(all_rows), "output": str(OUT)}, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
