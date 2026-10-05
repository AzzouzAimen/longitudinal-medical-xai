"""Populate the Phase 6 PDM database from validated fusion LOSO artifacts.

This command intentionally consumes already materialized model outputs.  It
does not train, score, reset, call an LLM, or delete database rows.  The one
authorized volume reset is an operator action performed before invocation.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import sys
from datetime import datetime, timedelta, timezone
from pathlib import Path
from typing import Any

import numpy as np
from psycopg2.extras import execute_values

from pdm.db import connect, connection_kwargs
from pdm.evidence import canonical_json_hash, compute_channel_evidence
from pdm.training import (
    APNEA_PATIENTS,
    DATA_ROOT,
    SLEEP_PATIENTS,
    SLEEP_RECORDINGS,
    _sha256_file,
    load_dataset,
    validate_loso_artifacts,
)
from scripts.prepare_sleep_edf import channel_sampling_rates, meaningful_recording_rate, timestamp_for_recording


ROOT = Path(__file__).resolve().parents[1]
ARTIFACT_ROOT = ROOT / "outputs" / "detector_training"
BASE_TIME = datetime(2000, 1, 1, tzinfo=timezone.utc)
APNEA_FEATURE_CHANNELS = ("mean", "std", "min", "max", "range", "slope", "energy")
SLEEP_MODELED_CHANNELS = ("EEG Fpz-Cz", "EEG Pz-Oz", "EOG horizontal")
SLEEP_AUX_CHANNELS = ("Resp oro-nasal", "EMG submental", "Temp rectal")
AUTHORIZED_DB_ENDPOINT = {"host": "127.0.0.1", "port": 5433, "dbname": "pdm"}


def _json_default(value: Any) -> Any:
    if isinstance(value, (np.integer,)):
        return int(value)
    if isinstance(value, (np.floating,)):
        return float(value)
    if isinstance(value, np.ndarray):
        return value.tolist()
    if isinstance(value, (datetime,)):
        return value.isoformat()
    raise TypeError(f"cannot serialize {type(value)!r}")


def _json_text(payload: Any) -> str:
    return json.dumps(payload, sort_keys=True, separators=(",", ":"), default=_json_default)


def _fail(message: str) -> None:
    raise ValueError(message)


def _load_json(path: Path) -> dict[str, Any]:
    try:
        value = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        raise ValueError(f"invalid JSON artifact: {path}") from exc
    if not isinstance(value, dict):
        raise ValueError(f"artifact must contain an object: {path}")
    return value


def ensure_authorized_database() -> dict[str, Any]:
    """Fail closed unless the destination is the local compose PDM endpoint."""

    try:
        configured = connection_kwargs()
    except (TypeError, ValueError) as exc:
        raise ValueError("invalid configured PDM database endpoint") from exc
    actual = {
        "host": configured.get("host"),
        "port": configured.get("port"),
        "dbname": configured.get("dbname"),
    }
    if actual != AUTHORIZED_DB_ENDPOINT:
        raise ValueError(
            "database destination is not the authorized local compose endpoint "
            f"{AUTHORIZED_DB_ENDPOINT}: resolved {actual}"
        )
    return configured


def validate_artifacts() -> dict[str, Any]:
    """Validate both root manifests and collect exact artifact hashes.

    ``promote=False`` is deliberate: population is read-only with respect to
    the artifact directory and never changes validation markers.
    """

    reports = {}
    for dataset in ("apnea", "sleep"):
        root = ARTIFACT_ROOT / dataset
        result = validate_loso_artifacts(dataset, root, promote=False)
        root_manifest = _load_json(root / "loso_manifest.json")
        if root_manifest.get("artifact_status") != "validated":
            _fail(f"{dataset} root manifest is not validated")
        hashes = {}
        for row in root_manifest["folds"]:
            fold_dir = root / Path(row["fold_manifest"]).parent
            fold_manifest = _load_json(root / row["fold_manifest"])
            if fold_manifest.get("artifact_status") != "validated":
                _fail(f"{dataset}/{row['test_patient']} fold manifest is not validated")
            hashes[row["test_patient"]] = {
                "fold_manifest": row["fold_manifest"],
                "fold_manifest_sha256": _sha256_file(root / row["fold_manifest"]),
                "model_checkpoint": row["model_checkpoint"],
                "model_checkpoint_sha256": _sha256_file(root / row["model_checkpoint"]),
                "artifact_hashes": _load_json(fold_dir / "artifact_hashes.json"),
            }
        reports[dataset] = {"validation": result, "root_manifest": root_manifest, "folds": hashes}
    return reports


def _fold_data(dataset: str, root: Path, row: dict[str, Any]) -> dict[str, Any]:
    fold_dir = root / Path(row["fold_manifest"]).parent
    fold = _load_json(root / row["fold_manifest"])
    decisions = _load_json(fold_dir / "held_out_decisions.json")
    thresholds = _load_json(fold_dir / "thresholds.json")
    normalization = _load_json(fold_dir / "normalization.json")
    evaluation = _load_json(fold_dir / "evaluation_annotations.json")
    arrays = {
        "errors": np.load(fold_dir / "held_out_errors.npy", mmap_mode="r"),
        "reconstruction": np.load(fold_dir / "held_out_reconstruction.npy", mmap_mode="r"),
        "residual": np.load(fold_dir / "held_out_residual.npy", mmap_mode="r"),
        "attention": np.load(fold_dir / "held_out_attention.npy", mmap_mode="r"),
        "channel_errors": np.load(fold_dir / "held_out_channel_errors.npy", mmap_mode="r"),
    }
    count = len(evaluation.get("annotations", []))
    if count == 0:
        _fail(f"empty held-out annotations: {fold_dir}")
    for name, values in decisions.items():
        if isinstance(values, list) and len(values) != count:
            _fail(f"{dataset}/{fold['test_patient']} {name} length mismatch")
    for name, values in arrays.items():
        if values.shape[0] != count:
            _fail(f"{dataset}/{fold['test_patient']} {name} length mismatch")
        if not np.all(np.isfinite(values)):
            _fail(f"{dataset}/{fold['test_patient']} {name} contains non-finite values")
    errors = np.asarray(arrays["errors"], dtype="float64")
    reconstruction = np.asarray(arrays["reconstruction"], dtype="float64")
    residual = np.asarray(arrays["residual"], dtype="float64")
    if reconstruction.ndim != 3 or residual.shape != reconstruction.shape:
        _fail(f"{dataset}/{fold['test_patient']} reconstruction/residual shape mismatch")
    calculated_errors = np.mean(np.square(residual), axis=tuple(range(1, residual.ndim)))
    if not np.allclose(errors, calculated_errors, rtol=2e-5, atol=2e-7):
        _fail(f"{dataset}/{fold['test_patient']} reconstruction errors disagree with residuals")
    decisions_is_anomaly = np.asarray(decisions["is_anomaly"], dtype=bool)
    if list(np.asarray(decisions["decision"], dtype=str)) != ["anomaly" if v else "normal" for v in decisions_is_anomaly]:
        _fail(f"{dataset}/{fold['test_patient']} decisions disagree with boolean flags")
    threshold = float(thresholds["overall"])
    if not np.isfinite(threshold) or threshold <= 0:
        _fail(f"{dataset}/{fold['test_patient']} threshold is not finite and positive")
    margins = np.asarray(decisions["anomaly_margin"], dtype="float64")
    normalized = np.asarray(decisions["normalized_margin"], dtype="float64")
    if not np.allclose(margins, errors - threshold, rtol=2e-5, atol=2e-7):
        _fail(f"{dataset}/{fold['test_patient']} anomaly margins disagree with threshold")
    if not np.allclose(normalized, margins / threshold, rtol=2e-5, atol=2e-7):
        _fail(f"{dataset}/{fold['test_patient']} normalized margins disagree with threshold")
    if not np.array_equal(decisions_is_anomaly, errors > threshold):
        _fail(f"{dataset}/{fold['test_patient']} anomaly flags disagree with threshold")
    if dataset == "sleep":
        channel_thresholds = np.asarray(thresholds.get("channel_thresholds", []), dtype="float64")
        if channel_thresholds.shape != (reconstruction.shape[-1],) or not np.all(np.isfinite(channel_thresholds)) or np.any(channel_thresholds <= 0):
            _fail(f"{dataset}/{fold['test_patient']} channel thresholds are invalid")
    else:
        channel_thresholds = None
    return {
        "root": root,
        "fold_dir": fold_dir,
        "fold": fold,
        "decisions": decisions,
        "thresholds": thresholds,
        "normalization": normalization,
        "evaluation": evaluation,
        "arrays": arrays,
        "channel_thresholds": channel_thresholds,
    }


def _metadata_match(dataset: str, artifact_rows: list[dict[str, Any]], metadata: list[dict[str, Any]], patient: str) -> list[int]:
    indices = [index for index, row in enumerate(metadata) if row.get("patient") == patient]
    if len(indices) != len(artifact_rows):
        _fail(f"{dataset}/{patient} artifact/data window count mismatch: {len(artifact_rows)} != {len(indices)}")
    for index, artifact in zip(indices, artifact_rows):
        expected = metadata[index]
        for key in ("patient", "recording_id", "window_index", "sample_start", "sample_end"):
            if str(expected.get(key)) != str(artifact.get(key)):
                _fail(f"{dataset}/{patient} metadata mismatch for {key} at index {index}")
        if dataset == "sleep" and bool(expected.get("transition", False)) != bool(artifact.get("transition", False)):
            _fail(f"{dataset}/{patient} transition metadata mismatch at index {index}")
    return indices


def _sleep_recording_metadata(recording_id: str, cache: dict[str, dict[str, Any]]) -> dict[str, Any]:
    if recording_id in cache:
        return cache[recording_id]
    import mne

    path = DATA_ROOT / "sleep_edf" / f"{recording_id}-PSG.edf"
    raw = mne.io.read_raw_edf(path, preload=False, verbose=False)
    rates = channel_sampling_rates(raw)
    pair_index = list(recording for patient in SLEEP_PATIENTS for recording in SLEEP_RECORDINGS[patient]).index(recording_id)
    start, provenance = timestamp_for_recording(raw, pair_index)
    raw_rate = float(raw.info["sfreq"])
    modeled_rate = float(rates[SLEEP_MODELED_CHANNELS[0]]) / 10.0
    info = {
        "recording_id": recording_id,
        "raw_sampling_rate_hz": raw_rate,
        "recording_sampling_rate_hz": meaningful_recording_rate(rates),
        "channel_sampling_rates": rates,
        "modeled_sampling_rates": {channel: float(rates[channel]) / 10.0 for channel in SLEEP_MODELED_CHANNELS},
        "aux_sampling_rates": {channel: float(rates[channel]) for channel in SLEEP_AUX_CHANNELS if channel in rates},
        "recording_start_time": start,
        "timestamp_provenance": provenance,
        "duration_seconds": float(raw.n_times / raw_rate),
        "raw_channel_names": list(raw.ch_names),
    }
    raw.close()
    cache[recording_id] = info
    return info


def _lock_window_model(cur, window_id: int, model_id: int | None = None) -> None:
    """Serialize all prediction/series writes for one analysis window.

    The lock is intentionally keyed only by ``window_id``: the database
    contract permits exactly one prediction-series row per analyzed window,
    so differing model IDs must not be able to race each other into two rows.
    """

    cur.execute(
        "SELECT pg_advisory_xact_lock(hashtextextended(%s, 6072026))",
        (f"pdm-prediction-series-window:{int(window_id)}",),
    )


def _apnea_recording_metadata(patient: str, cache: dict[str, dict[str, Any]]) -> dict[str, Any]:
    if patient in cache:
        return cache[patient]
    import wfdb

    record = wfdb.rdrecord(str(DATA_ROOT / "apnea" / patient))
    order = list(APNEA_PATIENTS).index(patient)
    info = {
        "recording_id": patient,
        "raw_sampling_rate_hz": float(record.fs),
        "recording_sampling_rate_hz": float(record.fs),
        "recording_start_time": BASE_TIME + timedelta(days=order),
        "timestamp_provenance": "normalized_experimental",
        "duration_seconds": float(len(record.p_signal) / record.fs),
        "raw_channel_names": ["ECG"],
    }
    cache[patient] = info
    return info


def _upsert_id(cur, select_sql: str, select_args: tuple[Any, ...], insert_sql: str, insert_args: tuple[Any, ...], update_sql: str | None = None, update_args: tuple[Any, ...] = ()) -> int:
    cur.execute(select_sql, select_args)
    rows = cur.fetchall()
    if len(rows) > 1:
        raise ValueError("database contains duplicate natural-key rows")
    if rows:
        identifier = int(rows[0][0])
        if update_sql:
            if update_args and update_args[-1] is None:
                update_args = (*update_args[:-1], identifier)
            cur.execute(update_sql, update_args)
        return identifier
    cur.execute(insert_sql, insert_args)
    return int(cur.fetchone()[0])


def _upsert_signal(cur, recording_id: str, signal_type: str, channel: str, rate: float | None) -> int:
    return _upsert_id(
        cur,
        "SELECT signal_id FROM signal WHERE recording_id=%s AND signal_type=%s AND channel=%s",
        (recording_id, signal_type, channel),
        "INSERT INTO signal(recording_id,signal_type,channel,sampling_rate_hz) VALUES(%s,%s,%s,%s) RETURNING signal_id",
        (recording_id, signal_type, channel, rate),
    )


def _fix_signal_rate(cur, signal_id: int, rate: float | None) -> None:
    cur.execute("UPDATE signal SET sampling_rate_hz=%s WHERE signal_id=%s", (rate, signal_id))


def _upsert_annotation(cur, recording_id: str, window_id: int, annotation_type: str, label: str, start: datetime, end: datetime, source: str) -> int:
    cur.execute(
        "SELECT annotation_id FROM annotation WHERE recording_id=%s AND window_id=%s AND annotation_type=%s AND source=%s",
        (recording_id, window_id, annotation_type, source),
    )
    rows = cur.fetchall()
    if len(rows) > 1:
        raise ValueError("database contains duplicate annotation natural-key rows")
    if rows:
        annotation_id = int(rows[0][0])
        cur.execute("UPDATE annotation SET label=%s,start_time=%s,end_time=%s WHERE annotation_id=%s", (label, start, end, annotation_id))
        return annotation_id
    cur.execute(
        "INSERT INTO annotation(recording_id,window_id,annotation_type,label,start_time,end_time,source) VALUES(%s,%s,%s,%s,%s,%s,%s) RETURNING annotation_id",
        (recording_id, window_id, annotation_type, label, start, end, source),
    )
    return int(cur.fetchone()[0])


def _dense_indices(metadata: list[dict[str, Any]], indices: list[int], is_anomaly: np.ndarray, margins: np.ndarray) -> list[int]:
    """Select one model-only event anchor per recording plus its neighbours."""

    position_by_global = {global_index: position for position, global_index in enumerate(indices)}
    selected: set[int] = set()
    for recording_id in sorted({metadata[index]["recording_id"] for index in indices}):
        local = [index for index in indices if metadata[index]["recording_id"] == recording_id]
        anomalies = [index for index in local if bool(is_anomaly[position_by_global[index]])]
        candidates = anomalies or local
        anchor = max(candidates, key=lambda index: (float(margins[position_by_global[index]]), -int(metadata[index]["window_index"])))
        position = local.index(anchor)
        selected.update(local[max(0, position - 1) : min(len(local), position + 2)])
    return sorted(selected, key=lambda index: (metadata[index]["recording_id"], int(metadata[index]["window_index"])))


def _upsert_window(cur, recording_id: str, metadata: dict[str, Any], start: datetime, end: datetime) -> int:
    return _upsert_id(
        cur,
        "SELECT window_id FROM analysis_window WHERE recording_id=%s AND window_index=%s",
        (recording_id, int(metadata["window_index"])),
        "INSERT INTO analysis_window(recording_id,window_index,start_sample,end_sample,start_time,end_time,duration_seconds) VALUES(%s,%s,%s,%s,%s,%s,%s) RETURNING window_id",
        (recording_id, int(metadata["window_index"]), int(metadata["sample_start"]), int(metadata["sample_end"]), start, end, (end - start).total_seconds()),
        "UPDATE analysis_window SET start_sample=%s,end_sample=%s,start_time=%s,end_time=%s,duration_seconds=%s WHERE window_id=%s",
        (int(metadata["sample_start"]), int(metadata["sample_end"]), start, end, (end - start).total_seconds(), None),
    )


def _upsert_prediction(cur, window_id: int, model_id: int, values: tuple[Any, ...]) -> int:
    _lock_window_model(cur, window_id, model_id)
    cur.execute("SELECT prediction_id FROM prediction WHERE window_id=%s AND model_id=%s", (window_id, model_id))
    rows = cur.fetchall()
    if len(rows) > 1:
        raise ValueError("database contains duplicate prediction natural-key rows")
    if rows:
        prediction_id = int(rows[0][0])
        cur.execute(
            "UPDATE prediction SET predicted_class=%s,reconstruction_error=%s,anomaly_threshold=%s,is_anomaly=%s,severity=%s,status='complete' WHERE prediction_id=%s",
            (*values, prediction_id),
        )
        return prediction_id
    cur.execute(
        "INSERT INTO prediction(window_id,model_id,predicted_class,reconstruction_error,anomaly_threshold,is_anomaly,severity,status) VALUES(%s,%s,%s,%s,%s,%s,%s,'complete') RETURNING prediction_id",
        (window_id, model_id, *values),
    )
    return int(cur.fetchone()[0])


def _upsert_prediction_series(cur, values: tuple[Any, ...]) -> None:
    window_id, prediction_id, model_id = values[3], values[4], values[5]
    _lock_window_model(cur, window_id, model_id)
    cur.execute("SELECT 1 FROM prediction_series WHERE window_id=%s", (window_id,))
    rows = cur.fetchall()
    if len(rows) > 1:
        raise ValueError("database contains duplicate prediction_series rows")
    if rows:
        cur.execute(
            "UPDATE prediction_series SET timestamp=%s,prediction_id=%s,patient_id=%s,recording_id=%s,model_id=%s,reconstruction_error=%s,anomaly_threshold=%s,anomaly_margin=%s,normalized_margin=%s,is_anomaly=%s,predicted_class=%s,probability=NULL WHERE window_id=%s",
            (values[0], values[4], values[1], values[2], values[5], values[6], values[7], values[8], values[9], values[10], values[11], values[3]),
        )
        return
    cur.execute(
        "INSERT INTO prediction_series(timestamp,patient_id,recording_id,window_id,prediction_id,model_id,reconstruction_error,anomaly_threshold,anomaly_margin,normalized_margin,is_anomaly,predicted_class,probability) VALUES(%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,NULL)",
        values,
    )


def _window_signal_links(cur, window_id: int, signal_ids: list[int]) -> None:
    execute_values(cur, "INSERT INTO window_signal(window_id,signal_id) VALUES %s ON CONFLICT DO NOTHING", [(window_id, signal_id) for signal_id in signal_ids])


def _upsert_model(cur, dataset: str, fold: dict[str, Any], fold_data: dict[str, Any], artifact_record: dict[str, Any]) -> int:
    patient = fold["test_patient"]
    model_name = f"fusion_{dataset}_loso_{patient}"
    model_version = f"{fold['fold_id']}_seed{fold['seed']}"
    fold_dir = fold_data["fold_dir"]
    provenance = {
        "schema_version": "phase6-model-provenance-v1",
        "dataset": dataset,
        "fold_id": fold["fold_id"],
        "test_patient": patient,
        "validation_patient": fold["validation_patient"],
        "fold_manifest": str(fold_dir.relative_to(ROOT).as_posix()),
        "model_checkpoint": str((fold_dir / "model.keras").relative_to(ROOT).as_posix()),
        "model_checkpoint_sha256": artifact_record["model_checkpoint_sha256"],
        "artifact_hashes": artifact_record["artifact_hashes"],
        "runtime_metadata": _load_json(fold_dir / "runtime_metadata.json"),
        "normalization": {"fit_scope": fold_data["normalization"].get("fit_scope"), "normalization_patients": fold_data["normalization"].get("normalization_patients")},
        "thresholds": fold_data["thresholds"],
    }
    config = _load_json(fold_dir / "model_config.json")
    return _upsert_id(
        cur,
        "SELECT model_id FROM model WHERE model_name=%s AND model_version=%s",
        (model_name, model_version),
        "INSERT INTO model(model_name,model_version,task,architecture,training_dataset,input_configuration) VALUES(%s,%s,%s,%s,%s,%s) RETURNING model_id",
        (model_name, model_version, "anomaly detection", config.get("architecture", "fusion temporal-attention autoencoder"), dataset, _json_text(provenance)),
        "UPDATE model SET task=%s,architecture=%s,training_dataset=%s,input_configuration=%s WHERE model_id=%s",
        ("anomaly detection", config.get("architecture", "fusion temporal-attention autoencoder"), dataset, _json_text(provenance), None),
    )


def _dense_expected_signal_counts(
    dataset: str,
    signal_ids: dict[str, int],
    raw_window: dict[str, np.ndarray],
    reconstruction: np.ndarray,
    residual: np.ndarray,
) -> dict[int, int]:
    """Return exact per-signal row counts expected for one dense window."""

    expected: dict[int, int] = {}
    modeled = raw_window["modeled"] if dataset == "sleep" else raw_window["features"]
    for channel_index, channel in enumerate(
        SLEEP_MODELED_CHANNELS if dataset == "sleep" else APNEA_FEATURE_CHANNELS
    ):
        expected[signal_ids[f"raw:{channel}"]] = int(len(modeled[:, channel_index]))
        expected[signal_ids[f"reconstruction:{channel}"]] = int(len(reconstruction[:, channel_index]))
        expected[signal_ids[f"residual:{channel}"]] = int(len(residual[:, channel_index]))
    if dataset == "sleep":
        for channel_index, channel in enumerate(SLEEP_AUX_CHANNELS):
            expected[signal_ids[f"raw:{channel}"]] = int(len(raw_window["aux"][:, channel_index]))
    else:
        expected[signal_ids["raw:ECG"]] = int(len(raw_window["raw"]))
    return expected


def _inspect_dense_state(
    cur,
    window_id: int,
    prediction_id: int,
    start: datetime,
    duration: float,
    attention_length: int,
    expected_signal_counts: dict[int, int],
    expected_locations: int,
    expected_features: int,
) -> str:
    """Classify a dense window as absent, complete, or partial.

    Signal data is not keyed by window in the Phase 5 schema, so each expected
    signal is checked by its exact signal ID and exact window timestamp range.
    Any count mismatch, unexpected attention owner, duplicate explanation, or
    incomplete XAI rows is treated as partial and aborts the transaction.
    """

    cur.execute(
        "SELECT COUNT(*),COUNT(*) FILTER (WHERE signal_id IS NOT NULL) FROM attention_data WHERE window_id=%s",
        (window_id,),
    )
    attention_count, attention_owned_count = (int(value) for value in cur.fetchone())
    signal_counts: dict[int, int] = {}
    end = start + timedelta(seconds=duration)
    for signal_id in expected_signal_counts:
        cur.execute(
            "SELECT COUNT(*) FROM signal_data WHERE signal_id=%s AND timestamp >= %s AND timestamp < %s",
            (signal_id, start, end),
        )
        signal_counts[signal_id] = int(cur.fetchone()[0])
    cur.execute(
        "SELECT explanation_id,xai_method FROM xai_explanation WHERE prediction_id=%s",
        (prediction_id,),
    )
    explanations = cur.fetchall()
    location_count = feature_count = 0
    if len(explanations) == 1:
        explanation_id, method = explanations[0]
        cur.execute("SELECT COUNT(*) FROM xai_location WHERE explanation_id=%s", (explanation_id,))
        location_count = int(cur.fetchone()[0])
        cur.execute("SELECT COUNT(*) FROM xai_feature WHERE explanation_id=%s", (explanation_id,))
        feature_count = int(cur.fetchone()[0])
    expected_complete = (
        attention_count == attention_length
        and attention_owned_count == 0
        and all(signal_counts[signal_id] == expected_count for signal_id, expected_count in expected_signal_counts.items())
        and len(explanations) == 1
        and explanations[0][1] == "temporal_attention"
        and location_count == expected_locations
        and feature_count == expected_features
    )
    any_state = bool(
        attention_count
        or any(signal_counts.values())
        or explanations
        or location_count
        or feature_count
    )
    if expected_complete:
        return "complete"
    return "partial" if any_state else "absent"


def _populate_dense(cur, dataset: str, fold_data: dict[str, Any], global_index: int, window_id: int, prediction_id: int, signal_ids: dict[str, int], start: datetime, raw_window: dict[str, np.ndarray] | None) -> None:
    arrays = fold_data["arrays"]
    attention = np.asarray(arrays["attention"][global_index], dtype="float64")
    residual = np.asarray(arrays["residual"][global_index], dtype="float64")
    reconstruction = np.asarray(arrays["reconstruction"][global_index], dtype="float64")
    if dataset == "sleep":
        mean = np.asarray(fold_data["normalization"]["mean"], dtype="float64").reshape(1, -1)
        std = np.asarray(fold_data["normalization"]["std"], dtype="float64").reshape(1, -1)
        reconstruction_physical = reconstruction * std + mean
        modeled_channels = list(SLEEP_MODELED_CHANNELS)
        model_rate = 10.0
        raw_values = raw_window["modeled"]
    else:
        mean = np.asarray(fold_data["normalization"]["mean"], dtype="float64").reshape(1, -1)
        std = np.asarray(fold_data["normalization"]["std"], dtype="float64").reshape(1, -1)
        reconstruction_physical = reconstruction * std + mean
        modeled_channels = list(APNEA_FEATURE_CHANNELS)
        model_rate = 100.0 / 60.0
        raw_values = raw_window["features"]

    expected_signal_counts = _dense_expected_signal_counts(
        dataset, signal_ids, raw_window, reconstruction, residual
    )
    duration = float(len(attention) / model_rate)
    dense_state = _inspect_dense_state(
        cur,
        window_id,
        prediction_id,
        start,
        duration,
        len(attention),
        expected_signal_counts,
        3 if dataset == "sleep" else 0,
        21 if dataset == "sleep" else 0,
    )
    if dense_state == "complete":
        return
    if dense_state == "partial":
        raise ValueError(
            f"partial dense evidence exists for window {window_id}; refusing to mutate it"
        )

    rows = []
    for channel_index, channel in enumerate(modeled_channels):
        raw_signal_id = signal_ids[f"raw:{channel}"]
        reconstruction_id = signal_ids[f"reconstruction:{channel}"]
        residual_id = signal_ids[f"residual:{channel}"]
        raw_channel = np.asarray(raw_values[:, channel_index], dtype="float64")
        native_rate = float(raw_window["raw_rates"][channel]) if dataset == "sleep" else float(raw_window["raw_rate"])
        for index, value in enumerate(raw_channel):
            rows.append((start + timedelta(seconds=index / native_rate), raw_signal_id, channel, float(value), None))
        for index, value in enumerate(reconstruction_physical[:, channel_index]):
            rows.append((start + timedelta(seconds=index / model_rate), reconstruction_id, channel, float(value), None))
        for index, value in enumerate(residual[:, channel_index]):
            rows.append((start + timedelta(seconds=index / model_rate), residual_id, channel, float(value), None))
    if dataset == "sleep":
        for channel_index, channel in enumerate(SLEEP_AUX_CHANNELS):
            values = raw_window["aux"][:, channel_index]
            signal_id = signal_ids[f"raw:{channel}"]
            rate = float(raw_window["raw_rates"][channel])
            for index, value in enumerate(values):
                rows.append((start + timedelta(seconds=index / rate), signal_id, channel, float(value), None))
    else:
        raw_signal_id = signal_ids["raw:ECG"]
        for index, value in enumerate(raw_window["raw"]):
            rows.append((start + timedelta(seconds=index / float(raw_window["raw_rate"])), raw_signal_id, "ECG", float(value), None))
    execute_values(cur, "INSERT INTO signal_data(timestamp,signal_id,channel,value,quality) VALUES %s", rows, page_size=10000)
    attention_rows = [(start + timedelta(seconds=index / model_rate), window_id, None, float(value)) for index, value in enumerate(attention)]
    execute_values(cur, "INSERT INTO attention_data(timestamp,window_id,signal_id,attention_value) VALUES %s", attention_rows, page_size=10000)

    explanation_id = _upsert_id(
        cur,
        "SELECT explanation_id FROM xai_explanation WHERE prediction_id=%s AND xai_method='temporal_attention'",
        (prediction_id,),
        "INSERT INTO xai_explanation(prediction_id,xai_method,explanation_type,explanation_text) VALUES(%s,'temporal_attention','window_level_temporal_relevance',%s) RETURNING explanation_id",
        (prediction_id, "Shared window-level temporal relevance; attention has no channel owner."),
    )
    if dataset == "sleep":
        times = np.asarray([start + timedelta(seconds=index / model_rate) for index in range(reconstruction.shape[0])], dtype=object)
        evidence = compute_channel_evidence(
            residual + reconstruction,
            reconstruction,
            fold_data["channel_thresholds"],
            times,
        )
        # The normalized model input is reconstructed + residual; this avoids
        # re-reading the full Sleep corpus and exactly matches held-out output.
        cur.executemany(
            "INSERT INTO xai_location(explanation_id,signal_id,channel,temporal_start,temporal_end,spatial_location) VALUES(%s,%s,%s,%s,%s,'peak_residual')",
            [(explanation_id, signal_ids[f"raw:{channel}"], channel, row["peak_residual_time"], row["peak_residual_time"]) for channel, row in zip(SLEEP_MODELED_CHANNELS, evidence)],
        )
        feature_rows = []
        for channel, row in zip(SLEEP_MODELED_CHANNELS, evidence):
            signal_id = signal_ids[f"raw:{channel}"]
            for name in ("mean_error", "max_error", "contribution_share", "validation_threshold", "raw_margin", "normalized_margin", "residual_threshold_flag"):
                value = row[name]
                feature_rows.append((explanation_id, signal_id, name, float(value) if not isinstance(value, bool) else float(value), None, row["channel_index"] + 1))
        execute_values(cur, "INSERT INTO xai_feature(explanation_id,signal_id,feature_name,feature_value,attribution_value,importance_rank) VALUES %s", feature_rows)


def _load_dense_raw(dataset: str, recording_id: str, metadata: dict[str, Any], model_x: np.ndarray, global_index: int) -> dict[str, Any]:
    if dataset == "apnea":
        import wfdb

        record = wfdb.rdrecord(str(DATA_ROOT / "apnea" / recording_id))
        start = int(metadata["sample_start"])
        end = int(metadata["sample_end"])
        raw = np.asarray(record.p_signal[start:end, 0], dtype="float64")
        return {"recording_id": f"apnea_{recording_id}", "raw": raw, "features": np.asarray(model_x, dtype="float64"), "raw_rate": float(record.fs), "raw_rates": {"ECG": float(record.fs), **{channel: float(record.fs) / 60.0 for channel in APNEA_FEATURE_CHANNELS}}}
    import mne

    info = _sleep_recording_metadata(recording_id, _SLEEP_METADATA_CACHE)
    path = DATA_ROOT / "sleep_edf" / f"{recording_id}-PSG.edf"
    raw = mne.io.read_raw_edf(path, preload=False, verbose=False)
    start = int(metadata["sample_start"])
    end = int(metadata["sample_end"])
    channel_names = list(SLEEP_MODELED_CHANNELS + SLEEP_AUX_CHANNELS)
    picks = [raw.ch_names.index(channel) for channel in channel_names if channel in raw.ch_names]
    matrix = raw.get_data(picks=picks, start=start, stop=end).T
    by_name = {name: matrix[:, index] for index, name in enumerate([channel_names[index] for index in range(len(channel_names)) if channel_names[index] in raw.ch_names])}
    raw.close()
    modeled = np.asarray(model_x, dtype="float64")
    aux_values = []
    for channel in SLEEP_AUX_CHANNELS:
        values = np.asarray(by_name.get(channel, np.empty(0)), dtype="float64")
        rate = float(info["channel_sampling_rates"].get(channel, info["raw_sampling_rate_hz"]))
        stride = max(1, int(round(info["raw_sampling_rate_hz"] / rate)))
        aux_values.append(values[::stride])
    aux = np.column_stack(aux_values) if aux_values else np.empty((0, 0))
    return {"recording_id": f"sleep_{recording_id}", "modeled": modeled, "aux": aux, "raw_rate": info["raw_sampling_rate_hz"], "raw_rates": {**info["modeled_sampling_rates"], **info["aux_sampling_rates"]}}


def populate_dataset(dataset: str, validation: dict[str, Any]) -> dict[str, Any]:
    model_x, metadata = load_dataset(dataset)
    root = ARTIFACT_ROOT / dataset
    fold_rows = validation["root_manifest"]["folds"]
    recording_cache: dict[str, dict[str, Any]] = {}
    fold_summaries = []
    total_dense = 0
    total_windows = 0
    for row in fold_rows:
        fold_data = _fold_data(dataset, root, row)
        fold = fold_data["fold"]
        patient = fold["test_patient"]
        artifact_rows = fold_data["evaluation"]["annotations"]
        indices = _metadata_match(dataset, artifact_rows, metadata, patient)
        is_anomaly = np.asarray(fold_data["decisions"]["is_anomaly"], dtype=bool)
        margins = np.asarray(fold_data["decisions"]["anomaly_margin"], dtype="float64")
        dense = _dense_indices(metadata, indices, is_anomaly, margins)
        with connect() as conn, conn.cursor() as cur:
            model_id = _upsert_model(cur, dataset, fold, fold_data, validation["folds"][patient])
            modality = "apnea_ecg" if dataset == "apnea" else "sleep_edf"
            cur.execute("INSERT INTO model_modality(model_id,modality) VALUES(%s,%s) ON CONFLICT DO NOTHING", (model_id, modality))
            patient_code = f"APNEA_{patient}" if dataset == "apnea" else f"SLEEP_{patient}"
            cur.execute("SELECT patient_id FROM patient WHERE patient_code=%s", (patient_code,))
            found = cur.fetchone()
            if found:
                patient_id = int(found[0])
            else:
                cur.execute("INSERT INTO patient(patient_code) VALUES(%s) RETURNING patient_id", (patient_code,))
                patient_id = int(cur.fetchone()[0])
            recording_ids = sorted({metadata[index]["recording_id"] for index in indices})
            signal_map: dict[str, dict[str, int]] = {}
            for recording_id in recording_ids:
                info = _apnea_recording_metadata(recording_id, recording_cache) if dataset == "apnea" else _sleep_recording_metadata(recording_id, recording_cache)
                db_recording_id = f"apnea_{recording_id}" if dataset == "apnea" else f"sleep_{recording_id}"
                source_path = (DATA_ROOT / "apnea" / recording_id) if dataset == "apnea" else (DATA_ROOT / "sleep_edf" / f"{recording_id}-PSG.edf")
                cur.execute(
                    "INSERT INTO recording(recording_id,patient_id,source_recording_name,modality,sampling_rate_hz,length_seconds,recording_start_time,data_location) VALUES(%s,%s,%s,%s,%s,%s,%s,%s) ON CONFLICT(recording_id) DO UPDATE SET patient_id=EXCLUDED.patient_id,source_recording_name=EXCLUDED.source_recording_name,modality=EXCLUDED.modality,sampling_rate_hz=EXCLUDED.sampling_rate_hz,length_seconds=EXCLUDED.length_seconds,recording_start_time=EXCLUDED.recording_start_time,data_location=EXCLUDED.data_location",
                    (db_recording_id, patient_id, recording_id, dataset, info["recording_sampling_rate_hz"], info["duration_seconds"], info["recording_start_time"], str(source_path.as_posix())),
                )
                signal_map[recording_id] = {}
                if dataset == "apnea":
                    definitions = [("raw", "ECG", info["raw_sampling_rate_hz"])] + [(kind, channel, info["raw_sampling_rate_hz"] / 60.0) for kind in ("raw", "reconstruction", "residual") for channel in APNEA_FEATURE_CHANNELS]
                else:
                    definitions = []
                    definitions.extend(("raw", channel, info["modeled_sampling_rates"][channel]) for channel in SLEEP_MODELED_CHANNELS)
                    definitions.extend(("raw", channel, info["aux_sampling_rates"].get(channel, info["raw_sampling_rate_hz"])) for channel in SLEEP_AUX_CHANNELS)
                    definitions.extend((kind, channel, info["modeled_sampling_rates"][channel]) for kind in ("reconstruction", "residual") for channel in SLEEP_MODELED_CHANNELS)
                for signal_type, channel, rate in definitions:
                    signal_id = _upsert_signal(cur, db_recording_id, signal_type, channel, rate)
                    _fix_signal_rate(cur, signal_id, rate)
                    signal_map[recording_id][f"{signal_type}:{channel}"] = signal_id
            for local, global_index in enumerate(indices):
                item = metadata[global_index]
                recording_id = item["recording_id"]
                info = _apnea_recording_metadata(recording_id, recording_cache) if dataset == "apnea" else _sleep_recording_metadata(recording_id, recording_cache)
                start = info["recording_start_time"] + timedelta(seconds=float(item["sample_start"]) / float(info["raw_sampling_rate_hz"]))
                end = info["recording_start_time"] + timedelta(seconds=float(item["sample_end"]) / float(info["raw_sampling_rate_hz"]))
                db_recording_id = f"apnea_{recording_id}" if dataset == "apnea" else f"sleep_{recording_id}"
                window_id = _upsert_window(cur, db_recording_id, item, start, end)
                signal_ids = list(signal_map[recording_id].values())
                _window_signal_links(cur, window_id, signal_ids)
                label = str(item.get("official_label", ""))
                annotation_type = "apnea_official" if dataset == "apnea" else "sleep_stage_official"
                _upsert_annotation(cur, db_recording_id, window_id, annotation_type, label, start, end, "official_evaluation_only")
                if dataset == "sleep" and bool(item.get("transition", False)):
                    _upsert_annotation(cur, db_recording_id, window_id, "stage_transition_derived", "transition", start, end, "derived_private_evaluation")
                error = float(fold_data["arrays"]["errors"][local])
                threshold = float(fold_data["thresholds"]["overall"])
                anomaly = bool(fold_data["decisions"]["is_anomaly"][local])
                # Persist margins computed from the stored float32 error and
                # threshold so the database contract is exactly internally
                # consistent after artifact serialization quantization.
                margin = error - threshold
                normalized_margin = margin / threshold
                prediction_id = _upsert_prediction(cur, window_id, model_id, ("anomaly" if anomaly else "normal", error, threshold, anomaly, "high" if anomaly else "normal"))
                _upsert_prediction_series(cur, (start, patient_id, db_recording_id, window_id, prediction_id, model_id, error, threshold, margin, normalized_margin, anomaly, "anomaly" if anomaly else "normal"))
                if global_index in dense:
                    raw_window = _load_dense_raw(dataset, recording_id, item, model_x[global_index] if dataset == "sleep" else model_x[global_index], global_index)
                    _populate_dense(cur, dataset, fold_data, local, window_id, prediction_id, signal_map[recording_id], start, raw_window)
            conn.commit()
        total_windows += len(indices)
        total_dense += len(dense)
        fold_summaries.append({"test_patient": patient, "fold_id": fold["fold_id"], "model_id": model_id, "window_count": len(indices), "dense_window_count": len(dense), "anomaly_count": int(is_anomaly.sum()), "model_checkpoint_sha256": validation["folds"][patient]["model_checkpoint_sha256"]})
    return {"dataset": dataset, "window_count": total_windows, "dense_window_count": total_dense, "folds": fold_summaries}


def refresh_aggregate() -> None:
    """Refresh all historical source timestamps after a complete population."""

    # TimescaleDB disallows refreshing a continuous aggregate from inside an
    # explicit transaction.  Keep this separate from each population
    # transaction and issue the procedure call in autocommit mode.
    with connect() as conn:
        conn.autocommit = True
        with conn.cursor() as cur:
            cur.execute("CALL refresh_continuous_aggregate('pdm_prediction_5m', NULL, NULL)")


def validate_selection_manifest(manifest: dict[str, Any]) -> list[dict[str, Any]]:
    """Validate the immutable Phase 9 selection core before any database write."""

    core = manifest.get("selection_core")
    if not isinstance(core, dict) or not isinstance(core.get("cases"), list):
        raise ValueError("selection manifest must contain selection_core.cases")
    expected_hash = canonical_json_hash(core)
    if manifest.get("selection_hash") != expected_hash:
        raise ValueError("selection hash does not match the immutable selection core")
    cases = core["cases"]
    prediction_ids = [int(case["prediction_id"]) for case in cases]
    if len(prediction_ids) != len(set(prediction_ids)):
        raise ValueError("selection manifest contains duplicate prediction IDs")
    return cases


def _selection_target_rows(cases: list[dict[str, Any]]) -> list[dict[str, Any]]:
    """Resolve frozen current cases and their five package-history events."""

    current_ids = [int(case["prediction_id"]) for case in cases]
    target_ids = set(current_ids)
    with connect() as conn, conn.cursor() as cur:
        for prediction_id in current_ids:
            cur.execute(
                """
                SELECT prior.prediction_id
                FROM prediction_series current
                JOIN LATERAL (
                    SELECT history.prediction_id
                    FROM prediction_series history
                    WHERE history.patient_id = current.patient_id
                      AND history.timestamp < current.timestamp
                    ORDER BY history.timestamp DESC, history.prediction_id DESC
                    LIMIT 5
                ) prior ON TRUE
                WHERE current.prediction_id = %s
                """,
                (prediction_id,),
            )
            history_ids = [int(row[0]) for row in cur.fetchall()]
            if len(history_ids) != 5:
                raise ValueError(
                    f"selected prediction {prediction_id} does not have five earlier history events"
                )
            target_ids.update(history_ids)
        cur.execute(
            """
            SELECT ps.prediction_id, ps.window_id, ps.patient_id, p.patient_code,
                   r.source_recording_name, w.window_index, w.start_time
            FROM prediction_series ps
            JOIN patient p ON p.patient_id = ps.patient_id
            JOIN recording r ON r.recording_id = ps.recording_id
            JOIN analysis_window w ON w.window_id = ps.window_id
            WHERE ps.prediction_id = ANY(%s)
            ORDER BY p.patient_code, ps.timestamp, ps.prediction_id
            """,
            (list(sorted(target_ids)),),
        )
        columns = (
            "prediction_id", "window_id", "patient_id", "patient_code",
            "source_recording_name", "window_index", "start_time",
        )
        rows = [dict(zip(columns, values)) for values in cur.fetchall()]
    if len(rows) != len(target_ids):
        found = {int(row["prediction_id"]) for row in rows}
        raise ValueError(f"selection references missing predictions: {sorted(target_ids - found)}")

    current_by_id = {int(case["prediction_id"]): case for case in cases}
    for row in rows:
        frozen = current_by_id.get(int(row["prediction_id"]))
        if frozen is None:
            continue
        for key in ("window_id", "patient_id", "patient_code"):
            if str(row[key]) != str(frozen[key]):
                raise ValueError(
                    f"frozen identity mismatch for prediction {row['prediction_id']}: {key}"
                )
    return rows


def materialize_selection(
    selection_path: Path, validation: dict[str, Any]
) -> dict[str, Any]:
    """Materialize only frozen Apnea current/history dense evidence."""

    manifest = _load_json(selection_path)
    cases = validate_selection_manifest(manifest)
    target_rows = _selection_target_rows(cases)
    model_x, metadata = load_dataset("apnea")
    metadata_lookup = {
        (str(row["recording_id"]), int(row["window_index"])): (index, row)
        for index, row in enumerate(metadata)
    }
    targets_by_patient: dict[str, list[dict[str, Any]]] = {}
    for row in target_rows:
        patient = str(row["patient_code"]).removeprefix("APNEA_")
        targets_by_patient.setdefault(patient, []).append(row)

    newly_materialized = 0
    already_complete = 0
    root = ARTIFACT_ROOT / "apnea"
    fold_rows = {
        str(row["test_patient"]): row
        for row in validation["root_manifest"]["folds"]
    }
    for patient, targets in sorted(targets_by_patient.items()):
        if patient not in fold_rows:
            raise ValueError(f"selection patient has no validated fold: {patient}")
        fold_data = _fold_data("apnea", root, fold_rows[patient])
        fold_indices = _metadata_match(
            "apnea", fold_data["evaluation"]["annotations"], metadata, patient
        )
        local_by_global = {global_index: local for local, global_index in enumerate(fold_indices)}
        with connect() as conn, conn.cursor() as cur:
            for target in targets:
                key = (str(target["source_recording_name"]), int(target["window_index"]))
                if key not in metadata_lookup:
                    raise ValueError(f"selection window is absent from Apnea source metadata: {key}")
                global_index, metadata_row = metadata_lookup[key]
                if global_index not in local_by_global:
                    raise ValueError(f"selection window {key} is not in held-out fold {patient}")
                cur.execute(
                    "SELECT signal_type,channel,signal_id FROM signal WHERE recording_id=%s",
                    (f"apnea_{target['source_recording_name']}",),
                )
                signal_ids = {
                    f"{signal_type}:{channel}": int(signal_id)
                    for signal_type, channel, signal_id in cur.fetchall()
                }
                expected_keys = {"raw:ECG"} | {
                    f"{kind}:{channel}"
                    for kind in ("raw", "reconstruction", "residual")
                    for channel in APNEA_FEATURE_CHANNELS
                }
                missing_keys = sorted(expected_keys - signal_ids.keys())
                if missing_keys:
                    raise ValueError(
                        f"selection recording is missing signal definitions: {missing_keys}"
                    )
                cur.execute(
                    "SELECT EXISTS(SELECT 1 FROM xai_explanation WHERE prediction_id=%s)",
                    (int(target["prediction_id"]),),
                )
                existed = bool(cur.fetchone()[0])
                raw_window = _load_dense_raw(
                    "apnea",
                    str(target["source_recording_name"]),
                    metadata_row,
                    model_x[global_index],
                    global_index,
                )
                _populate_dense(
                    cur,
                    "apnea",
                    fold_data,
                    local_by_global[global_index],
                    int(target["window_id"]),
                    int(target["prediction_id"]),
                    signal_ids,
                    target["start_time"],
                    raw_window,
                )
                already_complete += int(existed)
                newly_materialized += int(not existed)
            conn.commit()
    return {
        "selection_hash": manifest["selection_hash"],
        "selected_case_count": len(cases),
        "referenced_prediction_count": len(target_rows),
        "newly_materialized_count": newly_materialized,
        "already_complete_count": already_complete,
    }


_SLEEP_METADATA_CACHE: dict[str, dict[str, Any]] = {}


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    mode = parser.add_mutually_exclusive_group(required=True)
    mode.add_argument("--database")
    mode.add_argument("--materialize-selection", type=Path)
    parser.add_argument("--confirm-reset-target")
    return parser


def main(argv: list[str] | None = None) -> int:
    parser = build_parser()
    args = parser.parse_args(argv)
    if args.materialize_selection is not None:
        try:
            ensure_authorized_database()
            validation = validate_artifacts()
            summary = materialize_selection(
                args.materialize_selection, validation["apnea"]
            )
        except Exception as exc:
            print(f"ERROR: {type(exc).__name__}: {exc}", file=sys.stderr)
            return 1
        print(json.dumps(summary, indent=2, sort_keys=True, default=_json_default))
        return 0
    if args.database != "pdm":
        parser.error("--database must be exactly 'pdm'")
    if args.confirm_reset_target != "JAISE2026_pdm_project_pgdata":
        parser.error("--confirm-reset-target must exactly match JAISE2026_pdm_project_pgdata")
    try:
        ensure_authorized_database()
        validation = validate_artifacts()
        summaries = [populate_dataset(dataset, validation[dataset]) for dataset in ("apnea", "sleep")]
        refresh_aggregate()
    except Exception as exc:
        print(f"ERROR: {type(exc).__name__}: {exc}", file=sys.stderr)
        return 1
    print(json.dumps(summaries, indent=2, sort_keys=True, default=_json_default))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
