"""Canonical model evidence, privacy separation, and package hashing."""

from __future__ import annotations

import copy
import hashlib
import json
from datetime import date, datetime
from decimal import Decimal
from pathlib import Path

from .db import connect
from .retrieval import (
    get_clinical_context,
    get_detailed_patient_history,
    get_patient_recurrence_summary,
    get_prediction_trend,
)


ROOT = Path(__file__).resolve().parents[1]
EXPERIMENT_ROOT = ROOT / "outputs" / "narrative_experiments" / "apnea_ecg"
PROMPT_PATH = EXPERIMENT_ROOT / "prompt_v1.txt"
POLICY_PATH = EXPERIMENT_ROOT / "generation_policy_v2.json"
PROMPT_V3_PATH = EXPERIMENT_ROOT / "prompt_v3.txt"
POLICY_V3_PATH = EXPERIMENT_ROOT / "generation_policy_v3.json"

def compute_anomaly_margin(error: float, threshold: float) -> dict:
    """Return threshold distance and model-only anomaly decision.

    Margins are raw distances from the detector threshold. They are not
    probabilities or calibrated confidence values. A positive normalized
    margin means the reconstruction error is above a strictly positive
    threshold; zero is exactly on the threshold.
    """

    import math

    error = float(error)
    threshold = float(threshold)
    if not math.isfinite(threshold) or threshold <= 0:
        raise ValueError("anomaly threshold must be a finite positive number")
    if not math.isfinite(error):
        raise ValueError("reconstruction error must be finite")

    margin = error - threshold
    normalized = margin / threshold
    is_anomaly = error > threshold
    decision = "anomaly" if is_anomaly else "normal"
    return {
        "error": error,
        "anomaly_threshold": threshold,
        "threshold": threshold,
        "anomaly_margin": margin,
        "normalized_margin": normalized,
        "is_anomaly": is_anomaly,
        "decision": decision,
        "model_decision": decision,
        "predicted_class": decision,
    }

def compute_channel_evidence(x, reconstruction, thresholds, times) -> list[dict]:
    """Summarize per-channel residual evidence in model (normalized) space.

    ``x`` and ``reconstruction`` may describe one window with shape
    ``(time, channel)`` or a batch of windows with shape
    ``(window, time, channel)``.  The residual is squared before aggregation,
    matching the model's reconstruction-error convention.  ``thresholds``
    must contain one validation-derived threshold per channel.  ``times`` may
    be one time axis shared by all windows or one axis per window.

    The returned records intentionally contain channel residual evidence only;
    shared temporal attention is a window-level artifact and is not attached
    to any of these records.
    """

    import numpy as np

    try:
        values = np.asarray(x, dtype="float64")
        predicted = np.asarray(reconstruction, dtype="float64")
    except (TypeError, ValueError) as exc:
        raise ValueError("x and reconstruction must be numeric arrays") from exc

    if values.ndim == 2:
        values = values[None, ...]
    if predicted.ndim == 2:
        predicted = predicted[None, ...]
    if values.ndim != 3 or predicted.ndim != 3:
        raise ValueError("x and reconstruction must have shape (time, channels) or (windows, time, channels)")
    if values.shape != predicted.shape:
        raise ValueError("x and reconstruction must have identical shapes")
    windows, time_points, channels = values.shape
    if not windows or not time_points or not channels:
        raise ValueError("x and reconstruction must contain at least one window, time point, and channel")
    if not np.all(np.isfinite(values)) or not np.all(np.isfinite(predicted)):
        raise ValueError("x and reconstruction must contain only finite values")

    try:
        channel_thresholds = np.asarray(thresholds, dtype="float64")
    except (TypeError, ValueError) as exc:
        raise ValueError("thresholds must be a numeric vector with one value per channel") from exc
    if channel_thresholds.ndim != 1 or channel_thresholds.shape[0] != channels:
        raise ValueError("thresholds must contain exactly one value per channel")
    if not np.all(np.isfinite(channel_thresholds)) or np.any(channel_thresholds <= 0):
        raise ValueError("channel thresholds must be finite and strictly positive")

    # Keep timestamps opaque (numeric, datetime, or strings are all useful to
    # callers), while validating only their required window/time dimensions.
    timestamp_values = np.asarray(times)
    if timestamp_values.ndim == 1:
        if timestamp_values.shape[0] != time_points:
            raise ValueError("times must contain one value per time point")
        timestamp_values = np.broadcast_to(timestamp_values, (windows, time_points))
    elif timestamp_values.ndim == 2:
        if timestamp_values.shape != (windows, time_points):
            raise ValueError("two-dimensional times must match window and time dimensions")
    else:
        raise ValueError("times must have shape (time,) or (windows, time)")
    if np.issubdtype(timestamp_values.dtype, np.number) and not np.all(np.isfinite(timestamp_values)):
        raise ValueError("numeric times must be finite")

    squared_residual = np.square(values - predicted)
    mean_errors = np.mean(squared_residual, axis=(0, 1))
    max_errors = np.max(squared_residual, axis=(0, 1))
    total_error = float(np.sum(mean_errors))
    if total_error > 0:
        shares = mean_errors / total_error
        # Guard against tiny floating point drift while preserving non-negative
        # contribution shares for downstream JSON/database consumers.
        shares = np.maximum(shares, 0.0)
        shares = shares / float(np.sum(shares))
    else:
        shares = np.zeros(channels, dtype="float64")

    records: list[dict] = []
    for channel_index in range(channels):
        channel_residual = squared_residual[:, :, channel_index]
        peak_window, peak_time = np.unravel_index(
            int(np.argmax(channel_residual)), channel_residual.shape
        )
        mean_error = float(mean_errors[channel_index])
        threshold = float(channel_thresholds[channel_index])
        raw_margin = mean_error - threshold
        normalized_margin = raw_margin / threshold
        exceeds = bool(mean_error > threshold)
        peak_timestamp = timestamp_values[peak_window, peak_time]
        if hasattr(peak_timestamp, "item"):
            peak_timestamp = peak_timestamp.item()
        records.append(
            {
                "channel_index": channel_index,
                "mean_error": mean_error,
                "mean_reconstruction_error": mean_error,
                "max_error": float(max_errors[channel_index]),
                "max_reconstruction_error": float(max_errors[channel_index]),
                "peak_residual_time": peak_timestamp,
                "contribution_share": float(shares[channel_index]),
                "validation_threshold": threshold,
                "channel_threshold": threshold,
                "threshold": threshold,
                "raw_margin": float(raw_margin),
                "anomaly_margin": float(raw_margin),
                "margin": float(raw_margin),
                "normalized_margin": float(normalized_margin),
                "residual_threshold_flag": exceeds,
                "residual_exceeds_threshold": exceeds,
                "is_anomaly": exceeds,
            }
        )
    return records

def canonical_json_hash(payload: dict) -> str:
    """Return SHA-256 over stable compact UTF-8 JSON."""

    def json_default(value):
        if isinstance(value, (datetime, date)):
            return value.isoformat()
        if isinstance(value, Decimal):
            return float(value)
        raise TypeError(f"cannot canonicalize {type(value)!r}")

    encoded = json.dumps(
        payload,
        sort_keys=True,
        separators=(",", ":"),
        ensure_ascii=False,
        allow_nan=False,
        default=json_default,
    ).encode("utf-8")
    return hashlib.sha256(encoded).hexdigest()


def _catalog_block(
    evidence_id: str,
    scope: str,
    evidence_type: str,
    facts: dict,
    source_evidence_ids: list[str] | None = None,
) -> dict:
    """Return one compact, citation-ready evidence block."""

    block = {
        "evidence_id": evidence_id,
        "scope": scope,
        "evidence_type": evidence_type,
        "facts": facts,
    }
    if source_evidence_ids:
        block["source_evidence_ids"] = list(source_evidence_ids)
    return block


def _history_summary(current: dict, rows: list[dict]) -> dict | None:
    if not rows:
        return None
    identity = current["identity"]
    errors = [float(row["reconstruction_error"]) for row in rows]
    classes = [str(row["predicted_class"]) for row in rows]
    timestamps = [row["timestamp"] for row in rows]
    return _catalog_block(
        f"history:patient:{identity['patient_id']}:recent-summary:{identity['prediction_id']}",
        "context",
        "recent_history_summary",
        {
            "event_count": len(rows),
            "anomaly_count": classes.count("anomaly"),
            "normal_count": classes.count("normal"),
            "most_recent_time": timestamps[0],
            "earliest_time": timestamps[-1],
            "minimum_reconstruction_error": min(errors),
            "maximum_reconstruction_error": max(errors),
            "events": [
                {
                    key: row.get(key)
                    for key in (
                        "prediction_id", "timestamp", "predicted_class",
                        "reconstruction_error", "anomaly_margin", "normalized_margin",
                    )
                    if key in row
                }
                for row in rows
            ],
        },
        [str(row["evidence_id"]) for row in rows],
    )


def _trend_summary(current: dict, rows: list[dict]) -> dict | None:
    if not rows:
        return None
    identity = current["identity"]
    ordered = sorted(rows, key=lambda row: str(row["bucket"]))
    means = [float(row["mean_reconstruction_error"]) for row in ordered]
    anomaly_counts = [int(row["anomaly_count"]) for row in ordered]
    anomaly_rates = [float(row["anomaly_rate"]) for row in ordered]
    if means[-1] > means[0]:
        direction = "increased"
    elif means[-1] < means[0]:
        direction = "decreased"
    else:
        direction = "unchanged"
    return _catalog_block(
        f"aggregate:patient:{identity['patient_id']}:recent-summary:{identity['prediction_id']}",
        "context",
        "aggregate_trend_summary",
        {
            "bucket_count": len(ordered),
            "first_bucket": ordered[0]["bucket"],
            "last_bucket": ordered[-1]["bucket"],
            "first_mean_reconstruction_error": means[0],
            "last_mean_reconstruction_error": means[-1],
            "minimum_mean_reconstruction_error": min(means),
            "maximum_mean_reconstruction_error": max(means),
            "endpoint_direction": direction,
            "minimum_anomaly_count": min(anomaly_counts),
            "maximum_anomaly_count": max(anomaly_counts),
            "minimum_anomaly_rate": min(anomaly_rates),
            "maximum_anomaly_rate": max(anomaly_rates),
        },
        [str(row["evidence_id"]) for row in ordered],
    )


def build_citation_evidence(current_evidence: dict, pdm_context: dict) -> list[dict]:
    """Build the compact evidence catalog exposed to the v3 generator.

    The catalog deliberately assigns one native citation owner to facts that
    were split across nested v2 blocks. Detailed source rows remain in the
    canonical package and are recorded through ``source_evidence_ids``.
    """

    current = current_evidence
    identity = current["identity"]
    event = current["event"]
    detector = current["model_evidence"]
    signal = current.get("signal_evidence", {})
    attention = current.get("attention_evidence", {})
    xai = current.get("xai_evidence", {})
    explanations = xai.get("explanations", [])
    raw_streams = signal.get("streams", [])
    primary = raw_streams[0] if raw_streams else {}

    catalog = [
        _catalog_block(
            str(event["evidence_id"]),
            "current",
            "current_event_summary",
            {
                key: event.get(key)
                for key in (
                    "modality", "source_recording_name", "window_index",
                    "start_time", "end_time", "duration_seconds",
                )
            },
        ),
        _catalog_block(
            str(detector["evidence_id"]),
            "current",
            "detector_summary",
            {
                key: detector.get(key)
                for key in (
                    "predicted_class", "is_anomaly", "reconstruction_error",
                    "anomaly_threshold", "anomaly_margin", "normalized_margin",
                    "semantics",
                )
            },
        ),
        _catalog_block(
            str(current["model_provenance"]["evidence_id"]),
            "current",
            "model_provenance",
            copy.deepcopy(current["model_provenance"] | {"evidence_id": None}),
        ),
        _catalog_block(
            f"window:{identity['window_id']}:signal-summary",
            "current",
            "signal_summary",
            {
                "available": bool(signal.get("available")),
                "primary_channel": primary.get("channel"),
                "primary_sample_count": primary.get("sample_count"),
                "primary_sampling_rate_hz": primary.get("sampling_rate_hz"),
                "channel_names": [row.get("channel") for row in raw_streams],
            },
            [str(signal["evidence_id"])] if signal.get("evidence_id") else None,
        ),
        _catalog_block(
            f"window:{identity['window_id']}:attention-summary",
            "current",
            "attention_summary",
            {
                "available": bool(attention.get("available")),
                "sample_count": attention.get("sample_count"),
                "minimum": attention.get("minimum"),
                "maximum": attention.get("maximum"),
                "mean": attention.get("mean"),
                "peak_time": attention.get("peak_time"),
                "method": explanations[0].get("method") if explanations else None,
                "semantics": (
                    "shared window-level temporal relevance; not probability or calibrated "
                    "confidence; not channel-specific importance; no channel owner"
                ),
                "channel_level_evidence_available": bool(xai.get("channel_evidence")),
            },
            [
                evidence_id for evidence_id in (
                    attention.get("evidence_id"), xai.get("evidence_id")
                ) if evidence_id
            ],
        ),
        _catalog_block(
            f"pdm:patient:{identity['patient_id']}:status:{identity['prediction_id']}",
            "package",
            "pdm_context_status",
            {
                "included": bool(pdm_context.get("included")),
                "history_supplied": bool(pdm_context.get("detailed_history")),
                "aggregate_trend_supplied": bool(pdm_context.get("aggregate_trend")),
                "recurrence_summary_supplied": pdm_context.get("recurrence_summary") is not None,
            },
        ),
    ]

    history = _history_summary(current, list(pdm_context.get("detailed_history", [])))
    trend = _trend_summary(current, list(pdm_context.get("aggregate_trend", [])))
    if history:
        catalog.append(history)
    if trend:
        catalog.append(trend)
    recurrence = pdm_context.get("recurrence_summary")
    if isinstance(recurrence, dict) and recurrence.get("evidence_id"):
        catalog.append(_catalog_block(
            str(recurrence["evidence_id"]),
            "context",
            "recurrence_summary",
            {key: copy.deepcopy(value) for key, value in recurrence.items() if key != "evidence_id"},
        ))

    clinical = pdm_context.get("clinical_context", {})
    visits = list(clinical.get("visits", [])) if isinstance(clinical, dict) else []
    diagnoses = list(clinical.get("diagnoses", [])) if isinstance(clinical, dict) else []
    catalog.append(_catalog_block(
        f"clinical:patient:{identity['patient_id']}:summary:{identity['prediction_id']}",
        "context" if pdm_context.get("included") else "package",
        "clinical_context_summary",
        {
            "supplied": bool(visits or diagnoses),
            "source": clinical.get("source", "synthetic_prototype") if isinstance(clinical, dict) else "synthetic_prototype",
            "visit_count": len(visits),
            "diagnosis_count": len(diagnoses),
            "visits": copy.deepcopy(visits),
            "diagnoses": copy.deepcopy(diagnoses),
        },
        [
            str(row["evidence_id"])
            for row in visits + diagnoses
            if isinstance(row, dict) and row.get("evidence_id")
        ],
    ))

    identifiers = [row["evidence_id"] for row in catalog]
    if len(identifiers) != len(set(identifiers)):
        raise ValueError("citation evidence IDs must be unique")
    return catalog


def build_concept_checklist(current_evidence: dict, pdm_context: dict) -> dict:
    """Return the v3 current and context concepts used for coverage scoring."""

    current = current_evidence
    identity = current["identity"]
    event = current["event"]
    detector = current["model_evidence"]
    signal = current.get("signal_evidence", {})
    primary = signal.get("streams", [{}])[0] if signal.get("streams") else {}
    attention = current.get("attention_evidence", {})
    current_rows = [
        {
            "concept_id": "event_timing", "evidence_id": event["evidence_id"],
            "match_fields": ["start_time", "duration_seconds"],
            "match_values": [event.get("start_time"), event.get("duration_seconds")],
        },
        {
            "concept_id": "detector_decision", "evidence_id": detector["evidence_id"],
            "match_fields": ["predicted_class"],
            "match_values": [detector.get("predicted_class")],
        },
        {
            "concept_id": "reconstruction_comparison", "evidence_id": detector["evidence_id"],
            "match_fields": ["reconstruction_error", "anomaly_threshold"],
            "match_values": [detector.get("reconstruction_error"), detector.get("anomaly_threshold")],
        },
        {
            "concept_id": "detector_margin", "evidence_id": detector["evidence_id"],
            "match_fields": ["anomaly_margin", "normalized_margin"],
            "match_values": [detector.get("anomaly_margin"), detector.get("normalized_margin")],
        },
        {
            "concept_id": "signal_evidence",
            "evidence_id": f"window:{identity['window_id']}:signal-summary",
            "match_fields": ["primary_channel", "primary_sample_count"],
            "match_values": [primary.get("channel"), primary.get("sample_count")],
        },
        {
            "concept_id": "attention_result",
            "evidence_id": f"window:{identity['window_id']}:attention-summary",
            "match_fields": ["peak_time", "maximum"],
            "match_values": [attention.get("peak_time"), attention.get("maximum")],
        },
        {
            "concept_id": "xai_interpretation",
            "evidence_id": f"window:{identity['window_id']}:attention-summary",
            "match_fields": ["semantics"],
            "match_values": ["shared window-level temporal relevance", "no channel owner"],
        },
    ]
    if not signal.get("available"):
        current_rows = [row for row in current_rows if row["concept_id"] != "signal_evidence"]
    if not attention.get("available"):
        current_rows = [
            row for row in current_rows
            if row["concept_id"] not in {"attention_result", "xai_interpretation"}
        ]

    catalog = build_citation_evidence(current, pdm_context)
    by_type = {row["evidence_type"]: row for row in catalog}
    context_rows = []
    for concept_id, evidence_type in (
        ("recent_history", "recent_history_summary"),
        ("aggregate_trend", "aggregate_trend_summary"),
        ("recurrence", "recurrence_summary"),
        ("clinical_context", "clinical_context_summary"),
    ):
        block = by_type.get(evidence_type)
        if block is None or block["scope"] != "context":
            continue
        context_rows.append({
            "concept_id": concept_id,
            "evidence_id": block["evidence_id"],
            "match_fields": list(block["facts"]),
            "match_values": [value for value in block["facts"].values() if value is not None],
        })
    return {
        "schema_version": "fusion-evaluation-concepts-v3",
        "current": current_rows,
        "context": context_rows,
    }


def _generation_contract_metadata() -> dict:
    prompt = PROMPT_PATH.read_text(encoding="utf-8")
    policy = json.loads(POLICY_PATH.read_text(encoding="utf-8"))
    prompt_hash = hashlib.sha256(prompt.encode("utf-8")).hexdigest()
    if policy.get("prompt_sha256") != prompt_hash:
        raise ValueError("frozen prompt hash does not match generation policy")
    return {
        "prompt_version": policy["prompt_version"],
        "prompt_sha256": prompt_hash,
        "generation_policy_version": policy["policy_version"],
        "generation_policy_sha256": hashlib.sha256(POLICY_PATH.read_bytes()).hexdigest(),
        "provider": policy["provider"],
        "llm_model_id": policy["model"],
        "generation_settings": copy.deepcopy(policy["generation_settings"]),
    }

def build_current_evidence(prediction_id: int) -> dict:
    """Build model-only current evidence from one persisted prediction."""

    query = """
        SELECT ps.prediction_id, ps.patient_id, p.patient_code,
               ps.recording_id, ps.window_id, ps.model_id,
               r.modality, r.source_recording_name,
               w.window_index, w.start_sample, w.end_sample,
               w.start_time, w.end_time, w.duration_seconds,
               ps.predicted_class, ps.reconstruction_error,
               ps.anomaly_threshold, ps.anomaly_margin,
               ps.normalized_margin, ps.is_anomaly,
               m.model_name, m.model_version, m.architecture,
               m.training_dataset, m.input_configuration
        FROM prediction_series AS ps
        JOIN patient AS p ON p.patient_id = ps.patient_id
        JOIN recording AS r ON r.recording_id = ps.recording_id
        JOIN analysis_window AS w ON w.window_id = ps.window_id
        JOIN model AS m ON m.model_id = ps.model_id
        WHERE ps.prediction_id = %s
    """
    with connect() as conn, conn.cursor() as cur:
        cur.execute(query, (int(prediction_id),))
        row = cur.fetchone()
        if row is None:
            raise ValueError(f"prediction not found: {prediction_id}")
        (
            prediction_id, patient_id, patient_code, recording_id, window_id,
            model_id, modality, source_recording_name, window_index,
            start_sample, end_sample, start_time, end_time, duration_seconds,
            predicted_class, reconstruction_error, anomaly_threshold,
            anomaly_margin, normalized_margin, is_anomaly, model_name,
            model_version, architecture, training_dataset, input_configuration,
        ) = row
        if predicted_class not in {"anomaly", "normal"}:
            raise ValueError(f"invalid detector class for prediction {prediction_id}")
        expected = compute_anomaly_margin(reconstruction_error, anomaly_threshold)
        tolerance = 1e-9
        if (
            abs(float(anomaly_margin) - expected["anomaly_margin"]) > tolerance
            or abs(float(normalized_margin) - expected["normalized_margin"]) > tolerance
            or bool(is_anomaly) != expected["is_anomaly"]
            or predicted_class != expected["predicted_class"]
        ):
            raise ValueError(f"inconsistent detector evidence for prediction {prediction_id}")
        try:
            model_config = json.loads(input_configuration or "{}")
        except json.JSONDecodeError as exc:
            raise ValueError(f"invalid model provenance for prediction {prediction_id}") from exc

        def signal_summary(signal_type: str) -> dict:
            cur.execute(
                """
                SELECT s.signal_id, s.channel, s.unit, s.sampling_rate_hz,
                       COUNT(sd.*), MIN(sd.value), MAX(sd.value), AVG(sd.value)
                FROM window_signal AS ws
                JOIN signal AS s ON s.signal_id = ws.signal_id
                LEFT JOIN signal_data AS sd
                  ON sd.signal_id = s.signal_id
                 AND sd.timestamp >= %s AND sd.timestamp < %s
                WHERE ws.window_id = %s AND s.signal_type = %s
                GROUP BY s.signal_id, s.channel, s.unit, s.sampling_rate_hz
                HAVING COUNT(sd.*) > 0
                ORDER BY s.signal_id
                """,
                (start_time, end_time, window_id, signal_type),
            )
            streams = [
                {
                    "signal_id": values[0], "channel": values[1], "unit": values[2],
                    "sampling_rate_hz": values[3], "sample_count": values[4],
                    "minimum": values[5], "maximum": values[6], "mean": values[7],
                }
                for values in cur.fetchall()
            ]
            count = sum(int(stream["sample_count"]) for stream in streams)
            result = {"available": bool(streams), "stored_value_count": count, "streams": streams}
            if not streams:
                result["reason"] = "dense evidence not materialized for this window"
            return result

        raw_summary = signal_summary("raw")
        reconstruction_summary = signal_summary("reconstruction")
        residual_summary = signal_summary("residual")

        cur.execute(
            """
            SELECT COUNT(*), MIN(attention_value), MAX(attention_value),
                   AVG(attention_value)
            FROM attention_data
            WHERE window_id = %s AND timestamp >= %s AND timestamp < %s
            """,
            (window_id, start_time, end_time),
        )
        attention_count, attention_min, attention_max, attention_mean = cur.fetchone()
        cur.execute(
            """
            SELECT timestamp, attention_value
            FROM attention_data
            WHERE window_id = %s AND timestamp >= %s AND timestamp < %s
            ORDER BY attention_value DESC, timestamp ASC
            LIMIT 1
            """,
            (window_id, start_time, end_time),
        )
        attention_peak = cur.fetchone()
        attention = {
            "available": bool(attention_count),
            "sample_count": int(attention_count),
            "minimum": attention_min,
            "maximum": attention_max,
            "mean": attention_mean,
            "peak_time": attention_peak[0] if attention_peak else None,
            "semantics": "shared window-level temporal relevance; not probability",
        }
        if not attention_count:
            attention["reason"] = "dense evidence not materialized for this window"

        cur.execute(
            """
            SELECT xe.explanation_id, xe.xai_method, xe.explanation_type,
                   xe.explanation_text
            FROM xai_explanation AS xe
            WHERE xe.prediction_id = %s
            ORDER BY xe.explanation_id
            """,
            (prediction_id,),
        )
        explanations = [
            {
                "explanation_id": values[0], "method": values[1],
                "type": values[2], "description": values[3],
            }
            for values in cur.fetchall()
        ]
        cur.execute(
            """
            SELECT xf.explanation_id, xf.signal_id, s.channel,
                   xf.feature_name, xf.feature_value
            FROM xai_feature AS xf
            JOIN xai_explanation AS xe ON xe.explanation_id = xf.explanation_id
            LEFT JOIN signal AS s ON s.signal_id = xf.signal_id
            WHERE xe.prediction_id = %s
            ORDER BY s.channel, xf.feature_name
            """,
            (prediction_id,),
        )
        channel_map: dict[tuple, dict] = {}
        for explanation_id, signal_id, channel, feature_name, feature_value in cur.fetchall():
            key = (explanation_id, signal_id, channel)
            item = channel_map.setdefault(
                key,
                {
                    "evidence_id": f"xai:{explanation_id}:signal:{signal_id}",
                    "explanation_id": explanation_id,
                    "signal_id": signal_id,
                    "channel": channel,
                    "features": {},
                },
            )
            item["features"][feature_name] = feature_value
        channels = list(channel_map.values())

    return {
        "identity": {
            "patient_id": patient_id,
            "patient_code": patient_code,
            "recording_id": recording_id,
            "window_id": window_id,
            "prediction_id": prediction_id,
            "model_id": model_id,
        },
        "event": {
            "evidence_id": f"window:{window_id}:event",
            "modality": modality,
            "source_recording_name": source_recording_name,
            "window_index": window_index,
            "start_sample": start_sample,
            "end_sample": end_sample,
            "start_time": start_time,
            "end_time": end_time,
            "duration_seconds": duration_seconds,
        },
        "model_evidence": {
            "evidence_id": f"prediction:{prediction_id}:detector",
            "predicted_class": predicted_class,
            "is_anomaly": bool(is_anomaly),
            "reconstruction_error": reconstruction_error,
            "anomaly_threshold": anomaly_threshold,
            "anomaly_margin": anomaly_margin,
            "normalized_margin": normalized_margin,
            "semantics": "reconstruction threshold evidence; not probability or calibrated confidence",
        },
        "model_provenance": {
            "evidence_id": f"model:{model_id}:provenance",
            "model_name": model_name,
            "model_version": model_version,
            "architecture": architecture,
            "training_dataset": training_dataset,
            "fold_id": model_config.get("fold_id"),
            "test_patient": model_config.get("test_patient"),
            "model_checkpoint_sha256": model_config.get("model_checkpoint_sha256"),
            "fold_manifest": model_config.get("fold_manifest"),
        },
        "signal_evidence": {
            "evidence_id": f"window:{window_id}:signal:raw", **raw_summary,
        },
        "reconstruction_evidence": {
            "evidence_id": f"window:{window_id}:signal:reconstruction",
            **reconstruction_summary,
        },
        "residual_evidence": {
            "evidence_id": f"window:{window_id}:signal:residual", **residual_summary,
        },
        "attention_evidence": {
            "evidence_id": f"window:{window_id}:attention", **attention,
        },
        "xai_evidence": {
            "evidence_id": f"prediction:{prediction_id}:xai",
            "available": bool(explanations),
            "explanations": explanations,
            "channel_evidence": channels,
            "channel_semantics": "shared-model channel residual evidence; not an independent detector",
        },
        "provenance": {
            "source": "pdm_database",
            "prediction_series_id": f"prediction:{prediction_id}",
            "window_evidence_id": f"window:{window_id}",
            "patient_evidence_id": f"patient:{patient_id}",
            "dense_evidence_materialized": bool(raw_summary["available"]),
        },
    }

def build_paired_packages(prediction_id: int) -> tuple[dict, dict, dict]:
    """Build paired LLM-ready packages and separate private evaluation data."""

    current_evidence = build_current_evidence(prediction_id)
    identity = current_evidence["identity"]
    event_time = current_evidence["event"]["start_time"]
    patient_id = identity["patient_id"]
    current_hash = canonical_json_hash(current_evidence)
    metadata = {
        "evidence_package_version": 2,
        "source_database_ids": {
            "patient_id": patient_id,
            "recording_id": identity["recording_id"],
            "window_id": identity["window_id"],
            "prediction_id": identity["prediction_id"],
            "model_id": identity["model_id"],
        },
        "model_provenance": copy.deepcopy(current_evidence["model_provenance"]),
        "current_evidence_hash": current_hash,
        **_generation_contract_metadata(),
    }
    empty_context = {
        "included": False,
        "detailed_history": [],
        "aggregate_trend": [],
        "recurrence_summary": None,
        "clinical_context": {
            "source": "synthetic_prototype",
            "visits": [],
            "diagnoses": [],
        },
    }
    current_package = {
        "package_metadata": copy.deepcopy(metadata),
        "current_evidence": copy.deepcopy(current_evidence),
        "pdm_context": empty_context,
        "current_evidence_hash": current_hash,
    }

    detailed_history = get_detailed_patient_history(patient_id, event_time, limit=5)
    for item in detailed_history:
        item["seconds_before_current"] = (
            event_time - item["timestamp"]
        ).total_seconds()
    enabled_context = {
        "included": True,
        "history_policy": "five latest same-patient events strictly before current event",
        "trend_policy": "twelve latest earlier five-minute aggregate buckets",
        "detailed_history": detailed_history,
        "aggregate_trend": get_prediction_trend(patient_id, event_time, limit=12),
        "recurrence_summary": get_patient_recurrence_summary(patient_id, event_time),
        "clinical_context": get_clinical_context(patient_id, event_time),
    }
    pdm_package = {
        "package_metadata": copy.deepcopy(metadata),
        "current_evidence": copy.deepcopy(current_evidence),
        "pdm_context": enabled_context,
        "current_evidence_hash": current_hash,
    }
    current_package["package_hash"] = canonical_json_hash(current_package)
    pdm_package["package_hash"] = canonical_json_hash(pdm_package)

    with connect() as conn, conn.cursor() as cur:
        cur.execute(
            """
            SELECT a.annotation_id, a.annotation_type, a.label, a.source
            FROM annotation AS a
            WHERE a.window_id = %s
              AND a.source IN ('official_evaluation_only', 'derived_private_evaluation')
            ORDER BY a.annotation_id
            """,
            (identity["window_id"],),
        )
        annotation_rows = cur.fetchall()
    official = [row for row in annotation_rows if row[3] == "official_evaluation_only"]
    official_labels = [row[2] for row in official]
    category = None
    if current_evidence["event"]["modality"] == "apnea" and official_labels:
        positive = official_labels[0] == "A"
        anomalous = bool(current_evidence["model_evidence"]["is_anomaly"])
        category = (
            "TP" if positive and anomalous
            else "FN" if positive
            else "FP" if anomalous
            else "TN"
        )
    evaluation_only = {
        "evaluation_only": True,
        "private_model_development_data": True,
        "prediction_id": identity["prediction_id"],
        "window_id": identity["window_id"],
        "official_labels": official_labels,
        "case_category": category,
        "annotation_records": [
            {
                "annotation_id": row[0],
                "annotation_type": row[1],
                "label": row[2],
                "source": row[3],
            }
            for row in annotation_rows
        ],
        "separation_policy": "never include this object in an LLM request",
    }
    return current_package, pdm_package, evaluation_only


def _v3_generation_contract_metadata() -> dict:
    prompt = PROMPT_V3_PATH.read_text(encoding="utf-8")
    policy = json.loads(POLICY_V3_PATH.read_text(encoding="utf-8"))
    prompt_hash = hashlib.sha256(prompt.encode("utf-8")).hexdigest()
    if policy.get("prompt_sha256") != prompt_hash:
        raise ValueError("frozen v3 prompt hash does not match generation policy")
    if policy.get("prompt_version") != "fusion_prompt_v3":
        raise ValueError("unsupported v3 prompt version")
    if policy.get("policy_version") != "generation_policy_v3":
        raise ValueError("unsupported v3 generation policy version")
    return {
        "prompt_version": policy["prompt_version"],
        "prompt_sha256": prompt_hash,
        "generation_policy_version": policy["policy_version"],
        "generation_policy_sha256": hashlib.sha256(POLICY_V3_PATH.read_bytes()).hexdigest(),
        "provider": policy["provider"],
        "llm_model_id": policy["model"],
        "generation_settings": copy.deepcopy(policy["generation_settings"]),
        "response_schema_version": policy["response_schema_version"],
    }


def build_paired_packages_v3(prediction_id: int) -> tuple[dict, dict, dict]:
    """Build v3 packages without mutating or replacing frozen v2 artifacts."""

    current, enabled, private = build_paired_packages(prediction_id)
    generation = _v3_generation_contract_metadata()
    for package in (current, enabled):
        package["package_metadata"].update(generation)
        package["package_metadata"]["evidence_package_version"] = 3
        package["citation_evidence"] = build_citation_evidence(
            package["current_evidence"], package["pdm_context"]
        )
        package["evaluation_concepts"] = build_concept_checklist(
            package["current_evidence"], package["pdm_context"]
        )
        package.pop("package_hash", None)
        package["package_hash"] = canonical_json_hash(package)
    return current, enabled, private
