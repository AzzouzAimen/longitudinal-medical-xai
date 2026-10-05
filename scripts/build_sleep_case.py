"""Build, generate, persist, and verify the representative Sleep-EDF case."""

from __future__ import annotations

import hashlib
import json
import os
import time
from datetime import date, datetime
from decimal import Decimal
from pathlib import Path
from typing import Any

from dotenv import load_dotenv

from pdm.db import connect
from pdm.evidence import canonical_json_hash
from pdm.llm import (
    BASE_URL, MODEL_ID, _retryable_transport_error, load_frozen_generation_contract,
    parse_structured_response, verify_model,
)
from scripts.populate_fusion_pdm import _sleep_recording_metadata
from scripts.run_fusion_experiment import validate_response_evidence


ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "outputs" / "narrative_experiments" / "sleep_edf_case"
PROMPT_PATH = OUT / "prompt_v1.txt"
MODELED = ("EEG Fpz-Cz", "EEG Pz-Oz", "EOG horizontal")
AUXILIARY = ("Resp oro-nasal", "EMG submental", "Temp rectal")


def _jsonable(value: Any) -> Any:
    if isinstance(value, (datetime, date)):
        return value.isoformat()
    if isinstance(value, Decimal):
        return float(value)
    raise TypeError(type(value).__name__)


def _write_json(path: Path, value: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        json.dumps(value, indent=2, sort_keys=True, ensure_ascii=False, default=_jsonable) + "\n",
        encoding="utf-8",
    )


def select_case() -> tuple[int, list[dict]]:
    """Select without looking at generated narrative quality."""

    query = """
        SELECT pr.prediction_id,p.patient_code,r.recording_id,w.window_index,w.start_time,
               pr.reconstruction_error,pr.anomaly_threshold,
               (SELECT count(*) FROM xai_feature xf
                WHERE xf.explanation_id=x.explanation_id
                  AND xf.feature_name='residual_threshold_flag' AND xf.feature_value=1) AS flagged_channels,
               (SELECT count(DISTINCT r2.recording_id) FROM recording r2
                WHERE r2.patient_id=p.patient_id AND r2.modality='sleep'
                  AND r2.recording_start_time<r.recording_start_time) AS prior_nights,
               (SELECT count(*) FROM signal s WHERE s.recording_id=r.recording_id
                  AND s.signal_type='raw' AND s.channel=ANY(%s)) AS auxiliary_channels
        FROM prediction pr
        JOIN analysis_window w ON w.window_id=pr.window_id
        JOIN recording r ON r.recording_id=w.recording_id
        JOIN patient p ON p.patient_id=r.patient_id
        JOIN xai_explanation x ON x.prediction_id=pr.prediction_id
        WHERE r.modality='sleep' AND pr.is_anomaly
        ORDER BY prior_nights DESC, flagged_channels DESC,
                 pr.reconstruction_error/pr.anomaly_threshold DESC, pr.prediction_id
    """
    with connect() as conn, conn.cursor() as cur:
        cur.execute(query, (list(AUXILIARY),))
        columns = [description[0] for description in cur.description]
        candidates = [dict(zip(columns, row)) for row in cur.fetchall()]
    eligible = [
        row for row in candidates
        if int(row["prior_nights"]) >= 1
        and int(row["flagged_channels"]) >= 2
        and int(row["auxiliary_channels"]) == len(AUXILIARY)
    ]
    if not eligible:
        raise ValueError("no Sleep case meets the frozen representativeness criteria")
    return int(eligible[0]["prediction_id"]), candidates


def _channel_evidence(cur, explanation_id: int) -> list[dict]:
    cur.execute(
        """
        SELECT s.signal_id,s.channel,xf.feature_name,xf.feature_value
        FROM xai_feature xf JOIN signal s ON s.signal_id=xf.signal_id
        WHERE xf.explanation_id=%s ORDER BY s.channel,xf.feature_name
        """,
        (explanation_id,),
    )
    by_channel: dict[str, dict] = {}
    for signal_id, channel, name, value in cur.fetchall():
        row = by_channel.setdefault(channel, {
            "evidence_id": f"sleep-channel:{channel}", "signal_id": int(signal_id),
            "channel": channel,
            "semantics": "component evidence from one shared multivariate model; not an independent detector",
        })
        row[name] = bool(value) if name == "residual_threshold_flag" else float(value)
    cur.execute(
        "SELECT signal_id,temporal_start FROM xai_location WHERE explanation_id=%s",
        (explanation_id,),
    )
    peaks = {int(signal_id): timestamp for signal_id, timestamp in cur.fetchall()}
    for row in by_channel.values():
        row["peak_residual_time"] = peaks[row["signal_id"]]
    return [by_channel[channel] for channel in MODELED]


def build_package(prediction_id: int) -> dict:
    with connect() as conn, conn.cursor() as cur:
        cur.execute(
            """
            SELECT p.patient_id,p.patient_code,r.recording_id,r.source_recording_name,
                   r.recording_start_time,w.window_id,w.window_index,w.start_sample,w.end_sample,
                   w.start_time,w.end_time,w.duration_seconds,pr.model_id,pr.reconstruction_error,
                   pr.anomaly_threshold,pr.is_anomaly,m.model_name,m.model_version,m.input_configuration,
                   x.explanation_id,x.xai_method,x.explanation_type,x.explanation_text,
                   a.annotation_id,a.label,a.source,a.annotation_type
            FROM prediction pr JOIN analysis_window w ON w.window_id=pr.window_id
            JOIN recording r ON r.recording_id=w.recording_id
            JOIN patient p ON p.patient_id=r.patient_id JOIN model m ON m.model_id=pr.model_id
            JOIN xai_explanation x ON x.prediction_id=pr.prediction_id
            JOIN annotation a ON a.window_id=w.window_id AND a.source='official_evaluation_only'
            WHERE pr.prediction_id=%s
            """,
            (prediction_id,),
        )
        row = cur.fetchone()
        if not row:
            raise ValueError(f"Sleep prediction is incomplete: {prediction_id}")
        columns = [description[0] for description in cur.description]
        record = dict(zip(columns, row))
        channel_evidence = _channel_evidence(cur, int(record["explanation_id"]))
        cur.execute(
            """SELECT count(*),min(attention_value),max(attention_value),avg(attention_value),
                      (array_agg(timestamp ORDER BY attention_value DESC))[1]
               FROM attention_data WHERE window_id=%s""",
            (record["window_id"],),
        )
        attention_count, att_min, att_max, att_mean, att_peak = cur.fetchone()
        auxiliary = []
        for channel in AUXILIARY:
            cur.execute(
                """
                SELECT s.signal_id,s.sampling_rate_hz,count(sd.*),min(sd.value),max(sd.value),avg(sd.value)
                FROM signal s JOIN signal_data sd ON sd.signal_id=s.signal_id
                WHERE s.recording_id=%s AND s.signal_type='raw' AND s.channel=%s
                  AND sd.timestamp >= %s AND sd.timestamp < %s
                GROUP BY s.signal_id,s.sampling_rate_hz
                """,
                (record["recording_id"], channel, record["start_time"], record["end_time"]),
            )
            signal_id, rate, count, minimum, maximum, mean = cur.fetchone()
            auxiliary.append({
                "evidence_id": f"sleep-aux:{record['window_id']}:{channel}",
                "signal_id": int(signal_id), "channel": channel,
                "sampling_rate_hz": float(rate), "sample_count": int(count),
                "minimum": float(minimum), "maximum": float(maximum), "mean": float(mean),
                "semantics": "auxiliary context only; not modeled reconstruction input",
            })
        cur.execute(
            """
            SELECT r2.recording_id,r2.source_recording_name,r2.recording_start_time,
                   min(w2.start_time),max(w2.end_time),count(*),
                   count(*) FILTER (WHERE p2.is_anomaly),avg(p2.reconstruction_error),max(p2.reconstruction_error)
            FROM recording r2 JOIN analysis_window w2 ON w2.recording_id=r2.recording_id
            JOIN prediction p2 ON p2.window_id=w2.window_id
            WHERE r2.patient_id=%s AND r2.modality='sleep'
              AND r2.recording_start_time < %s
            GROUP BY r2.recording_id ORDER BY r2.recording_start_time DESC LIMIT 1
            """,
            (record["patient_id"], record["recording_start_time"]),
        )
        prior = cur.fetchone()
        if not prior:
            raise ValueError("selected Sleep case has no previous night")
        prior_id = prior[0]
        cur.execute(
            """SELECT a.label,count(*) FROM annotation a
               WHERE a.recording_id=%s AND a.source='official_evaluation_only'
               GROUP BY a.label ORDER BY a.label""",
            (prior_id,),
        )
        stage_counts = {label: int(count) for label, count in cur.fetchall()}
        cur.execute(
            """
            SELECT * FROM (
              SELECT p2.prediction_id,w2.start_time,p2.reconstruction_error,p2.anomaly_threshold,
                     p2.is_anomaly,a2.label
              FROM prediction p2 JOIN analysis_window w2 ON w2.window_id=p2.window_id
              JOIN annotation a2 ON a2.window_id=w2.window_id AND a2.source='official_evaluation_only'
              WHERE w2.recording_id=%s ORDER BY p2.reconstruction_error/p2.anomaly_threshold DESC LIMIT 5
            ) ranked ORDER BY start_time
            """,
            (prior_id,),
        )
        prior_events = [{
            "evidence_id": f"sleep-history:prediction:{item[0]}",
            "prediction_id": int(item[0]), "timestamp": item[1],
            "reconstruction_error": float(item[2]), "anomaly_threshold": float(item[3]),
            "anomaly_margin": float(item[2] - item[3]), "is_anomaly": bool(item[4]),
            "recorded_sleep_stage": item[5],
        } for item in cur.fetchall()]

    source_id = record["source_recording_name"]
    metadata = _sleep_recording_metadata(source_id, {})
    model_provenance = json.loads(record["input_configuration"])
    margin = float(record["reconstruction_error"] - record["anomaly_threshold"])
    package = {
        "schema_version": "sleep-fusion-case-v1",
        "selection": {
            "evidence_id": f"sleep-selection:prediction:{prediction_id}",
            "criteria": "second patient night; at least two flagged modeled channels; shared attention; recorded stage; all three auxiliary channels",
        },
        "current_event": {
            "evidence_id": f"sleep-window:{record['window_id']}:event",
            "patient_id": int(record["patient_id"]), "patient_code": record["patient_code"],
            "recording_id": record["recording_id"], "source_recording_name": source_id,
            "prediction_id": prediction_id, "window_id": int(record["window_id"]),
            "window_index": int(record["window_index"]), "start_sample": int(record["start_sample"]),
            "end_sample": int(record["end_sample"]), "start_time": record["start_time"],
            "end_time": record["end_time"], "duration_seconds": float(record["duration_seconds"]),
            "timestamp_provenance": metadata["timestamp_provenance"],
        },
        "detector_evidence": {
            "evidence_id": f"sleep-prediction:{prediction_id}:detector",
            "predicted_class": "anomaly" if record["is_anomaly"] else "normal",
            "reconstruction_error": float(record["reconstruction_error"]),
            "anomaly_threshold": float(record["anomaly_threshold"]),
            "anomaly_margin": margin,
            "normalized_margin": margin / float(record["anomaly_threshold"]),
            "semantics": "shared-model reconstruction outlier relative to validation-normal stable stage 2; not probability, transition detector, or diagnosis",
        },
        "channel_evidence": channel_evidence,
        "attention_evidence": {
            "evidence_id": f"sleep-window:{record['window_id']}:attention",
            "sample_count": int(attention_count), "minimum": float(att_min),
            "maximum": float(att_max), "mean": float(att_mean), "peak_time": att_peak,
            "semantics": "shared window-level temporal relevance; no channel owner; not probability",
        },
        "xai_provenance": {
            "evidence_id": f"sleep-prediction:{prediction_id}:xai",
            "explanation_id": int(record["explanation_id"]), "method": record["xai_method"],
            "type": record["explanation_type"], "description": record["explanation_text"],
        },
        "auxiliary_context": auxiliary,
        "recorded_sleep_stage_context": {
            "evidence_id": f"sleep-annotation:{record['annotation_id']}",
            "annotation_id": int(record["annotation_id"]), "label": record["label"],
            "annotation_type": record["annotation_type"], "source": record["source"],
            "semantics": "Sleep-EDF hypnogram context supplied to the narrative; not a diagnosis or proof of detector correctness",
        },
        "previous_night_context": {
            "evidence_id": f"sleep-night:{prior_id}", "recording_id": prior_id,
            "source_recording_name": prior[1], "recording_start_time": prior[2],
            "retained_start_time": prior[3], "retained_end_time": prior[4],
            "window_count": int(prior[5]), "anomaly_count": int(prior[6]),
            "anomaly_rate": int(prior[6]) / int(prior[5]),
            "mean_reconstruction_error": float(prior[7]), "maximum_reconstruction_error": float(prior[8]),
            "recorded_stage_counts": stage_counts, "representative_events": prior_events,
            "chronology": "previous same-patient recording, strictly before the selected current recording",
        },
        "model_provenance": {
            "evidence_id": f"sleep-model:{record['model_id']}", "model_id": int(record["model_id"]),
            "model_name": record["model_name"], "model_version": record["model_version"],
            "fold_id": model_provenance["fold_id"], "test_patient": model_provenance["test_patient"],
            "validation_patient": model_provenance["validation_patient"],
            "normalization_patients": model_provenance["normalization"]["normalization_patients"],
            "thresholds": model_provenance["thresholds"],
        },
    }
    package["package_hash"] = canonical_json_hash(package)
    return package


def _request(package: dict) -> dict:
    _, policy = load_frozen_generation_contract()
    prompt = PROMPT_PATH.read_text(encoding="utf-8")
    settings = policy["generation_settings"]
    return {
        "model": policy["model"],
        "messages": [
            {"role": "system", "content": prompt},
            {"role": "user", "content": json.dumps(package, sort_keys=True, separators=(",", ":"), default=_jsonable)},
        ],
        "temperature": settings["temperature"], "top_p": settings["top_p"],
        "max_tokens": settings["max_tokens"],
        "extra_body": {"chat_template_kwargs": settings["chat_template_kwargs"]},
    }


def _execute(client, request: dict) -> dict:
    _, policy = load_frozen_generation_contract()
    attempts = []
    transport = policy["transport_retry"]
    statuses = set(transport["retryable_http_statuses"])
    for response_number in range(1, policy["malformed_json_retry"]["maximum_additional_requests"] + 2):
        for transport_attempt in range(1, transport["maximum_attempts"] + 1):
            try:
                response = client.chat.completions.create(**request)
                break
            except Exception as exc:
                attempts.append({"response_number": response_number, "transport_attempt": transport_attempt,
                                 "transport_error": f"{type(exc).__name__}: {exc}", "raw_response": None})
                if transport_attempt >= transport["maximum_attempts"] or not _retryable_transport_error(exc, statuses):
                    raise
                time.sleep(transport["backoff_seconds"][transport_attempt - 1])
        raw = response.choices[0].message.content
        attempt = {"response_number": response_number, "transport_attempt": transport_attempt,
                   "transport_error": None, "raw_response": raw,
                   "response_metadata": {"id": getattr(response, "id", None), "model": getattr(response, "model", None),
                                         "usage": response.usage.model_dump() if getattr(response, "usage", None) else None}}
        attempts.append(attempt)
        try:
            return {"attempts": attempts, "parsed": parse_structured_response(raw), "parse_failure": False}
        except ValueError as exc:
            attempt["parse_error"] = str(exc)
    return {"attempts": attempts, "parsed": None, "parse_failure": True}


def _persist(package: dict, parsed: dict) -> int:
    event = package["current_event"]
    xai = package["xai_provenance"]
    stage = package["recorded_sleep_stage_context"]
    marker = f"{MODEL_ID}|sleep_case_v1|pdm_enabled"
    text = json.dumps(parsed, sort_keys=True, separators=(",", ":"), ensure_ascii=False)
    expected = {
        (event["prediction_id"], None, None, "current_prediction"),
        (None, None, xai["explanation_id"], "current_xai"),
        (None, stage["annotation_id"], None, "sleep_stage_context"),
    }
    with connect() as conn, conn.cursor() as cur:
        cur.execute("SELECT narrative_id,narrative_text FROM narrative WHERE prediction_id=%s AND llm_model_version=%s",
                    (event["prediction_id"], marker))
        row = cur.fetchone()
        if row:
            narrative_id = int(row[0])
            if row[1] != text:
                raise ValueError("existing Sleep narrative differs")
        else:
            cur.execute("INSERT INTO narrative(patient_id,window_id,prediction_id,narrative_text,llm_model_version) VALUES(%s,%s,%s,%s,%s) RETURNING narrative_id",
                        (event["patient_id"], event["window_id"], event["prediction_id"], text, marker))
            narrative_id = int(cur.fetchone()[0])
            cur.execute("INSERT INTO narrative_evidence(narrative_id,prediction_id,evidence_role) VALUES(%s,%s,'current_prediction')", (narrative_id,event["prediction_id"]))
            cur.execute("INSERT INTO narrative_evidence(narrative_id,explanation_id,evidence_role) VALUES(%s,%s,'current_xai')", (narrative_id,xai["explanation_id"]))
            cur.execute("INSERT INTO narrative_evidence(narrative_id,annotation_id,evidence_role) VALUES(%s,%s,'sleep_stage_context')", (narrative_id,stage["annotation_id"]))
            conn.commit()
        cur.execute("SELECT prediction_id,annotation_id,explanation_id,evidence_role FROM narrative_evidence WHERE narrative_id=%s", (narrative_id,))
        if set(cur.fetchall()) != expected:
            raise ValueError("Sleep narrative links differ from supplied evidence")
    return narrative_id


def main() -> int:
    OUT.mkdir(parents=True, exist_ok=True)
    prediction_id, candidates = select_case()
    package = build_package(prediction_id)
    _write_json(OUT / "selection.json", {
        "schema_version": "sleep-case-selection-v1", "selected_prediction_id": prediction_id,
        "criteria_order": ["previous same-patient night", "at least two flagged channels", "all auxiliary channels", "largest detector normalized margin", "prediction ID"],
        "eligible_candidate_count": sum(int(row["prior_nights"]) >= 1 and int(row["flagged_channels"]) >= 2 and int(row["auxiliary_channels"]) == 3 for row in candidates),
        "candidate_audit": candidates,
    })
    _write_json(OUT / "evidence_package.json", package)
    request = _request(package)
    _write_json(OUT / "request.json", request)
    parsed_path = OUT / "parsed_response.json"
    if parsed_path.exists():
        parsed = json.loads(parsed_path.read_text(encoding="utf-8"))
    else:
        load_dotenv(ROOT / ".env")
        key = os.getenv("FEATHERLESS_API_KEY")
        if not key:
            raise RuntimeError("FEATHERLESS_API_KEY is not set")
        verify_model(key)
        from openai import OpenAI
        client = OpenAI(base_url=BASE_URL, api_key=key, default_headers={"X-Title": "JAISE2026 Sleep fusion case"})
        audit = _execute(client, request)
        _write_json(OUT / "response_audit.json", audit)
        for index, attempt in enumerate(audit["attempts"], start=1):
            if attempt.get("raw_response") is not None:
                (OUT / f"raw_response_{index}.txt").write_text(attempt["raw_response"], encoding="utf-8")
        if audit["parse_failure"]:
            raise RuntimeError("Sleep response requires manual parsing; automatic regeneration is forbidden")
        parsed = audit["parsed"]
        _write_json(parsed_path, parsed)
    validation = validate_response_evidence(parsed, package)
    _write_json(OUT / "response_validation.json", validation)
    narrative_id = _persist(package, parsed)
    prompt_hash = hashlib.sha256(PROMPT_PATH.read_bytes()).hexdigest()
    _write_json(OUT / "manifest.json", {
        "schema_version": "sleep-case-manifest-v1", "status": "complete",
        "prediction_id": prediction_id, "narrative_id": narrative_id,
        "package_hash": package["package_hash"], "prompt_sha256": prompt_hash,
        "model": MODEL_ID, "response_validation": validation,
    })
    sections = parsed["narrative_sections"]
    markdown = f"""# Representative Sleep-EDF fusion case

**Prediction:** {prediction_id}  
**PDM narrative:** {narrative_id}  
**Package hash:** `{package['package_hash']}`  
**Prompt hash:** `{prompt_hash}`

## Current event

{sections['current_event']}

## Model and XAI evidence

{sections['model_and_xai_evidence']}

## Previous-night history

{sections['previous_monitoring_history']}

## Recorded stage and auxiliary context

{sections['recorded_clinical_context']}

## Evidence-grounded summary

{sections['evidence_grounded_summary']}

## Interpretation limits

This is one patient-held-out prototype case. Channel flags are components of one shared model, attention has no channel owner, and the supplied Sleep-EDF stage is context rather than a diagnosis or proof of detector correctness.
"""
    (OUT / "SLEEP_CASE.md").write_text(markdown, encoding="utf-8")
    print(json.dumps({"status": "complete", "prediction_id": prediction_id, "narrative_id": narrative_id, "output": str(OUT)}, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
