"""Read-only adapter for the supervisor's evidence viewer. No model execution."""
import json
import math
from viewer.provenance import generation_provenance


def claim_source(ids):
    history = any("history" in item or "sleep-night:" in item or item.startswith(("aggregate:", "recurrence:", "clinical:", "visit:", "diagnosis:")) for item in ids)
    current = any(item.startswith(("prediction:", "window:", "model:", "xai:", "sleep-prediction:", "sleep-window:", "sleep-channel:", "sleep-model:", "sleep-stage:", "sleep-annotation:", "sleep-aux:")) for item in ids)
    return "both" if history and current else "history" if history else "current" if current else "unknown"


def parse_narrative(record):
    if not record:
        return None
    result = dict(record)
    text = result.pop("narrative_text", "") or ""
    result.update(text=text, claims=[])
    try:
        parsed = json.loads(text)
    except (ValueError, TypeError):
        return result
    if isinstance(parsed, dict) and isinstance(parsed.get("claims"), list):
        for index, item in enumerate(parsed["claims"]):
            if not isinstance(item, dict) or not isinstance(item.get("text"), str):
                continue
            ids = item.get("evidence_ids", [])
            ids = [value for value in ids if isinstance(value, str)] if isinstance(ids, list) else []
            result["claims"].append({"claim_id": item.get("claim_id", f"claim-{index + 1}"),
                "text": item["text"], "section": item.get("section", ""),
                "evidence_ids": ids, "source": claim_source(ids)})
    return result


def number(value):
    return float(value) if value is not None and math.isfinite(float(value)) else None


def read_points(query, signal_id, event):
    # Retain the min and max in each time bucket, rather than losing short peaks
    # through uniform sampling. A trace contains at most 1,200 actual points.
    return query("""
        WITH bucketed AS (
          SELECT timestamp, value, floor(extract(epoch FROM (timestamp - %s)) /
            GREATEST(%s, 0.001) * 600)::int AS bucket
          FROM signal_data WHERE signal_id=%s AND timestamp >= %s AND timestamp < %s
        ), extremes AS (
          SELECT bucket, (array_agg(timestamp ORDER BY value, timestamp))[1] AS min_time,
            min(value) AS min_value,
            (array_agg(timestamp ORDER BY value DESC, timestamp))[1] AS max_time,
            max(value) AS max_value FROM bucketed GROUP BY bucket
        )
        SELECT DISTINCT point.timestamp, point.value FROM extremes
        CROSS JOIN LATERAL (VALUES (min_time, min_value), (max_time, max_value)) AS point(timestamp, value)
        ORDER BY point.timestamp
    """, (event["start_time"], event["duration_seconds"] or 0, signal_id, event["start_time"], event["end_time"]))


def build_evidence(prediction_id, query):
    records = query("""
        SELECT pr.prediction_id, pr.window_id, p.patient_id, p.patient_code,
          r.recording_id, r.source_recording_name, r.modality, r.recording_start_time,
          w.window_index, w.start_time, w.end_time, w.duration_seconds,
          pr.is_anomaly, pr.predicted_class, pr.reconstruction_error, pr.anomaly_threshold,
          m.model_id, m.model_name, m.model_version,
          a.annotation_id, a.label AS official_label, a.source AS annotation_source
        FROM prediction pr JOIN analysis_window w USING(window_id)
        JOIN recording r USING(recording_id) JOIN patient p USING(patient_id)
        JOIN model m USING(model_id)
        LEFT JOIN LATERAL (SELECT annotation_id, label, source FROM annotation
          WHERE window_id=w.window_id AND source IN ('official', 'official_evaluation_only')
          ORDER BY annotation_id LIMIT 1) a ON true
        WHERE pr.prediction_id=%s
    """, (prediction_id,))
    if not records:
        return None
    event = records[0]
    streams = query("""SELECT s.signal_id, s.channel, s.unit, s.signal_type, s.sampling_rate_hz
        FROM signal s JOIN window_signal ws USING(signal_id)
        WHERE ws.window_id=%s AND s.signal_type IN ('raw', 'reconstruction', 'residual')
        ORDER BY s.signal_id""", (event["window_id"],))
    features = query("""SELECT xf.signal_id, s.channel, xf.feature_name, xf.feature_value
        FROM xai_feature xf JOIN xai_explanation xe USING(explanation_id)
        LEFT JOIN signal s USING(signal_id) WHERE xe.prediction_id=%s
        ORDER BY xe.generated_at, xe.explanation_id, xf.xai_feature_id""", (prediction_id,))
    by_channel = {}
    for row in features:
        if row["channel"]:
            by_channel.setdefault(row["channel"], {})[row["feature_name"]] = row["feature_value"]
    locations = query("""SELECT xl.channel, xl.temporal_start, xl.temporal_end
        FROM xai_location xl JOIN xai_explanation xe USING(explanation_id)
        WHERE xe.prediction_id=%s ORDER BY xl.location_id""", (prediction_id,))
    channels = []
    auxiliary = []
    for raw in [row for row in streams if row["signal_type"] == "raw"]:
        name = raw["channel"] or f"Signal {raw['signal_id']}"
        feature = by_channel.get(name, {})
        item = dict(raw, name=name, contribution=None, flagged=None, features=feature,
                    signal=read_points(query, raw["signal_id"], event), reconstruction=[], residual=[],
                    regions=[row for row in locations if row["channel"] == name])
        if feature.get("contribution_share") is not None:
            share = number(feature["contribution_share"])
            item["contribution"] = share * 100 if share is not None else None
        if feature.get("residual_threshold_flag") is not None:
            item["flagged"] = bool(feature["residual_threshold_flag"])
        matching = [row for row in streams if row["channel"] == raw["channel"] and row["signal_type"] != "raw"]
        for row in matching:
            item[row["signal_type"]] = read_points(query, row["signal_id"], event)
            item[row["signal_type"] + "_unit"] = row["unit"]
        # Auxiliary streams have no model residual metadata or reconstruction.
        if not feature and not matching and event["modality"] == "sleep" and name in {"Resp oro-nasal", "EMG submental", "Temp rectal"}:
            auxiliary.append(item)
        else:
            channels.append(item)
    # Channel evidence can survive even when dense arrays have not been retained.
    for name, feature in by_channel.items():
        if not any(row["name"] == name for row in channels):
            share = number(feature.get("contribution_share"))
            flag = feature.get("residual_threshold_flag")
            channels.append(dict(name=name, signal_id=None, unit=None, sampling_rate_hz=None,
                contribution=share * 100 if share is not None else None, flagged=bool(flag) if flag is not None else None,
                features=feature, signal=[], reconstruction=[], residual=[], regions=[]))
    attention = query("""SELECT timestamp, attention_value AS value FROM attention_data
        WHERE window_id=%s AND timestamp >= %s AND timestamp < %s
        ORDER BY timestamp LIMIT 5000""", (event["window_id"], event["start_time"], event["end_time"]))
    peak = max(attention, key=lambda row: row["value"]) if attention else None
    previous = query("""
        SELECT r.recording_id, r.source_recording_name, r.recording_start_time,
          count(*) AS windows, count(*) FILTER (WHERE pr.is_anomaly) AS anomalies,
          100.0 * count(*) FILTER (WHERE pr.is_anomaly) / NULLIF(count(*),0) AS rate
        FROM recording r JOIN analysis_window w USING(recording_id) JOIN prediction pr USING(window_id)
        WHERE r.patient_id=%s AND r.modality=%s AND r.recording_id <> %s
          AND COALESCE(r.recording_start_time,w.start_time) < %s AND w.end_time <= %s
        GROUP BY r.recording_id ORDER BY r.recording_start_time DESC NULLS LAST LIMIT 1
    """, (event["patient_id"], event["modality"], event["recording_id"],
           event["recording_start_time"] or event["start_time"], event["start_time"]))
    recent = query("""SELECT pr.prediction_id, w.start_time, w.window_index, r.recording_id, pr.is_anomaly
        FROM prediction pr JOIN analysis_window w USING(window_id) JOIN recording r USING(recording_id)
        WHERE r.patient_id=%s AND w.start_time < %s ORDER BY w.start_time DESC, pr.prediction_id DESC LIMIT 5""",
        (event["patient_id"], event["start_time"]))
    narratives = query("""SELECT narrative_id, narrative_text, llm_model_version, generated_at
        FROM narrative WHERE prediction_id=%s ORDER BY generated_at DESC, narrative_id DESC LIMIT 1""", (prediction_id,))
    narrative = parse_narrative(narratives[0] if narratives else None)
    generation = generation_provenance(narratives[0] if narratives else None, prediction_id)
    evidence = query("""SELECT narrative_evidence_id, explanation_id, annotation_id, prediction_id,
        source_narrative_id, evidence_role FROM narrative_evidence WHERE narrative_id=%s ORDER BY narrative_evidence_id""",
        (narrative["narrative_id"],)) if narrative else []
    return {"event": event, "channels": channels, "auxiliary": auxiliary,
        "attention": {"points": attention, "peak_time": peak["timestamp"] if peak else None,
                      "semantics": "Shared window-level temporal relevance; not channel-specific or a probability."},
        "history": {"previous_recording": previous[0] if previous else None, "recent_events": recent},
        "narrative": narrative, "provenance": {"source": "pdm_database", **generation,
            "narrative_evidence": evidence}}
