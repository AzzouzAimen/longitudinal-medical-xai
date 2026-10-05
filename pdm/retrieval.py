"""Bounded read-only retrieval for canonical evidence packages."""

from .db import connect


TREND_COLUMNS = (
    "bucket",
    "patient_id",
    "model_id",
    "mean_reconstruction_error",
    "max_reconstruction_error",
    "mean_anomaly_margin",
    "total_count",
    "anomaly_count",
    "anomaly_rate",
)


def _bounded_limit(value: int, maximum: int = 100) -> int:
    limit = int(value)
    if limit < 1 or limit > maximum:
        raise ValueError(f"limit must be between 1 and {maximum}")
    return limit


def get_detailed_patient_history(patient_id: int, before_time, limit: int = 5) -> list[dict]:
    """Return the latest earlier model events for exactly one patient."""

    if before_time is None:
        raise ValueError("before_time is required for historical retrieval")
    limit = _bounded_limit(limit)
    query = """
        SELECT ps.timestamp, ps.patient_id, p.patient_code, ps.recording_id,
               ps.window_id, ps.prediction_id, ps.model_id, w.window_index,
               w.start_time, w.end_time, ps.predicted_class,
               ps.reconstruction_error, ps.anomaly_threshold,
               ps.anomaly_margin, ps.normalized_margin, ps.is_anomaly,
               m.model_name, m.model_version,
               xe.explanation_id, xe.xai_method, xe.explanation_type,
               xe.explanation_text,
               COUNT(ad.*) AS attention_count,
               MIN(ad.attention_value) AS attention_min,
               MAX(ad.attention_value) AS attention_max,
               AVG(ad.attention_value) AS attention_mean
        FROM prediction_series AS ps
        JOIN patient AS p ON p.patient_id = ps.patient_id
        JOIN analysis_window AS w ON w.window_id = ps.window_id
        JOIN model AS m ON m.model_id = ps.model_id
        LEFT JOIN xai_explanation AS xe ON xe.prediction_id = ps.prediction_id
        LEFT JOIN attention_data AS ad ON ad.window_id = ps.window_id
        WHERE ps.patient_id = %s AND ps.timestamp < %s
        GROUP BY ps.timestamp, ps.patient_id, p.patient_code, ps.recording_id,
                 ps.window_id, ps.prediction_id, ps.model_id, w.window_index,
                 w.start_time, w.end_time, ps.predicted_class,
                 ps.reconstruction_error, ps.anomaly_threshold,
                 ps.anomaly_margin, ps.normalized_margin, ps.is_anomaly,
                 m.model_name, m.model_version, xe.explanation_id,
                 xe.xai_method, xe.explanation_type, xe.explanation_text
        ORDER BY ps.timestamp DESC, ps.prediction_id DESC
        LIMIT %s
    """
    columns = (
        "timestamp", "patient_id", "patient_code", "recording_id", "window_id",
        "prediction_id", "model_id", "window_index", "start_time", "end_time",
        "predicted_class", "reconstruction_error", "anomaly_threshold",
        "anomaly_margin", "normalized_margin", "is_anomaly", "model_name",
        "model_version", "explanation_id", "xai_method", "explanation_type",
        "explanation_text", "attention_count", "attention_min", "attention_max",
        "attention_mean",
    )
    with connect() as conn, conn.cursor() as cur:
        cur.execute(query, (int(patient_id), before_time, limit))
        rows = []
        for values in cur.fetchall():
            row = dict(zip(columns, values))
            row["evidence_id"] = f"history:prediction:{row['prediction_id']}"
            row["xai_summary"] = {
                "available": row.pop("explanation_id") is not None,
                "method": row.pop("xai_method"),
                "type": row.pop("explanation_type"),
                "description": row.pop("explanation_text"),
                "attention_sample_count": row.pop("attention_count"),
                "attention_min": row.pop("attention_min"),
                "attention_max": row.pop("attention_max"),
                "attention_mean": row.pop("attention_mean"),
            }
            rows.append(row)
        return rows


def get_prediction_trend(patient_id: int, before_time=None, limit: int = 12) -> list[dict]:
    """Return earlier same-patient trend buckets from the continuous aggregate.

    ``before_time`` is exclusive so the current event's bucket is not returned
    as historical context.  The query intentionally reads only
    ``pdm_prediction_5m``; it never recomputes the aggregate from raw
    ``prediction_series`` rows.
    """

    limit = _bounded_limit(limit)
    query = """
        SELECT bucket, patient_id, model_id, mean_reconstruction_error,
               max_reconstruction_error, mean_anomaly_margin, total_count,
               anomaly_count, anomaly_rate
        FROM (
            SELECT bucket, patient_id, model_id, mean_reconstruction_error,
                   max_reconstruction_error, mean_anomaly_margin, total_count,
                   anomaly_count, anomaly_rate
            FROM pdm_prediction_5m
            WHERE patient_id = %s
    """
    params = [patient_id]
    if before_time is not None:
        query += " AND bucket < %s\n"
        params.append(before_time)
    query += " ORDER BY bucket DESC, model_id DESC LIMIT %s\n"
    params.append(limit)
    query += ") AS latest ORDER BY bucket ASC, model_id ASC"

    with connect() as conn, conn.cursor() as cur:
        cur.execute(query, tuple(params))
        rows = [dict(zip(TREND_COLUMNS, row)) for row in cur.fetchall()]
    for row in rows:
        bucket = row["bucket"].isoformat()
        row["evidence_id"] = (
            f"aggregate:patient:{row['patient_id']}:model:{row['model_id']}:bucket:{bucket}"
        )
    return rows


def get_patient_recurrence_summary(patient_id: int, before_time) -> dict:
    """Summarize all prior lightweight detector history for one patient."""

    if before_time is None:
        raise ValueError("before_time is required for recurrence retrieval")
    query = """
        SELECT COUNT(*), COUNT(*) FILTER (WHERE is_anomaly),
               AVG(reconstruction_error), MAX(reconstruction_error),
               AVG(anomaly_margin), MIN(timestamp), MAX(timestamp)
        FROM prediction_series
        WHERE patient_id = %s AND timestamp < %s
    """
    with connect() as conn, conn.cursor() as cur:
        cur.execute(query, (int(patient_id), before_time))
        count, anomalies, mean_error, max_error, mean_margin, first_time, last_time = cur.fetchone()
    count = int(count)
    anomalies = int(anomalies)
    return {
        "evidence_id": f"recurrence:patient:{int(patient_id)}:before:{before_time.isoformat()}",
        "patient_id": int(patient_id),
        "before_time": before_time,
        "analyzed_window_count": count,
        "anomaly_count": anomalies,
        "anomaly_rate": (anomalies / count) if count else None,
        "mean_reconstruction_error": mean_error,
        "max_reconstruction_error": max_error,
        "mean_anomaly_margin": mean_margin,
        "first_event_time": first_time,
        "latest_event_time": last_time,
    }


def get_clinical_context(patient_id: int, before_time) -> dict:
    """Return earlier synthetic visits and diagnoses, never narratives."""

    if before_time is None:
        raise ValueError("before_time is required for clinical-context retrieval")
    with connect() as conn, conn.cursor() as cur:
        cur.execute(
            """
            SELECT visit_id, visit_start, visit_end, visit_type, reason,
                   clinical_notes, status
            FROM clinical_visit
            WHERE patient_id = %s AND visit_start < %s
            ORDER BY visit_start ASC, visit_id ASC
            """,
            (int(patient_id), before_time),
        )
        visit_columns = (
            "visit_id", "visit_start", "visit_end", "visit_type", "reason",
            "notes", "status",
        )
        visits = [dict(zip(visit_columns, row)) for row in cur.fetchall()]
        for visit in visits:
            visit["evidence_id"] = f"clinical:visit:{visit['visit_id']}"
        cur.execute(
            """
            SELECT diagnosis_id, visit_id, diagnosis_code, diagnosis_name,
                   diagnosis_status, diagnosis_time, notes
            FROM diagnosis
            WHERE patient_id = %s
              AND diagnosis_time < %s
            ORDER BY diagnosis_time ASC, diagnosis_id ASC
            """,
            (int(patient_id), before_time),
        )
        diagnosis_columns = (
            "diagnosis_id", "visit_id", "code", "name", "status", "time", "notes",
        )
        diagnoses = [dict(zip(diagnosis_columns, row)) for row in cur.fetchall()]
        for diagnosis in diagnoses:
            diagnosis["evidence_id"] = f"clinical:diagnosis:{diagnosis['diagnosis_id']}"
    return {
        "source": "synthetic_prototype",
        "patient_id": int(patient_id),
        "visits": visits,
        "diagnoses": diagnoses,
    }
