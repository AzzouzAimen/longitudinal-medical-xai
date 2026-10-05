"""Evidence viewer and human review API for the Patient Digital Memory prototype."""
import os
from datetime import date, datetime
from decimal import Decimal
from pathlib import Path

from dotenv import load_dotenv
from flask import Flask, jsonify, render_template, request
from psycopg2.errors import UndefinedTable

ROOT = Path(__file__).resolve().parents[1]
load_dotenv(ROOT / ".env")

import sys
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))
from pdm.db import connect
from viewer.evidence import build_evidence as assemble_evidence
from viewer.feedback import EventNotFound, list_feedback, save_feedback, validate_feedback

app = Flask(__name__, template_folder="templates", static_folder="static")
app.config["MAX_CONTENT_LENGTH"] = 65536


def jsonable(value):
    if isinstance(value, (datetime, date)):
        return value.isoformat()
    if isinstance(value, Decimal):
        return float(value)
    return value


def rows(cur):
    columns = [d[0] for d in cur.description]
    return [{key: jsonable(value) for key, value in zip(columns, row)} for row in cur.fetchall()]


def query(sql, params=()):
    with connect() as conn, conn.cursor() as cur:
        cur.execute(sql, params)
        return rows(cur)


def error_response(exc):
    app.logger.warning("Viewer database request failed: %s", exc)
    return jsonify({"error": "Database unavailable or viewer query failed. Check the local PDM service."}), 503


def build_evidence(prediction_id):
    return assemble_evidence(prediction_id, query)


@app.get("/api/event/<int:prediction_id>/evidence")
def event_evidence(prediction_id):
    try:
        data = build_evidence(prediction_id)
        return jsonify(data) if data else (jsonify({"error": "Event not found."}), 404)
    except Exception as exc:
        return error_response(exc)


@app.route("/api/event/<int:prediction_id>/feedback", methods=["GET", "POST"])
def feedback(prediction_id):
    if request.method == "POST":
        origin = request.headers.get("Origin")
        if (origin and origin.rstrip("/") != request.host_url.rstrip("/")) or request.headers.get("Sec-Fetch-Site") == "cross-site":
            return jsonify({"error": "Feedback must be submitted from this viewer."}), 403
        if not request.is_json:
            return jsonify({"error": "Expected JSON feedback."}), 400
    try:
        if request.method == "GET":
            return jsonify(list_feedback(prediction_id))
        data = validate_feedback(request.get_json(silent=True))
        return jsonify(save_feedback(prediction_id, data)), 201
    except EventNotFound as exc:
        return jsonify({"error": str(exc)}), 404
    except ValueError as exc:
        return jsonify({"error": str(exc)}), 400
    except UndefinedTable:
        return jsonify({"error": "Feedback storage is not initialized. Run python -m viewer.feedback."}), 503
    except Exception as exc:
        return error_response(exc)


@app.get("/")
def index():
    return render_template("index.html")


@app.get("/api/patients")
def patients():
    try:
        return jsonify(query("""
            SELECT p.patient_code, COALESCE(string_agg(DISTINCT r.modality, ', '), '') AS modality,
                   count(DISTINCT pr.prediction_id) AS event_count
            FROM patient p
            LEFT JOIN recording r ON r.patient_id = p.patient_id
            LEFT JOIN analysis_window w ON w.recording_id = r.recording_id
            LEFT JOIN prediction pr ON pr.window_id = w.window_id
            GROUP BY p.patient_id, p.patient_code
            ORDER BY CASE WHEN p.patient_code = 'APNEA_a03' THEN 0 ELSE 1 END, p.patient_code
        """))
    except Exception as exc:
        return error_response(exc)


@app.get("/api/patient/<path:patient_code>/events")
def patient_events(patient_code):
    try:
        return jsonify(query("""
            SELECT pr.prediction_id, r.recording_id, r.modality, w.window_index,
                   w.start_time, w.end_time, w.duration_seconds, pr.is_anomaly,
                   pr.predicted_class, a.label AS official_label,
                   EXISTS (SELECT 1 FROM narrative n WHERE n.prediction_id=pr.prediction_id) AS has_narrative
            FROM prediction pr
            JOIN analysis_window w USING(window_id)
            JOIN recording r USING(recording_id)
            JOIN patient p USING(patient_id)
            LEFT JOIN LATERAL (SELECT label FROM annotation
                WHERE window_id=w.window_id AND source IN ('official','official_evaluation_only')
                ORDER BY annotation_id LIMIT 1) a ON true
            WHERE p.patient_code=%s
            ORDER BY w.start_time, pr.prediction_id
        """, (patient_code,)))
    except Exception as exc:
        return error_response(exc)


@app.get("/api/event/<int:prediction_id>")
def event(prediction_id):
    try:
        case = query("""
            SELECT pr.prediction_id, pr.window_id, p.patient_code, r.recording_id,
                   r.modality, r.source_recording_name, w.window_index, w.start_time,
                   w.end_time, w.duration_seconds, pr.is_anomaly, pr.predicted_class,
                   pr.reconstruction_error, pr.anomaly_threshold, m.model_name,
                   m.model_version, x.explanation_id, a.annotation_id,
                   a.label AS official_label, a.source AS annotation_source
            FROM prediction pr
            JOIN analysis_window w USING(window_id)
            JOIN recording r USING(recording_id)
            JOIN patient p USING(patient_id)
            JOIN model m USING(model_id)
            LEFT JOIN xai_explanation x ON x.prediction_id=pr.prediction_id
            LEFT JOIN annotation a ON a.window_id=w.window_id AND a.source='official'
            WHERE pr.prediction_id=%s
            ORDER BY a.annotation_id NULLS LAST LIMIT 1
        """, (prediction_id,))
        if not case:
            return jsonify({"error": "Event not found"}), 404
        detail = case[0]
        context = query("""
            SELECT visit_id, visit_start, visit_type, reason, clinical_notes, status
            FROM clinical_visit WHERE patient_id=(SELECT patient_id FROM patient WHERE patient_code=%s)
            ORDER BY visit_start
        """, (detail["patient_code"],))
        diagnoses = query("""
            SELECT diagnosis_id, diagnosis_code, diagnosis_name, diagnosis_status, diagnosis_time, notes
            FROM diagnosis WHERE patient_id=(SELECT patient_id FROM patient WHERE patient_code=%s)
            ORDER BY diagnosis_time
        """, (detail["patient_code"],))
        history = query("""
            SELECT * FROM (
              SELECT pr.prediction_id, w.window_index, w.start_time, w.end_time,
                     pr.is_anomaly, a.label AS official_label
              FROM prediction pr JOIN analysis_window w USING(window_id)
              LEFT JOIN annotation a ON a.window_id=w.window_id AND a.source='official'
              WHERE w.recording_id=(SELECT recording_id FROM analysis_window WHERE window_id=%s)
                AND w.start_time <= %s
              ORDER BY w.start_time DESC LIMIT 6
            ) recent ORDER BY start_time
        """, (detail["window_id"], detail["start_time"]))
        narrative = query("""
            SELECT narrative_id, narrative_text, llm_model_version, generated_at
            FROM narrative WHERE prediction_id=%s ORDER BY generated_at DESC LIMIT 1
        """, (prediction_id,))
        evidence = query("""
            SELECT narrative_evidence_id, explanation_id, annotation_id, prediction_id, evidence_role
            FROM narrative_evidence WHERE narrative_id=(SELECT narrative_id FROM narrative
              WHERE prediction_id=%s ORDER BY generated_at DESC LIMIT 1)
            ORDER BY narrative_evidence_id
        """, (prediction_id,))
        detail.update({"clinical_context": {"source": "synthetic_prototype", "visits": context, "diagnoses": diagnoses},
                       "history": history, "narrative": narrative[0] if narrative else None,
                       "narrative_evidence": evidence})
        return jsonify(detail)
    except Exception as exc:
        return error_response(exc)


@app.get("/api/event/<int:prediction_id>/llm-input")
def llm_input(prediction_id):
    """Return only the PDM evidence package sent to the LLM."""
    try:
        # Keep this lazy: the viewer does not need the LLM client or API key.
        from pdm.llm import build_evidence_package
        evidence = build_evidence_package(prediction_id)
        return jsonify(evidence)
    except Exception as exc:
        return error_response(exc)


if __name__ == "__main__":
    app.run(host=os.getenv("VIEWER_HOST", "127.0.0.1"),
            port=int(os.getenv("VIEWER_PORT", "5000")), debug=False)
