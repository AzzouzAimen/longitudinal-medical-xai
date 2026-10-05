"""Persistent review notes, separate from inference and generation."""
from pathlib import Path

from pdm.db import connect


class EventNotFound(ValueError):
    pass


def validate_feedback(payload):
    if not isinstance(payload, dict):
        raise ValueError("Feedback must be an object.")
    notes = payload.get("notes")
    if not isinstance(notes, str) or not 1 <= len(notes.strip()) <= 10000 or "\x00" in notes:
        raise ValueError("Enter a note between 1 and 10,000 characters.")
    narrative_id = payload.get("narrative_id")
    if narrative_id is not None and (type(narrative_id) is not int or narrative_id <= 0):
        raise ValueError("Invalid narrative identifier.")
    subject = payload.get("subject", "overall")
    if not isinstance(subject, str) or subject not in {"overall", "narrative", "channel"}:
        raise ValueError("Invalid feedback subject.")
    result = dict(notes=notes.strip(), narrative_id=narrative_id, subject=subject)
    for key in ("channel", "reviewer"):
        value = payload.get(key)
        if value is not None and (not isinstance(value, str) or len(value) > 128 or "\x00" in value):
            raise ValueError(f"Invalid {key}.")
        result[key] = value.strip() or None if value else None
    if subject == "channel" and not result["channel"]:
        raise ValueError("Select a channel for channel feedback.")
    if subject == "narrative" and narrative_id is None:
        raise ValueError("No narrative is available for this event.")
    return result


def _record(cur):
    row = cur.fetchone()
    if row is None:
        return None
    return dict(zip((column[0] for column in cur.description), row))


def list_feedback(prediction_id):
    with connect() as conn, conn.cursor() as cur:
        cur.execute("SELECT 1 FROM prediction WHERE prediction_id=%s", (prediction_id,))
        if not cur.fetchone():
            raise EventNotFound("Event not found.")
        cur.execute("""SELECT feedback_id, prediction_id, narrative_id, subject, channel, reviewer, notes, created_at
            FROM viewer_feedback WHERE prediction_id=%s ORDER BY created_at DESC, feedback_id DESC""", (prediction_id,))
        columns = [column[0] for column in cur.description]
        return [dict(zip(columns, row)) for row in cur.fetchall()]


def save_feedback(prediction_id, payload):
    data = validate_feedback(payload)
    with connect() as conn, conn.cursor() as cur:
        cur.execute("SELECT 1 FROM prediction WHERE prediction_id=%s", (prediction_id,))
        if not cur.fetchone():
            raise EventNotFound("Event not found.")
        if data["narrative_id"] is not None:
            cur.execute("SELECT 1 FROM narrative WHERE narrative_id=%s AND prediction_id=%s", (data["narrative_id"], prediction_id))
            if not cur.fetchone():
                raise ValueError("Narrative does not belong to this event.")
        if data["channel"]:
            cur.execute("""SELECT 1 FROM window_signal ws JOIN signal s USING(signal_id)
                JOIN prediction pr USING(window_id) WHERE pr.prediction_id=%s AND s.channel=%s LIMIT 1""",
                (prediction_id, data["channel"]))
            if not cur.fetchone():
                raise ValueError("Channel does not belong to this event.")
        cur.execute("""INSERT INTO viewer_feedback(prediction_id,narrative_id,subject,channel,reviewer,notes)
            VALUES(%s,%s,%s,%s,%s,%s)
            RETURNING feedback_id,prediction_id,narrative_id,subject,channel,reviewer,notes,created_at""",
            (prediction_id, data["narrative_id"], data["subject"], data["channel"], data["reviewer"], data["notes"]))
        result = _record(cur)
        conn.commit()
        return result


def initialize():
    migration = Path(__file__).resolve().parents[1] / "schemas/migrations/002_viewer_feedback.sql"
    with connect() as conn, conn.cursor() as cur:
        cur.execute(migration.read_text(encoding="utf-8"))
        conn.commit()


if __name__ == "__main__":
    from dotenv import load_dotenv
    load_dotenv(Path(__file__).resolve().parents[1] / ".env")
    initialize()
    print("Viewer feedback storage is ready.")
