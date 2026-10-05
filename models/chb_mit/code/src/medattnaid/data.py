import os
from dataclasses import dataclass
import numpy as np
import psycopg2


@dataclass
class WindowBatch:
    windows: np.ndarray
    labels: np.ndarray
    subject_ids: np.ndarray
    recording_ids: np.ndarray
    channels: np.ndarray
    start_times: np.ndarray


def timescale_connection():
    return psycopg2.connect(
        host=os.getenv("TIMESCALE_HOST", "localhost"),
        port=int(os.getenv("TIMESCALE_PORT", "5432")),
        dbname=os.getenv("TIMESCALE_DB", "eeg_timescale"),
        user=os.getenv("TIMESCALE_USER", "postgres"),
        password=os.getenv("TIMESCALE_PASSWORD", ""),
    )


def postgres_connection():
    return psycopg2.connect(
        host=os.getenv("POSTGRES_HOST", os.getenv("PGHOST", "localhost")),
        port=int(os.getenv("POSTGRES_PORT", os.getenv("PGPORT", "5432"))),
        dbname=os.getenv("POSTGRES_DB", os.getenv("PGDATABASE", "eeg_db")),
        user=os.getenv("POSTGRES_USER", os.getenv("PGUSER", "postgres")),
        password=os.getenv("POSTGRES_PASSWORD", os.getenv("PGPASSWORD", "")),
    )


DEFAULT_CHANNELS = ("FP1-F7", "F7-T7", "T7-P7", "P7-O1", "FP2-F8", "F8-T8", "T8-P8", "P8-O2")


def fetch_windows(subject_id="chb01", channel="ALL", window_length=100, stride=100, limit=20000):
    backend = os.getenv("EEG_DB_BACKEND", "timescale").lower()
    if backend in {"postgres", "postgresql", "pg"}:
        return fetch_windows_postgres(subject_id, channel, window_length, stride, limit)
    query = """
        SELECT recording_id, channel, time, amplitude_uv, is_seizure
        FROM eeg_signals
        WHERE subject_id = %s AND channel = %s
        ORDER BY recording_id, time
        LIMIT %s
    """
    with timescale_connection() as conn:
        with conn.cursor() as cur:
            cur.execute(query, (subject_id, channel, limit))
            rows = cur.fetchall()
    values = np.array([r[3] for r in rows], dtype=np.float32)
    labels = np.array([r[4] for r in rows], dtype=np.int64)
    recordings = np.array([r[0] for r in rows])
    times = np.array([r[2] for r in rows])
    if values.size == 0:
        raise RuntimeError("No samples were returned from TimescaleDB.")
    low, high = np.percentile(values, [1, 99])
    values = np.clip(values, low, high)
    values = (values - values.min()) / max(values.max() - values.min(), 1e-8)
    windows, y, recs, starts = [], [], [], []
    for start in range(0, len(values) - window_length + 1, stride):
        end = start + window_length
        windows.append(values[start:end, None])
        y.append(int(labels[start:end].mean() >= 0.5))
        recs.append(recordings[start])
        starts.append(times[start])
    return WindowBatch(
        windows=np.asarray(windows, dtype=np.float32),
        labels=np.asarray(y, dtype=np.int64),
        subject_ids=np.asarray([subject_id] * len(windows)),
        recording_ids=np.asarray(recs),
        channels=np.asarray([channel] * len(windows)),
        start_times=np.asarray(starts),
    )


def fetch_windows_postgres(subject_id="chb01", channel="ALL", window_length=100, stride=100, limit=20000):
    selected_channels = DEFAULT_CHANNELS if channel == "ALL" else (channel,)
    recording_query = """
        SELECT r.recording_id, r.filename, c.channel_name, c.channel_id
        FROM recordings r
        JOIN channels c ON c.recording_id = r.recording_id
        WHERE r.subject_id = %s AND c.channel_name = ANY(%s)
        ORDER BY r.recording_id, c.channel_name
    """
    seizure_query = """
        SELECT recording_id, start_sec, end_sec
        FROM seizure_annotations
        ORDER BY recording_id, start_sec
    """
    sample_query = """
        SELECT ss.time_offset_ms, ss.amplitude_uv,
               EXISTS (
                   SELECT 1
                   FROM seizure_annotations sa
                   WHERE sa.recording_id = ss.recording_id
                     AND ss.time_offset_ms BETWEEN sa.start_sec * 1000 AND sa.end_sec * 1000
               ) AS is_seizure
        FROM signal_samples ss
        WHERE ss.recording_id = %s
          AND ss.channel_id = %s
          AND ss.time_offset_ms BETWEEN %s AND %s
        ORDER BY ss.sample_index
    """
    with postgres_connection() as conn:
        with conn.cursor() as cur:
            cur.execute(recording_query, (subject_id, list(selected_channels)))
            channel_rows = cur.fetchall()
            recordings_by_id = {}
            for recording_id, filename, channel_name, channel_id in channel_rows:
                entry = recordings_by_id.setdefault(recording_id, {"filename": filename, "channels": {}})
                entry["channels"][channel_name] = channel_id
            recordings = [
                (recording_id, entry["filename"], entry["channels"])
                for recording_id, entry in recordings_by_id.items()
                if all(name in entry["channels"] for name in selected_channels)
            ]
            cur.execute(seizure_query)
            seizure_rows = cur.fetchall()
            seizure_by_recording = {}
            for recording_id, start_sec, end_sec in seizure_rows:
                seizure_by_recording.setdefault(recording_id, []).append((start_sec, end_sec))
            segments = []
            healthy_recordings = [row for row in recordings if row[0] not in seizure_by_recording]
            for recording_id, filename, channel_ids in healthy_recordings[:3]:
                segments.append((recording_id, filename, channel_ids, 0, 180_000, "healthy"))
            for recording_id, filename, channel_ids in recordings:
                for start_sec, end_sec in seizure_by_recording.get(recording_id, []):
                    start_ms = max(0, (start_sec - 30) * 1000)
                    end_ms = (end_sec + 30) * 1000
                    segments.append((recording_id, filename, channel_ids, start_ms, end_ms, "anomalous"))
                    normal_start = max(0, (start_sec - 180) * 1000)
                    normal_end = max(0, (start_sec - 60) * 1000)
                    if normal_end > normal_start:
                        segments.append((recording_id, filename, channel_ids, normal_start, normal_end, "preictal_baseline"))
            values, labels, recs, starts = [], [], [], []
            per_segment_limit = max(window_length * 20, min(limit // max(len(segments), 1), 120_000))
            for recording_id, filename, channel_ids, start_ms, end_ms, _ in segments:
                channel_values = []
                segment_labels = None
                segment_times = None
                for channel_name in selected_channels:
                    cur.execute(sample_query, (recording_id, channel_ids[channel_name], start_ms, end_ms))
                    rows = cur.fetchmany(per_segment_limit)
                    vals = [float(row[1]) for row in rows]
                    labs = [int(row[2]) for row in rows]
                    times = [int(row[0]) for row in rows]
                    if not vals:
                        break
                    channel_values.append(vals)
                    if segment_labels is None:
                        segment_labels = labs
                        segment_times = times
                if len(channel_values) != len(selected_channels):
                    continue
                n = min(len(vals) for vals in channel_values)
                if n < window_length:
                    continue
                stacked = np.stack([np.asarray(vals[:n], dtype=np.float32) for vals in channel_values], axis=1)
                values.append(stacked)
                labels.append(np.asarray(segment_labels[:n], dtype=np.int64))
                recs.append(np.asarray([filename.replace(".edf", "")] * n))
                starts.append(np.asarray(segment_times[:n], dtype=np.int64))
    values = np.concatenate(values, axis=0).astype(np.float32)
    labels = np.concatenate(labels, axis=0).astype(np.int64)
    recs = np.concatenate(recs)
    starts = np.concatenate(starts)
    if values.size == 0:
        raise RuntimeError("No samples were returned from PostgreSQL.")
    low, high = np.percentile(values, [1, 99])
    values = np.clip(values, low, high)
    mins = values.min(axis=0, keepdims=True)
    maxs = values.max(axis=0, keepdims=True)
    values = (values - mins) / np.maximum(maxs - mins, 1e-8)
    windows, y, window_recs, window_starts = [], [], [], []
    for start in range(0, len(values) - window_length + 1, stride):
        end = start + window_length
        if len(set(recs[start:end])) > 1:
            continue
        windows.append(values[start:end])
        y.append(int(labels[start:end].mean() >= 0.5))
        window_recs.append(recs[start])
        window_starts.append(starts[start])
    return WindowBatch(
        windows=np.asarray(windows, dtype=np.float32),
        labels=np.asarray(y, dtype=np.int64),
        subject_ids=np.asarray([subject_id] * len(windows)),
        recording_ids=np.asarray(window_recs),
        channels=np.asarray(["+".join(selected_channels)] * len(windows)),
        start_times=np.asarray(window_starts),
    )


def split_healthy_windows(batch, train_ratio=0.70, val_ratio=0.15):
    healthy = np.where(batch.labels == 0)[0]
    anomalous = np.where(batch.labels == 1)[0]
    rng = np.random.default_rng(42)
    rng.shuffle(healthy)
    n_train = int(len(healthy) * train_ratio)
    n_val = int(len(healthy) * val_ratio)
    train = healthy[:n_train]
    val = healthy[n_train:n_train + n_val]
    test_healthy = healthy[n_train + n_val:]
    test = np.concatenate([test_healthy, anomalous])
    return train, val, test
