"""
ingest_tsdb.py

 – T5 : Dual DB Ingestion
Topic M6 : Synthetic Thermal Time-Series


Purpose
-------
Load all files produced by T4 (etl_pipeline.py) into a TimescaleDB database.

Tables / hypertables filled
    subjects         ← etl_summary.json  (norm params, per-patient stats)
    thermal_readings ← cleaned_csv/patient_XX_cleaned.csv  (hypertable, 5 184 000 rows)
    windows_tsdb     ← all_windows_meta.csv  (172 780 rows)

Strategy
--------
  • thermal_readings is a TimescaleDB hypertable partitioned by time.
  • Bulk load via COPY FROM STDIN chunk by chunk (CHUNK_ROWS rows at a time).
  • Compression policy applied AFTER load.
  • Measures rows/s, raw disk size, and compressed disk size.

Usage
-----
  python ingest_tsdb.py [--etl-dir PATH] [--dsn DSN] [--patients N]

  Credentials are loaded from .env automatically (no need to pass --dsn).

Requirements
------------
  pip install psycopg2-binary pandas tqdm python-dotenv
"""

import argparse
import io
import json
import logging
import os
import sys
import time
from pathlib import Path

import pandas as pd
import psycopg2
from psycopg2.extras import execute_values
from tqdm import tqdm
from dotenv import load_dotenv

# ---------------------------------------------------------------------------
# Load .env
# ---------------------------------------------------------------------------
load_dotenv()

def _build_tsdb_dsn() -> str:
    host     = os.getenv("TSDB_HOST",     "localhost")
    port     = os.getenv("TSDB_PORT",     "5432")
    user     = os.getenv("TSDB_USER",     "postgres")
    password = os.getenv("TSDB_PASSWORD", "postgres")
    db       = os.getenv("TSDB_DB",       "m6_thermal_tsdb")
    return f"postgresql://{user}:{password}@{host}:{port}/{db}"

def _default_etl_dir() -> str:
    return os.getenv("ETL_DIR", "etl_output")

# ---------------------------------------------------------------------------
# Logging
# ---------------------------------------------------------------------------
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    datefmt="%H:%M:%S",
)
log = logging.getLogger("ingest_tsdb")

CHUNK_ROWS = 50_000

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def get_conn(dsn: str):
    conn = psycopg2.connect(dsn)
    conn.autocommit = False
    return conn


def run_sql_file(conn, path: Path):
    sql = path.read_text(encoding="utf-8")
    with conn.cursor() as cur:
        cur.execute(sql)
    conn.commit()
    log.info(f"Schema applied from {path.name}")


def table_row_count(conn, table: str) -> int:
    with conn.cursor() as cur:
        cur.execute(f"SELECT COUNT(*) FROM {table};")
        return cur.fetchone()[0]


def hypertable_size_bytes(conn, table: str) -> int:
    with conn.cursor() as cur:
        cur.execute("SELECT hypertable_size(%s);", (table,))
        row = cur.fetchone()
        return row[0] if row and row[0] else 0


def hypertable_compressed_size(conn, table: str) -> dict:
    with conn.cursor() as cur:
        cur.execute("""
            SELECT
                pg_size_pretty(before_compression_total_bytes) AS before,
                pg_size_pretty(after_compression_total_bytes)  AS after,
                ROUND(
                    100.0 * (1 - after_compression_total_bytes::numeric
                               / NULLIF(before_compression_total_bytes, 0)),
                    1
                ) AS pct_saved
            FROM hypertable_compression_stats('thermal_readings');
        """)
        row = cur.fetchone()
        if row:
            return {"before": row[0], "after": row[1], "pct_saved": float(row[2])}
        return {"before": "N/A", "after": "N/A", "pct_saved": 0}


def copy_chunk(conn, df_chunk: pd.DataFrame, table: str, columns: list[str]) -> int:
    buf = io.StringIO()
    df_chunk[columns].to_csv(buf, index=False, header=False)
    buf.seek(0)
    col_str = ", ".join(columns)
    with conn.cursor() as cur:
        cur.copy_expert(
            f"COPY {table} ({col_str}) FROM STDIN WITH (FORMAT CSV, NULL '')",
            buf,
        )
    conn.commit()
    return len(df_chunk)


# ---------------------------------------------------------------------------
# Stage A : subjects
# ---------------------------------------------------------------------------

def ingest_subjects(conn, etl_dir: Path, max_patients: int) -> int:
    summary_path = etl_dir / "logs" / "etl_summary.json"
    if not summary_path.exists():
        log.error(f"etl_summary.json not found at {summary_path}")
        sys.exit(1)

    with open(summary_path) as f:
        summary = json.load(f)

    patients = summary["patients"][:max_patients]
    rows = []
    for p in patients:
        norm = p["norm_params"]
        rows.append((
            p["patient_id"],
            norm["norm_mean_left"],
            norm["norm_std_left"],
            norm["norm_mean_right"],
            norm["norm_std_right"],
            p["rows_input"],
            p["rows_output"],
            p["windows"],
            p["anomaly_windows"],
            p["segments"],
            p["interpolated_rows"],
        ))

    sql = """
        INSERT INTO subjects (
            patient_id,
            norm_mean_left, norm_std_left,
            norm_mean_right, norm_std_right,
            rows_input, rows_output,
            n_windows, n_anomaly_windows,
            n_segments, n_interpolated_rows
        ) VALUES %s
        ON CONFLICT (patient_id) DO NOTHING;
    """
    t0 = time.perf_counter()
    with conn.cursor() as cur:
        execute_values(cur, sql, rows)
    conn.commit()
    log.info(f"subjects: {len(rows)} rows in {time.perf_counter()-t0:.2f}s")
    return len(rows)


# ---------------------------------------------------------------------------
# Stage B : thermal_readings (hypertable)
# ---------------------------------------------------------------------------

THERMAL_COLUMNS = [
    "timestamp", "patient_id",
    "left_temperature", "right_temperature",
    "left_temperature_norm", "right_temperature_norm",
    "temp_asymmetry",
    "anomaly_label", "anomaly_type",
    "is_interpolated", "segment_id", "session_second",
]


def ingest_thermal_readings(conn, etl_dir: Path, max_patients: int) -> dict:
    cleaned_dir = etl_dir / "cleaned_csv"
    csv_files = sorted(cleaned_dir.glob("patient_*_cleaned.csv"))[:max_patients]

    if not csv_files:
        log.error(f"No cleaned CSV files found in {cleaned_dir}")
        sys.exit(1)

    total_rows = 0
    total_time = 0.0
    per_patient = []

    for csv_path in tqdm(csv_files, desc="thermal_readings", unit="patient"):
        pid = int(csv_path.stem.split("_")[1])
        patient_rows = 0
        patient_time = 0.0

        reader = pd.read_csv(
            csv_path,
            parse_dates=["timestamp"],
            chunksize=CHUNK_ROWS,
        )
        for chunk in reader:
            chunk["is_interpolated"] = chunk["is_interpolated"].astype(bool)
            t0 = time.perf_counter()
            copy_chunk(conn, chunk, "thermal_readings", THERMAL_COLUMNS)
            patient_time += time.perf_counter() - t0
            patient_rows += len(chunk)

        rate = patient_rows / patient_time if patient_time > 0 else float("inf")
        log.info(
            f"  patient_{pid:02d}: {patient_rows:,} rows in {patient_time:.2f}s "
            f"→ {rate:,.0f} rows/s"
        )
        total_rows += patient_rows
        total_time += patient_time
        per_patient.append({
            "patient_id": pid,
            "rows": patient_rows,
            "elapsed_s": round(patient_time, 3),
            "rows_per_s": round(rate, 0),
        })

    overall_rate = total_rows / total_time if total_time > 0 else 0
    log.info(
        f"thermal_readings TOTAL: {total_rows:,} rows in {total_time:.1f}s "
        f"→ {overall_rate:,.0f} rows/s"
    )
    return {
        "table": "thermal_readings",
        "total_rows": total_rows,
        "total_elapsed_s": round(total_time, 2),
        "overall_rows_per_s": round(overall_rate, 0),
        "per_patient": per_patient,
    }


# ---------------------------------------------------------------------------
# Stage C : windows_tsdb
# ---------------------------------------------------------------------------

WINDOWS_COLUMNS = [
    "window_id", "patient_id", "segment_id", "window_index",
    "window_start", "window_end",
    "label", "anomaly_ratio", "is_interpolated",
]


def ingest_windows(conn, etl_dir: Path) -> dict:
    meta_path = etl_dir / "all_windows_meta.csv"
    if not meta_path.exists():
        log.error(f"all_windows_meta.csv not found at {meta_path}")
        sys.exit(1)

    df = pd.read_csv(meta_path, parse_dates=["window_start", "window_end"])
    df["is_interpolated"] = df["is_interpolated"].astype(bool)

    with conn.cursor() as cur:
        cur.execute("TRUNCATE TABLE windows_tsdb;")
    conn.commit()
    log.info("windows_tsdb truncated before load")

    t0 = time.perf_counter()
    copy_chunk(conn, df, "windows_tsdb", WINDOWS_COLUMNS)
    elapsed = time.perf_counter() - t0
    rate = len(df) / elapsed if elapsed > 0 else float("inf")

    log.info(f"windows_tsdb: {len(df):,} rows in {elapsed:.2f}s → {rate:,.0f} rows/s")
    return {
        "table": "windows_tsdb",
        "total_rows": len(df),
        "elapsed_s": round(elapsed, 2),
        "rows_per_s": round(rate, 0),
    }


# ---------------------------------------------------------------------------
# Stage D : Post-load indexes + compression
# ---------------------------------------------------------------------------

def create_indexes(conn):
    log.info("Creating indexes on thermal_readings...")
    index_sql = [
        "CREATE INDEX IF NOT EXISTS idx_tr_patient_ts ON thermal_readings (patient_id, timestamp DESC);",
        "CREATE INDEX IF NOT EXISTS idx_tr_ts ON thermal_readings (timestamp DESC);",
        "CREATE INDEX IF NOT EXISTS idx_tr_anomaly ON thermal_readings (anomaly_label) WHERE anomaly_label = 1;",
    ]
    t0 = time.perf_counter()
    with conn.cursor() as cur:
        for stmt in index_sql:
            cur.execute(stmt)
    conn.commit()
    elapsed = time.perf_counter() - t0
    log.info(f"Indexes created in {elapsed:.1f}s")
    return round(elapsed, 2)


def trigger_compression(conn) -> dict:
    log.info("Compressing all thermal_readings chunks...")
    t0 = time.perf_counter()
    with conn.cursor() as cur:
        cur.execute("""
            SELECT compress_chunk(i, if_not_compressed => TRUE)
            FROM show_chunks('thermal_readings') i;
        """)
    conn.commit()
    elapsed = time.perf_counter() - t0
    log.info(f"Compression complete in {elapsed:.1f}s")

    stats = hypertable_compressed_size(conn, "thermal_readings")
    log.info(
        f"  Disk before: {stats['before']}  →  after: {stats['after']}  "
        f"({stats['pct_saved']}% saved)"
    )
    stats["compression_elapsed_s"] = round(elapsed, 2)
    return stats


# ---------------------------------------------------------------------------
# Stage E : Spot-check + size report
# ---------------------------------------------------------------------------

def spot_check_and_size(conn) -> dict:
    counts = {}
    for table in ("subjects", "thermal_readings", "windows_tsdb"):
        n = table_row_count(conn, table)
        counts[table] = n
        log.info(f"  {table:20s} : {n:,} rows")

    assert counts["thermal_readings"] >= 4_000_000, \
        f"thermal_readings too low: {counts['thermal_readings']:,}"
    assert counts["windows_tsdb"] == 172_780, \
        f"windows_tsdb expected 172 780, got {counts['windows_tsdb']:,}"
    log.info("Spot-check PASSED")

    raw_size = hypertable_size_bytes(conn, "thermal_readings")
    log.info(f"  thermal_readings raw size: {raw_size / 1e6:.1f} MB")

    return {"row_counts": counts, "thermal_readings_bytes": raw_size}


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def parse_args():
    parser = argparse.ArgumentParser(description="T5 – TimescaleDB ingestion")
    parser.add_argument("--etl-dir", default=_default_etl_dir())
    parser.add_argument("--dsn", default=_build_tsdb_dsn(),
                        help="TimescaleDB DSN (default: built from .env)")
    parser.add_argument("--patients", type=int, default=20)
    parser.add_argument(
        "--schema-file",
        default=str(Path(__file__).parent / "schema_timescaledb.sql"),
    )
    parser.add_argument("--skip-schema", action="store_true")
    parser.add_argument("--skip-compress", action="store_true",
                        help="Skip manual compression trigger")
    return parser.parse_args()


def main():
    args = parse_args()
    etl_dir = Path(args.etl_dir)

    log.info("=" * 60)
    log.info("T5 – TimescaleDB Ingestion")
    log.info(f"  ETL dir  : {etl_dir}")
    log.info(f"  DSN      : {args.dsn}")
    log.info(f"  Patients : {args.patients}")
    log.info("=" * 60)

    conn = get_conn(args.dsn)

    if not args.skip_schema:
        schema_file = Path(args.schema_file)
        if not schema_file.exists():
            log.error(f"Schema file not found: {schema_file}")
            sys.exit(1)
        run_sql_file(conn, schema_file)

    pipeline_start = time.perf_counter()

    log.info("── Stage A : subjects ──────────────────────────")
    n_subjects = ingest_subjects(conn, etl_dir, args.patients)

    log.info("── Stage B : thermal_readings (hypertable) ──────")
    thermal_metrics = ingest_thermal_readings(conn, etl_dir, args.patients)

    log.info("── Stage C : windows_tsdb ──────────────────────")
    windows_metrics = ingest_windows(conn, etl_dir)

    log.info("── Stage D : indexes ───────────────────────────")
    idx_elapsed = create_indexes(conn)

    compression_stats = {"skipped": True}
    if not args.skip_compress:
        log.info("── Stage E : compression ───────────────────────")
        compression_stats = trigger_compression(conn)

    log.info("── Stage F : spot-check ────────────────────────")
    check = spot_check_and_size(conn)

    total_elapsed = time.perf_counter() - pipeline_start
    report = {
        "database":         "timescaledb",
        "dsn":              args.dsn,
        "total_elapsed_s":  round(total_elapsed, 2),
        "n_subjects":       n_subjects,
        "thermal_readings": thermal_metrics,
        "windows":          windows_metrics,
        "index_build_s":    idx_elapsed,
        "compression":      compression_stats,
        "spot_check":       check,
    }

    report_path = etl_dir / "logs" / "ingest_tsdb_report.json"
    with open(report_path, "w") as f:
        json.dump(report, f, indent=2)

    log.info("=" * 60)
    log.info(f"TimescaleDB ingestion COMPLETE in {total_elapsed:.1f}s")
    log.info(f"Report saved → {report_path}")
    log.info("=" * 60)

    conn.close()


if __name__ == "__main__":
    main()
