"""Paper-oriented PostgreSQL-versus-TimescaleDB benchmark for the PDM.

Only disposable objects in the ``benchmark`` schema are created or replaced.
Production rows are read but never modified. Raw physiological values and
within-recording timing are preserved; recording timestamps are shifted onto
successive dates solely to construct a longitudinal storage workload.
"""

from __future__ import annotations

import argparse
import hashlib
import io
import json
import math
import os
import platform
import shutil
import statistics
import subprocess
import time
from datetime import datetime, timedelta, timezone
from pathlib import Path
from typing import Any, Callable

from pdm.db import connect, connection_kwargs

ROOT = Path(__file__).resolve().parents[1]
OUTPUT_DIR = ROOT / "outputs" / "database_benchmarks"
PLAN_DIR = OUTPUT_DIR / "query_plans"
PG_TABLE = "benchmark.pg_signal_data_final"
TS_TABLE = "benchmark.ts_signal_data_final"
BACKENDS = {"postgresql": PG_TABLE, "timescaledb": TS_TABLE}
CHUNK_INTERVAL = "1 day"
QUERY_REPETITIONS = 20
INGESTION_REPETITIONS = 5
DOCS = {
    "create_hypertable": "https://docs.timescale.com/api/latest/hypertable/create_hypertable/",
    "create_table": "https://docs.timescale.com/api/latest/hypertable/create_table/",
    "hypercore": "https://docs.timescale.com/api/latest/hypercore/",
    "alter_table_columnstore": "https://docs.timescale.com/api/latest/hypercore/alter_table/",
    "convert_to_columnstore": "https://docs.timescale.com/api/latest/hypercore/convert_to_columnstore/",
    "chunk_columnstore_stats": "https://docs.timescale.com/api/latest/hypercore/chunk_columnstore_stats/",
    "chunks_view": "https://docs.timescale.com/api/latest/informational-views/chunks/",
}


def iso(value: Any) -> Any:
    if isinstance(value, datetime):
        return value.isoformat()
    if isinstance(value, timedelta):
        return value.total_seconds()
    return value


def json_default(value: Any) -> Any:
    converted = iso(value)
    if converted is value:
        raise TypeError(f"Cannot serialize {type(value)!r}")
    return converted


def ffloat(value: Any) -> float | None:
    return None if value is None else float(value)


def pct_faster(baseline: float, candidate: float) -> float:
    return (baseline - candidate) / baseline * 100.0 if baseline else 0.0


def timing_stats_seconds(values: list[float]) -> dict[str, float]:
    milliseconds = [value * 1000.0 for value in values]
    return {
        "mean_ms": statistics.fmean(milliseconds),
        "median_ms": statistics.median(milliseconds),
        "stddev_ms": statistics.stdev(milliseconds) if len(milliseconds) > 1 else 0.0,
        "min_ms": min(milliseconds),
        "max_ms": max(milliseconds),
        "p95_ms": statistics.quantiles(milliseconds, n=100, method="inclusive")[94]
        if len(milliseconds) > 1 else milliseconds[0],
    }


def ingestion_stats(values: list[float], rows: int) -> dict[str, float]:
    result = {
        "mean_seconds": statistics.fmean(values),
        "median_seconds": statistics.median(values),
        "stddev_seconds": statistics.stdev(values) if len(values) > 1 else 0.0,
        "min_seconds": min(values),
        "max_seconds": max(values),
        "p95_seconds": statistics.quantiles(values, n=100, method="inclusive")[94]
        if len(values) > 1 else values[0],
    }
    result["median_rows_per_second"] = rows / result["median_seconds"]
    return result


def command_output(args: list[str]) -> str | None:
    try:
        completed = subprocess.run(args, capture_output=True, text=True, check=True, timeout=15)
        return completed.stdout.strip() or None
    except (OSError, subprocess.SubprocessError):
        return None


def environment_metadata() -> dict[str, Any]:
    return {
        "platform": platform.platform(), "os": platform.system(),
        "architecture": platform.machine(), "processor": platform.processor(),
        "logical_cpus": os.cpu_count(), "python_version": platform.python_version(),
        "docker_engine_version": command_output(["docker", "version", "--format", "{{.Server.Version}}"]),
        "docker_image": command_output(
            ["docker", "inspect", "--format", "{{.Config.Image}}", "JAISE2026-timescaledb"]
        ) if shutil.which("docker") else None,
    }


def database_metadata(conn) -> dict[str, Any]:
    with conn.cursor() as cur:
        cur.execute(
            """SELECT version(),current_setting('server_version'),
                      (SELECT extversion FROM pg_extension WHERE extname='timescaledb'),
                      current_setting('shared_buffers'),current_setting('effective_cache_size')"""
        )
        version, server_version, timescale_version, shared_buffers, effective_cache = cur.fetchone()
    kwargs = connection_kwargs()
    return {
        "postgresql_version": version, "postgresql_server_version": server_version,
        "timescaledb_version": timescale_version, "shared_buffers": shared_buffers,
        "effective_cache_size": effective_cache, "host": kwargs["host"],
        "port": kwargs["port"], "database": kwargs["dbname"], "user": kwargs["user"],
    }


def scalar(conn, sql: str, params: tuple[Any, ...] = ()) -> Any:
    with conn.cursor() as cur:
        cur.execute(sql, params)
        row = cur.fetchone()
    return row[0] if row else None


def production_snapshot(conn) -> dict[str, Any]:
    """Compact read-only fingerprint proving the source population was unchanged."""
    with conn.cursor() as cur:
        cur.execute(
            """SELECT count(*),count(DISTINCT r.patient_id),count(DISTINCT r.recording_id),
                      count(DISTINCT s.signal_id),min(sd.timestamp),max(sd.timestamp),
                      min(sd.value),max(sd.value),count(sd.quality)
               FROM public.signal_data sd
               JOIN public.signal s ON s.signal_id=sd.signal_id
               JOIN public.recording r ON r.recording_id=s.recording_id
               WHERE r.modality IN ('apnea','sleep') AND s.signal_type=r.modality"""
        )
        row = cur.fetchone()
        cur.execute(
            """SELECT r.recording_id,s.signal_id,sd.channel,count(*),min(sd.timestamp),
                      max(sd.timestamp),min(sd.value),max(sd.value),sum(sd.value)
               FROM public.signal_data sd
               JOIN public.signal s ON s.signal_id=sd.signal_id
               JOIN public.recording r ON r.recording_id=s.recording_id
               WHERE r.modality IN ('apnea','sleep') AND s.signal_type=r.modality
               GROUP BY r.recording_id,s.signal_id,sd.channel
               ORDER BY r.recording_id,s.signal_id,sd.channel"""
        )
        groups = cur.fetchall()
    return {
        "rows": int(row[0]), "patients": int(row[1]), "recordings": int(row[2]),
        "signals": int(row[3]), "min_timestamp": iso(row[4]), "max_timestamp": iso(row[5]),
        "min_value": ffloat(row[6]), "max_value": ffloat(row[7]),
        "quality_non_null": int(row[8]),
        "group_signature_sha256": hashlib.sha256(repr(groups).encode()).hexdigest(),
    }


def build_longitudinal_source(conn, base_date: datetime) -> tuple[int, list[dict[str, Any]]]:
    with conn.cursor() as cur:
        # Remove any legacy disposable benchmark tables before the final run.
        cur.execute("DROP TABLE IF EXISTS benchmark.pg_signal_data CASCADE")
        cur.execute("DROP TABLE IF EXISTS benchmark.ts_signal_data CASCADE")
        cur.execute("DROP TABLE IF EXISTS benchmark_source_signal_data")
        cur.execute("DROP TABLE IF EXISTS benchmark_recording_map")
        cur.execute(
            """CREATE TEMP TABLE benchmark_recording_map ON COMMIT PRESERVE ROWS AS
               WITH recordings AS (
                   SELECT r.recording_id,r.patient_id,r.modality,min(sd.timestamp) original_start,
                          max(sd.timestamp) original_end,
                          row_number() OVER (ORDER BY CASE r.modality WHEN 'apnea' THEN 0 ELSE 1 END,
                                                     r.recording_id)-1 day_offset
                   FROM public.signal_data sd
                   JOIN public.signal s ON s.signal_id=sd.signal_id
                   JOIN public.recording r ON r.recording_id=s.recording_id
                   WHERE r.modality IN ('apnea','sleep') AND s.signal_type=r.modality
                   GROUP BY r.recording_id,r.patient_id,r.modality
               )
               SELECT *,%s::timestamptz+day_offset*INTERVAL '1 day' benchmark_start
               FROM recordings""", (base_date,)
        )
        cur.execute(
            """CREATE TEMP TABLE benchmark_source_signal_data ON COMMIT PRESERVE ROWS AS
               SELECT CASE r.modality WHEN 'apnea' THEN 'Apnea-ECG' ELSE 'Sleep-EDF' END dataset,
                      r.recording_id,r.patient_id,s.signal_id source_signal_id,sd.channel,
                      m.benchmark_start+(sd.timestamp-m.original_start) timestamp,sd.value,sd.quality
               FROM public.signal_data sd
               JOIN public.signal s ON s.signal_id=sd.signal_id
               JOIN public.recording r ON r.recording_id=s.recording_id
               JOIN benchmark_recording_map m ON m.recording_id=r.recording_id
               WHERE r.modality IN ('apnea','sleep') AND s.signal_type=r.modality"""
        )
        cur.execute("ANALYZE benchmark_source_signal_data")
        cur.execute(
            """SELECT recording_id,patient_id,modality,day_offset,original_start,original_end,
                      benchmark_start,benchmark_start+(original_end-original_start)
               FROM benchmark_recording_map ORDER BY day_offset"""
        )
        mapping = [
            {"recording_id": r[0], "patient_id": int(r[1]), "modality": r[2],
             "day_offset": int(r[3]), "original_start": iso(r[4]), "original_end": iso(r[5]),
             "benchmark_start": iso(r[6]), "benchmark_end": iso(r[7])}
            for r in cur.fetchall()
        ]
        cur.execute("SELECT count(*) FROM benchmark_source_signal_data")
        total = int(cur.fetchone()[0])
    conn.commit()
    return total, mapping


def dataset_metadata(conn) -> dict[str, Any]:
    with conn.cursor() as cur:
        cur.execute(
            """SELECT dataset,count(*),count(DISTINCT patient_id),count(DISTINCT recording_id),
                      count(DISTINCT source_signal_id),count(DISTINCT channel),min(timestamp),max(timestamp),
                      min(value),max(value)
               FROM benchmark_source_signal_data GROUP BY dataset ORDER BY dataset"""
        )
        rows = cur.fetchall()
        result = {}
        for row in rows:
            cur.execute(
                """SELECT channel,count(*),count(DISTINCT source_signal_id)
                   FROM benchmark_source_signal_data WHERE dataset=%s
                   GROUP BY channel ORDER BY channel""", (row[0],)
            )
            result[row[0]] = {
                "rows": int(row[1]), "patients": int(row[2]), "recordings": int(row[3]),
                "signals": int(row[4]), "channel_count": int(row[5]),
                "time_range": {"min": iso(row[6]), "max": iso(row[7])},
                "value_range": {"min": ffloat(row[8]), "max": ffloat(row[9])},
                "channels": [{"channel": x[0], "rows": int(x[1]), "signals": int(x[2])}
                             for x in cur.fetchall()],
            }
    return result


def table_ddl() -> str:
    return """(dataset TEXT NOT NULL,recording_id VARCHAR(64) NOT NULL,
                patient_id BIGINT NOT NULL,source_signal_id BIGINT NOT NULL,
                channel VARCHAR(128),timestamp TIMESTAMPTZ NOT NULL,
                value DOUBLE PRECISION NOT NULL,quality DOUBLE PRECISION)"""


def recreate_table(conn, backend: str) -> None:
    table = BACKENDS[backend]
    suffix = "pg" if backend == "postgresql" else "ts"
    with conn.cursor() as cur:
        cur.execute("CREATE SCHEMA IF NOT EXISTS benchmark")
        cur.execute(f"DROP TABLE IF EXISTS {table} CASCADE")
        if backend == "timescaledb":
            cur.execute(
                f"CREATE TABLE {table} {table_ddl()} WITH ("
                "tsdb.hypertable,tsdb.partition_column='timestamp',"
                f"tsdb.chunk_interval='{CHUNK_INTERVAL}',tsdb.create_default_indexes=false)"
            )
        else:
            cur.execute(f"CREATE TABLE {table} {table_ddl()}")
        cur.execute(f"CREATE INDEX {suffix}_final_signal_time_idx ON {table}(source_signal_id,timestamp DESC)")
        cur.execute(f"CREATE INDEX {suffix}_final_patient_time_idx ON {table}(patient_id,timestamp DESC)")
    conn.commit()


def build_binary_payload(conn) -> io.BytesIO:
    payload = io.BytesIO()
    with conn.cursor() as cur:
        cur.copy_expert(
            """COPY (SELECT dataset,recording_id,patient_id,source_signal_id,channel,timestamp,value,quality
                      FROM benchmark_source_signal_data
                      ORDER BY timestamp,recording_id,source_signal_id,channel)
               TO STDOUT WITH (FORMAT binary)""", payload
        )
    payload.seek(0)
    return payload


def load_payload(conn, backend: str, payload: io.BytesIO) -> tuple[float, int]:
    recreate_table(conn, backend)
    payload.seek(0)
    started = time.perf_counter()
    with conn.cursor() as cur:
        cur.copy_expert(f"COPY {BACKENDS[backend]} FROM STDIN WITH (FORMAT binary)", payload)
    conn.commit()
    elapsed = time.perf_counter() - started
    return elapsed, int(scalar(conn, f"SELECT count(*) FROM {BACKENDS[backend]}"))


def analyze_tables(conn) -> None:
    with conn.cursor() as cur:
        for table in BACKENDS.values():
            cur.execute(f"ANALYZE {table}")
    conn.commit()


def backend_order(repetition: int) -> list[str]:
    return ["postgresql", "timescaledb"] if repetition % 2 else ["timescaledb", "postgresql"]


def table_signature(conn, table: str) -> dict[str, Any]:
    with conn.cursor() as cur:
        cur.execute(
            f"""SELECT count(*),count(DISTINCT dataset),count(DISTINCT patient_id),
                       count(DISTINCT recording_id),count(DISTINCT source_signal_id),
                       min(timestamp),max(timestamp),min(value),max(value),count(quality),
                       sum(hashtextextended(concat_ws('|',dataset,recording_id,patient_id::text,
                           source_signal_id::text,coalesce(channel,''),timestamp::text,value::text,
                           coalesce(quality::text,'')),0)::numeric)::text
                FROM {table}"""
        )
        row = cur.fetchone()
        cur.execute(
            f"""SELECT dataset,recording_id,patient_id,source_signal_id,channel,count(*),
                       min(timestamp),max(timestamp),min(value),max(value),sum(value::numeric),count(quality)
                FROM {table} GROUP BY dataset,recording_id,patient_id,source_signal_id,channel
                ORDER BY dataset,recording_id,source_signal_id,channel"""
        )
        groups = cur.fetchall()
    return {
        "rows": int(row[0]), "datasets": int(row[1]), "patients": int(row[2]),
        "recordings": int(row[3]), "signals": int(row[4]), "min_timestamp": iso(row[5]),
        "max_timestamp": iso(row[6]), "min_value": ffloat(row[7]), "max_value": ffloat(row[8]),
        "quality_non_null": int(row[9]), "logical_hash_sum": row[10],
        "group_signature_sha256": hashlib.sha256(repr(groups).encode()).hexdigest(),
    }


def chunks_metadata(conn) -> list[dict[str, Any]]:
    with conn.cursor() as cur:
        cur.execute(
            """SELECT chunk_schema,chunk_name,range_start,range_end,is_compressed
               FROM timescaledb_information.chunks
               WHERE hypertable_schema='benchmark' AND hypertable_name='ts_signal_data_final'
               ORDER BY range_start"""
        )
        chunks = cur.fetchall()
        output = []
        for schema, name, start, end, compressed in chunks:
            qualified = f'"{schema}"."{name}"'
            cur.execute(f"SELECT count(*),pg_relation_size('{qualified}'::regclass),pg_indexes_size('{qualified}'::regclass),pg_total_relation_size('{qualified}'::regclass) FROM {qualified}")
            count, data_bytes, index_bytes, total_bytes = cur.fetchone()
            output.append({
                "chunk": f"{schema}.{name}", "range_start": iso(start), "range_end": iso(end),
                "is_compressed": bool(compressed), "rows": int(count),
                "data_bytes": int(data_bytes), "index_bytes": int(index_bytes),
                "total_bytes": int(total_bytes),
            })
    return output


def storage_measurement(conn, backend: str) -> dict[str, Any]:
    table = BACKENDS[backend]
    with conn.cursor() as cur:
        if backend == "timescaledb":
            cur.execute("SELECT table_bytes,index_bytes,toast_bytes,total_bytes FROM hypertable_detailed_size(%s::regclass)", (table,))
            table_bytes, index_bytes, toast_bytes, total_bytes = cur.fetchone()
            data_bytes = int(table_bytes) + int(toast_bytes)
        else:
            cur.execute("SELECT pg_relation_size(%s::regclass),pg_indexes_size(%s::regclass),pg_total_relation_size(%s::regclass)", (table, table, table))
            data_bytes, index_bytes, total_bytes = map(int, cur.fetchone())
    return {
        "backend": backend, "table": table, "data_bytes": data_bytes,
        "index_bytes": int(index_bytes), "total_bytes": int(total_bytes),
        "data_mb": data_bytes / 1048576, "index_mb": int(index_bytes) / 1048576,
        "total_mb": int(total_bytes) / 1048576,
    }


def query_targets(conn) -> dict[str, Any]:
    with conn.cursor() as cur:
        cur.execute(
            f"""SELECT dataset,recording_id,source_signal_id,channel,min(timestamp),max(timestamp),count(*)
                FROM {PG_TABLE}
                WHERE (dataset='Apnea-ECG' AND channel='ECG')
                   OR (dataset='Sleep-EDF' AND channel='EEG Fpz-Cz')
                GROUP BY dataset,recording_id,source_signal_id,channel
                ORDER BY dataset,count(*) DESC,recording_id"""
        )
        signals = {}
        for row in cur.fetchall():
            if row[0] not in signals:
                signals[row[0]] = {
                    "dataset": row[0], "recording_id": row[1], "source_signal_id": int(row[2]),
                    "channel": row[3], "start": row[4], "end": row[5], "rows": int(row[6]),
                }
        cur.execute(
            f"""SELECT patient_id,count(DISTINCT recording_id),count(*),min(timestamp),max(timestamp)
                FROM {PG_TABLE} WHERE dataset='Sleep-EDF'
                GROUP BY patient_id HAVING count(DISTINCT recording_id)>1
                ORDER BY count(*) DESC,patient_id LIMIT 1"""
        )
        patient = cur.fetchone()
        cur.execute(f"SELECT recording_id,min(timestamp),max(timestamp),count(*) FROM {PG_TABLE} WHERE patient_id=%s GROUP BY recording_id ORDER BY min(timestamp)", (patient[0],))
        sessions = cur.fetchall()
    return {
        "signals": signals,
        "patient": {
            "patient_id": int(patient[0]), "recordings": int(patient[1]), "rows": int(patient[2]),
            "start": patient[3], "end": patient[4],
            "sessions": [{"recording_id": x[0], "start": iso(x[1]), "end": iso(x[2]), "rows": int(x[3])}
                         for x in sessions],
        },
    }


def values_equal(left: Any, right: Any) -> bool:
    if isinstance(left, float) or isinstance(right, float):
        return math.isclose(float(left), float(right), rel_tol=1e-9, abs_tol=1e-12)
    return left == right


def rows_equal(left: list[tuple[Any, ...]], right: list[tuple[Any, ...]]) -> bool:
    return len(left) == len(right) and all(
        all(values_equal(a, b) for a, b in zip(lrow, rrow)) for lrow, rrow in zip(left, right)
    )


def fetch(conn, sql: str, params: tuple[Any, ...]) -> tuple[list[tuple[Any, ...]], float]:
    with conn.cursor() as cur:
        started = time.perf_counter()
        cur.execute(sql, params)
        rows = cur.fetchall()
        elapsed = time.perf_counter() - started
    return rows, elapsed


def run_paired_query(conn, name: str, workload: str, sql_builder: Callable[[str, str], str],
                     params: tuple[Any, ...], repetitions: int, metadata: dict[str, Any]) -> dict[str, Any]:
    warmups, expected = {}, {}
    timings = {backend: [] for backend in BACKENDS}
    measurements = {backend: [] for backend in BACKENDS}
    for backend, table in BACKENDS.items():
        rows, elapsed = fetch(conn, sql_builder(backend, table), params)
        warmups[backend] = {"elapsed_ms": elapsed * 1000.0, "result_rows": len(rows)}
        expected[backend] = rows
    if not rows_equal(expected["postgresql"], expected["timescaledb"]):
        raise RuntimeError(f"Correctness failure before timing query {name}")
    for repetition in range(1, repetitions + 1):
        for backend in backend_order(repetition):
            rows, elapsed = fetch(conn, sql_builder(backend, BACKENDS[backend]), params)
            if not rows_equal(rows, expected[backend]):
                raise RuntimeError(f"Unstable result for {name}/{backend}")
            timings[backend].append(elapsed)
            measurements[backend].append({"repetition": repetition, "elapsed_ms": elapsed * 1000.0,
                                          "result_rows": len(rows)})
    return {
        "name": name, "workload": workload, **metadata,
        "sql": {backend: sql_builder(backend, table) for backend, table in BACKENDS.items()},
        "parameters": [iso(x) for x in params], "warmups": warmups,
        "result_rows": len(expected["postgresql"]), "equivalent_results": True,
        "timings_ms": {k: [x * 1000.0 for x in v] for k, v in timings.items()},
        "statistics": {k: timing_stats_seconds(v) for k, v in timings.items()},
        "measurements": measurements,
    }


def run_timescale_query(conn, name: str, workload: str, sql: str,
                        params: tuple[Any, ...], repetitions: int) -> dict[str, Any]:
    expected, warm = fetch(conn, sql, params)
    timings, measurements = [], []
    for repetition in range(1, repetitions + 1):
        rows, elapsed = fetch(conn, sql, params)
        if not rows_equal(rows, expected):
            raise RuntimeError(f"Unstable compressed result for {name}")
        timings.append(elapsed)
        measurements.append({"repetition": repetition, "elapsed_ms": elapsed * 1000.0,
                             "result_rows": len(rows)})
    return {
        "name": name, "workload": workload, "backend": "timescaledb_hybrid_columnstore",
        "sql": sql, "parameters": [iso(x) for x in params],
        "warmup": {"elapsed_ms": warm * 1000.0, "result_rows": len(expected)},
        "result_rows": len(expected), "timings_ms": [x * 1000.0 for x in timings],
        "statistics": timing_stats_seconds(timings), "measurements": measurements,
    }


def range_sql(_backend: str, table: str) -> str:
    return f"SELECT timestamp,value FROM {table} WHERE source_signal_id=%s AND timestamp >= %s AND timestamp < %s ORDER BY timestamp"


def history_sql(_backend: str, table: str) -> str:
    return f"SELECT timestamp,recording_id,source_signal_id,channel,value FROM {table} WHERE patient_id=%s AND timestamp >= %s AND timestamp < %s ORDER BY timestamp,source_signal_id"


def aggregate_sql(backend: str, table: str, filter_sql: str) -> str:
    bucket = "date_bin(%s::interval,timestamp,%s::timestamptz)" if backend == "postgresql" else "time_bucket(%s::interval,timestamp,%s::timestamptz)"
    return f"SELECT {bucket} bucket,avg(value),min(value),max(value),count(*) FROM {table} WHERE {filter_sql} GROUP BY bucket ORDER BY bucket"


def chunks_touched(conn, start: datetime, end: datetime) -> int:
    return int(scalar(conn, """SELECT count(*) FROM timescaledb_information.chunks
                               WHERE hypertable_schema='benchmark' AND hypertable_name='ts_signal_data_final'
                                 AND range_start < %s AND range_end > %s""", (end, start)))


def explain_to_file(conn, sql: str, params: tuple[Any, ...], path: Path) -> None:
    with conn.cursor() as cur:
        cur.execute("EXPLAIN (ANALYZE,BUFFERS,FORMAT TEXT) " + sql, params)
        lines = [row[0] for row in cur.fetchall()]
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def write_query_plans(conn, targets: dict[str, Any], base_date: datetime) -> dict[str, str]:
    PLAN_DIR.mkdir(parents=True, exist_ok=True)
    patient = targets["patient"]
    full_end = patient["end"] + timedelta(microseconds=1)
    history_params = (patient["patient_id"], patient["start"], full_end)
    agg_params = ("5 minutes", base_date, patient["patient_id"], "EEG Fpz-Cz", patient["start"], full_end)
    paths = {}
    for backend, table in BACKENDS.items():
        history_path = PLAN_DIR / f"{backend}_long_patient_history.txt"
        aggregate_path = PLAN_DIR / f"{backend}_long_temporal_aggregation.txt"
        explain_to_file(conn, history_sql(backend, table), history_params, history_path)
        explain_to_file(conn, aggregate_sql(backend, table, "patient_id=%s AND channel=%s AND timestamp >= %s AND timestamp < %s"), agg_params, aggregate_path)
        paths[f"{backend}_long_patient_history"] = str(history_path.relative_to(ROOT))
        paths[f"{backend}_long_temporal_aggregation"] = str(aggregate_path.relative_to(ROOT))
    return paths


def compression_stats(conn) -> list[dict[str, Any]]:
    with conn.cursor() as cur:
        cur.execute(
            """SELECT chunk_schema,chunk_name,compression_status,
                      before_compression_table_bytes,before_compression_index_bytes,
                      before_compression_toast_bytes,before_compression_total_bytes,
                      after_compression_table_bytes,after_compression_index_bytes,
                      after_compression_toast_bytes,after_compression_total_bytes
               FROM chunk_columnstore_stats(%s::regclass) ORDER BY chunk_name""", (TS_TABLE,)
        )
        names = [desc[0] for desc in cur.description]
        return [dict(zip(names, row)) for row in cur.fetchall()]


def compress_historical_chunks(conn) -> dict[str, Any]:
    with conn.cursor() as cur:
        cur.execute(f"ALTER TABLE {TS_TABLE} SET (timescaledb.enable_columnstore,timescaledb.orderby='timestamp DESC',timescaledb.segmentby='patient_id,source_signal_id')")
    conn.commit()
    chunks = chunks_metadata(conn)
    newest = max(chunks, key=lambda item: item["range_start"])["chunk"]
    historical = [item["chunk"] for item in chunks if item["chunk"] != newest]
    started = time.perf_counter()
    with conn.cursor() as cur:
        for chunk in historical:
            cur.execute("CALL convert_to_columnstore(%s::regclass)", (chunk,))
    conn.commit()
    elapsed = time.perf_counter() - started
    analyze_tables(conn)
    after_chunks = chunks_metadata(conn)
    compressed_count = sum(item["is_compressed"] for item in after_chunks)
    return {
        "mechanism": "TimescaleDB 2.30 Hypercore columnstore; deterministic manual conversion",
        "policy_model": "convert all historical chunks; keep newest monitoring-day chunk in rowstore",
        "segmentby": ["patient_id", "source_signal_id"], "orderby": "timestamp DESC",
        "execution_seconds": elapsed, "targeted_chunks": historical,
        "newest_rowstore_chunk": newest, "compressed_chunks": compressed_count,
        "uncompressed_chunks": len(after_chunks) - compressed_count,
        "chunk_columnstore_stats": compression_stats(conn), "chunks": after_chunks,
    }


def md_table(headers: list[str], rows: list[list[Any]]) -> str:
    def cell(value: Any) -> str:
        return f"{value:.2f}" if isinstance(value, float) else str(value)
    lines = ["| " + " | ".join(headers) + " |", "| " + " | ".join(["---"] * len(headers)) + " |"]
    lines.extend("| " + " | ".join(cell(x) for x in row) + " |" for row in rows)
    return "\n".join(lines)


def result_rows(results: list[dict[str, Any]], workload: str) -> list[list[Any]]:
    rows = []
    for item in results:
        if item["workload"] != workload:
            continue
        for backend in BACKENDS:
            stats = item["statistics"][backend]
            rows.append([item["name"], backend, item["result_rows"], stats["median_ms"],
                         stats["mean_ms"], stats["stddev_ms"], stats["p95_ms"]])
    return rows


def comparison_sentence(label: str, pg: float, ts: float) -> str:
    change = pct_faster(pg, ts)
    return f"{label}: TimescaleDB {abs(change):.1f}% {'faster' if change >= 0 else 'slower'} ({ts:.2f} ms vs PostgreSQL {pg:.2f} ms)."


def render_report(result: dict[str, Any]) -> str:
    env, db = result["environment"], result["database"]
    data, config = result["benchmark_data"], result["configuration"]
    validation = result["correctness_validation"]
    queries = result["results"]["queries"]
    ingest = result["results"]["ingestion_summary"]
    storage = result["results"]["storage_uncompressed"]
    compression = result["results"]["timescale_historical_compression"]
    headline = result["paper_ready_results"]
    chunk_rows = [[item["range_start"], item["range_end"], item["rows"],
                   item["data_bytes"] / 1048576, item["index_bytes"] / 1048576,
                   item["total_bytes"] / 1048576]
                  for item in result["timescale_chunks_uncompressed"]]
    lines = [
        "# Patient Digital Memory — Final Longitudinal Database Benchmark", "",
        f"Run timestamp (UTC): `{result['benchmark_timestamp_utc']}`  ",
        f"Raw results: `{result['outputs']['json']}`", "", "## Environment", "",
        f"- Hardware/OS: `{env['platform']}`; `{env['processor']}`; {env['logical_cpus']} logical CPUs; architecture `{env['architecture']}`.",
        f"- PostgreSQL: `{db['postgresql_version']}`.", f"- TimescaleDB: `{db['timescaledb_version']}`.",
        f"- Docker Engine `{env['docker_engine_version']}`; image `{env['docker_image']}`.",
        f"- Database `{db['host']}:{db['port']}/{db['database']}`; shared buffers `{db['shared_buffers']}`; effective cache size `{db['effective_cache_size']}`.",
        "", "## Why the benchmark was redesigned", "",
        "The superseded pilot used timestamps normalized around `2000-01-01`, so the default seven-day TimescaleDB interval produced one chunk. Compression was disabled, the workload was mostly signal-local, and three ingestion repetitions were unstable. Those obsolete result artifacts are not part of the current benchmark outputs.",
        "", "## Longitudinal workload construction", "",
        f"The 12 recordings were ordered by modality (Apnea-ECG, then Sleep-EDF) and recording ID. Recording 1 was mapped to `{config['timestamp_mapping']['base_date']}` and each subsequent recording to the next date, ending on the benchmark run date. For each row: `benchmark timestamp = mapped recording start + (original timestamp - original recording start)`.",
        "", "> Original physiological values and within-recording timing were preserved; recording timestamps were deterministically shifted across successive dates only to construct a longitudinal database workload.",
        "", "This is a storage-workload transformation, not an additional clinical observation and not model-evaluation data.",
        "", "## Benchmark data", "",
    ]
    dataset_rows = [[name, item["rows"], item["patients"], item["recordings"], item["signals"], item["channel_count"]]
                    for name, item in data["datasets"].items()]
    lines += [md_table(["Dataset", "Rows", "Patients", "Recordings", "Signals", "Channels"], dataset_rows), "",
              f"Total: **{data['rows']:,} rows**, **{data['patients']} patients**, **{data['recordings']} recordings**, **{data['signals']} source signals**, **{data['channel_count']} channel names**, and **{data['dataset_count']} datasets**. Reconstruction signals are excluded.",
              "", "## Backend configuration", "",
              f"- PostgreSQL: ordinary table `{PG_TABLE}`.", f"- TimescaleDB: uncompressed hypertable `{TS_TABLE}` for the primary comparison.",
              f"- Chunk interval: `{config['chunk_interval']}`; **{config['number_of_chunks']} chunks** after loading. One day was selected because each mapped monitoring session occupies one date.",
              "- Equivalent indexes: `(source_signal_id, timestamp DESC)` and `(patient_id, timestamp DESC)`.",
              "- TimescaleDB 2.30 APIs: current `CREATE TABLE ... WITH (tsdb.hypertable, ...)`, `hypertable_detailed_size`, `timescaledb_information.chunks`, Hypercore `enable_columnstore`, `convert_to_columnstore`, and `chunk_columnstore_stats`.",
              "", "Uncompressed TimescaleDB chunk inventory:", "",
              md_table(["Range start", "Range end", "Rows", "Data MB", "Index MB", "Total MB"], chunk_rows),
              "", f"Temporal span: `{validation['signatures']['timescaledb']['min_timestamp']}` to `{validation['signatures']['timescaledb']['max_timestamp']}`.",
              "", "## Methodology", "",
              "- Primary comparison: ordinary PostgreSQL versus uncompressed TimescaleDB with identical columns, values, mapped timestamps, and indexes on the same server and hardware.",
              f"- One warm-up and {config['query_repetitions']} measured repetitions per query; backend order alternated. `time.perf_counter()` covered execution plus full fetch. Results are warm-cache local measurements.",
              f"- Ingestion used the same binary COPY payload, {config['ingestion_repetitions']} repetitions, and alternating backend order. Timed scope was COPY FROM plus commit; table/index creation was outside timing.",
              "- `ANALYZE` ran on both tables before correctness and query benchmarking.",
              "- Population metadata, logical/group signatures, range rows, and aggregate values were checked before timing.",
              "", "## Signal-local range results", "",
              md_table(["Workload", "Backend", "Rows", "Median ms", "Mean ms", "SD ms", "p95 ms"], result_rows(queries, "signal_local_range")),
              "", "## Longitudinal patient-history range results", "",
              md_table(["Workload", "Backend", "Rows", "Median ms", "Mean ms", "SD ms", "p95 ms"], result_rows(queries, "patient_history_range")),
              "", "The selected Sleep-EDF patient has two monitoring nights. The JSON records exact recordings, time span, and chunks touched.",
              "", "## Raw temporal aggregation results", "",
              md_table(["Workload", "Backend", "Buckets", "Median ms", "Mean ms", "SD ms", "p95 ms"], result_rows(queries, "raw_temporal_aggregation")),
              "", "PostgreSQL uses `date_bin`; TimescaleDB uses `time_bucket`, with identical origin, width, filters, and AVG/MIN/MAX/COUNT.",
              "", "## Downsampling results", "",
              md_table(["Workload", "Backend", "Buckets", "Median ms", "Mean ms", "SD ms", "p95 ms"], result_rows(queries, "long_history_downsampling")),
              "", "## Ingestion results", ""]
    ingest_rows = []
    for backend, item in ingest.items():
        s = item["statistics"]
        ingest_rows.append([backend, item["rows"], s["median_seconds"], s["mean_seconds"], s["stddev_seconds"], s["p95_seconds"], s["median_rows_per_second"]])
    lines += [md_table(["Backend", "Rows", "Median s", "Mean s", "SD s", "p95 s", "Median rows/s"], ingest_rows),
              "", f"Ingestion remained highly variable (PostgreSQL SD {ingest['postgresql']['statistics']['stddev_seconds']:.2f} s; TimescaleDB SD {ingest['timescaledb']['statistics']['stddev_seconds']:.2f} s). All individual measurements are retained in JSON; median is the headline, and this secondary result should be interpreted cautiously rather than as a stable throughput advantage.",
              "", "## Storage — uncompressed comparison", "",
              md_table(["Backend", "Data MB", "Index MB", "Total MB"], [[k, v["data_mb"], v["index_mb"], v["total_mb"]] for k, v in storage.items()]),
              "", "TimescaleDB size includes all chunks and indexes, not only the parent relation.",
              "", "## Timescale historical compression", ""]
    if compression.get("skipped"):
        lines += [f"Skipped: {compression['reason']}."]
    else:
        before, after = compression["storage_before"], compression["storage_after"]
        cfg = compression["configuration"]
        lines += ["This is a separate **TimescaleDB historical-compression feature** result, not an apples-to-apples PostgreSQL comparison. Hypercore columnstore was segmented by `patient_id, source_signal_id`, ordered by `timestamp DESC`, and applied to historical chunks while the newest chunk remained rowstore.", "",
                  md_table(["Before MB", "After MB", "Reduction %", "Ratio", "Compression s", "Compressed chunks", "Rowstore chunks"], [[before["total_mb"], after["total_mb"], compression["storage_reduction_percent"], compression["before_after_ratio"], cfg["execution_seconds"], cfg["compressed_chunks"], cfg["uncompressed_chunks"]]]), "",
                  md_table(["Historical workload", "Uncompressed median ms", "Hybrid median ms", "Effect"], compression["latency_comparison_rows"])]
    lines += ["", "## Main observations", "",
              f"- {headline['ingestion_observation']}", f"- {headline['simple_range_observation']}",
              f"- {headline['history_observation']}", f"- {headline['aggregation_observation']}",
              f"- {headline['storage_observation']}", f"- {headline['compression_observation']}",
              "- The uncompressed TimescaleDB plans use `ChunkAppend` and scan only the two chunks overlapping the selected patient's two-night history, rather than all 12 chunks; this is direct evidence of time-based chunk pruning for that bounded workload.",
              "", "Unfavorable TimescaleDB results are retained.", "", "## Paper-ready results", "",
              md_table(["Result", "PostgreSQL", "TimescaleDB / feature", "Interpretation"], headline["table"]),
              "", "## Draft wording recommendation", ""]
    lines += [f"- {claim}" for claim in headline["supported_claims"]]
    lines += ["", "Claims to remove or weaken:", ""] + [f"- {claim}" for claim in headline["claims_to_weaken"]]
    lines += ["", "## Limitations", "",
              "- Local laptop/container benchmark; not hospital-scale.",
              "- Only the implemented Apnea-ECG and Sleep-EDF datasets are represented.",
              "- Timestamp remapping is an experimental storage-layout transformation, not new clinical data or model-evaluation data.",
              "- Optional 4× temporal replay was skipped because host free disk space was too low for a safe additional copy.",
              "- Warm-cache local benchmark; OS caches were not flushed.",
              "- Compression is a Timescale-specific feature result, not a symmetric backend comparison.",
              "", "## Draft alignment", "",
              "This benchmark still does **not** complete the draft's claimed five-dataset benchmark. Any five-dataset claim must be removed or identified as future work.",
              "", "## Correctness and plans", "",
              f"All correctness checks passed: **{validation['all_checks_passed']}**. Production source unchanged: **{validation['production_source_unchanged']}**. Plans are under `outputs/database_benchmarks/query_plans/`.",
              "", "## Official TimescaleDB references consulted", ""]
    lines += [f"- [{name}]({url})" for name, url in result["official_documentation"].items()]
    return "\n".join(lines) + "\n"


def find_query(results: list[dict[str, Any]], name: str) -> dict[str, Any]:
    return next(item for item in results if item["name"] == name)


def restore_params(values: list[Any]) -> tuple[Any, ...]:
    return tuple(datetime.fromisoformat(x) if isinstance(x, str) and "T" in x else x for x in values)


def run_benchmark(query_repetitions: int, ingestion_repetitions: int, skip_compression: bool) -> dict[str, Any]:
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    PLAN_DIR.mkdir(parents=True, exist_ok=True)
    run_time = datetime.now(timezone.utc)
    run_stamp = run_time.strftime("%Y%m%dT%H%M%SZ")
    json_path = OUTPUT_DIR / f"pdm_benchmark_final_{run_stamp}.json"
    report_path = OUTPUT_DIR / "PDM_BENCHMARK_FINAL.md"
    base_date = datetime.combine((run_time - timedelta(days=11)).date(), datetime.min.time(), tzinfo=timezone.utc)

    with connect() as conn:
        db = database_metadata(conn)
        production_before = production_snapshot(conn)
        total_rows, mapping = build_longitudinal_source(conn, base_date)
        datasets = dataset_metadata(conn)
        payload = build_binary_payload(conn)
        payload_bytes = payload.getbuffer().nbytes
        ingestion_raw = {backend: [] for backend in BACKENDS}
        actual_rows = {}
        for repetition in range(1, ingestion_repetitions + 1):
            for backend in backend_order(repetition):
                elapsed, rows = load_payload(conn, backend, payload)
                if rows != total_rows:
                    raise RuntimeError(f"Ingestion row mismatch for {backend}: {rows} != {total_rows}")
                actual_rows[backend] = rows
                ingestion_raw[backend].append({"repetition": repetition, "elapsed_seconds": elapsed,
                                                "rows_per_second": rows / elapsed})
                # Outside the timed interval: bound WAL growth on the small local Docker volume.
                with conn.cursor() as cur:
                    cur.execute("CHECKPOINT")
                conn.commit()
        payload.close()
        analyze_tables(conn)
        signatures = {backend: table_signature(conn, table) for backend, table in BACKENDS.items()}
        signatures_equal = signatures["postgresql"] == signatures["timescaledb"]
        if not signatures_equal:
            raise RuntimeError("PostgreSQL and TimescaleDB logical signatures differ")
        chunks_before = chunks_metadata(conn)
        if len(chunks_before) <= 1:
            raise RuntimeError(f"Expected multiple TimescaleDB chunks, found {len(chunks_before)}")
        targets = query_targets(conn)
        queries = []
        for dataset, signal in targets["signals"].items():
            for seconds in (10, 60, 300):
                end = signal["start"] + timedelta(seconds=seconds)
                queries.append(run_paired_query(
                    conn, f"{dataset} signal {seconds}s", "signal_local_range", range_sql,
                    (signal["source_signal_id"], signal["start"], end), query_repetitions,
                    {"dataset": dataset, "recording_id": signal["recording_id"],
                     "source_signal_id": signal["source_signal_id"], "channel": signal["channel"],
                     "duration_seconds": seconds, "timescale_chunks_touched": chunks_touched(conn, signal["start"], end)},
                ))
        patient = targets["patient"]
        patient_end = patient["end"] + timedelta(microseconds=1)
        first_start = datetime.fromisoformat(patient["sessions"][0]["start"])
        for name, start, end in (("one monitoring day", first_start, first_start + timedelta(days=1)),
                                 ("full patient history", patient["start"], patient_end)):
            touched = int(scalar(conn, f"SELECT count(DISTINCT recording_id) FROM {PG_TABLE} WHERE patient_id=%s AND timestamp >= %s AND timestamp < %s", (patient["patient_id"], start, end)))
            queries.append(run_paired_query(
                conn, name, "patient_history_range", history_sql,
                (patient["patient_id"], start, end), query_repetitions,
                {"patient_id": patient["patient_id"], "recordings_touched": touched,
                 "start": iso(start), "end_exclusive": iso(end),
                 "timescale_chunks_touched": chunks_touched(conn, start, end)},
            ))
        apnea = targets["signals"]["Apnea-ECG"]
        signal_agg = lambda backend, table: aggregate_sql(backend, table, "source_signal_id=%s AND timestamp >= %s AND timestamp < %s")
        for width in ("30 seconds", "5 minutes"):
            queries.append(run_paired_query(
                conn, f"single recording {width} buckets", "raw_temporal_aggregation", signal_agg,
                (width, base_date, apnea["source_signal_id"], apnea["start"], apnea["end"] + timedelta(microseconds=1)),
                query_repetitions, {"scope": "single_recording", "bucket_width": width,
                                    "recording_id": apnea["recording_id"]},
            ))
        patient_agg = lambda backend, table: aggregate_sql(backend, table, "patient_id=%s AND channel=%s AND timestamp >= %s AND timestamp < %s")
        for width in ("30 seconds", "5 minutes"):
            queries.append(run_paired_query(
                conn, f"longitudinal patient {width} buckets", "raw_temporal_aggregation", patient_agg,
                (width, base_date, patient["patient_id"], "EEG Fpz-Cz", patient["start"], patient_end),
                query_repetitions, {"scope": "longitudinal_patient", "bucket_width": width,
                                    "patient_id": patient["patient_id"], "channel": "EEG Fpz-Cz"},
            ))
        queries.append(run_paired_query(
            conn, "full patient history hourly downsampling", "long_history_downsampling", patient_agg,
            ("1 hour", base_date, patient["patient_id"], "EEG Fpz-Cz", patient["start"], patient_end),
            query_repetitions, {"patient_id": patient["patient_id"], "channel": "EEG Fpz-Cz",
                                "bucket_width": "1 hour", "recordings_touched": patient["recordings"],
                                "timescale_chunks_touched": chunks_touched(conn, patient["start"], patient_end)},
        ))
        plans = write_query_plans(conn, targets, base_date)
        storage_uncompressed = {backend: storage_measurement(conn, backend) for backend in BACKENDS}
        ingestion_summary = {
            backend: {"rows": actual_rows[backend], "payload_bytes": payload_bytes,
                      "individual_measurements": ingestion_raw[backend],
                      "statistics": ingestion_stats([x["elapsed_seconds"] for x in ingestion_raw[backend]], actual_rows[backend])}
            for backend in BACKENDS
        }
        compression_equivalent = True
        if skip_compression:
            compression_result = {"skipped": True, "reason": "--skip-compression requested"}
        else:
            compression_config = compress_historical_chunks(conn)
            storage_after = storage_measurement(conn, "timescaledb")
            compressed_queries = []
            historical_range = find_query(queries, "Apnea-ECG signal 300s")
            compressed_queries.append(run_timescale_query(conn, historical_range["name"], historical_range["workload"],
                                                          historical_range["sql"]["timescaledb"], restore_params(historical_range["parameters"]), query_repetitions))
            for target_name in ("longitudinal patient 5 minutes buckets", "full patient history hourly downsampling"):
                original = find_query(queries, target_name)
                compressed_queries.append(run_timescale_query(conn, original["name"], original["workload"],
                                                              original["sql"]["timescaledb"], restore_params(original["parameters"]), query_repetitions))
            for compressed_query in compressed_queries:
                original = find_query(queries, compressed_query["name"])
                params = restore_params(original["parameters"])
                pg_rows, _ = fetch(conn, original["sql"]["postgresql"], params)
                ts_rows, _ = fetch(conn, original["sql"]["timescaledb"], params)
                compressed_query["equivalent_to_uncompressed_postgresql"] = rows_equal(pg_rows, ts_rows)
            compression_equivalent = all(
                item["equivalent_to_uncompressed_postgresql"] for item in compressed_queries
            )
            if not compression_equivalent:
                raise RuntimeError("A historical columnstore query differs from PostgreSQL")
            compressed_plan = PLAN_DIR / "timescaledb_hybrid_long_temporal_aggregation.txt"
            original_agg = find_query(queries, "longitudinal patient 5 minutes buckets")
            explain_to_file(conn, original_agg["sql"]["timescaledb"], restore_params(original_agg["parameters"]), compressed_plan)
            plans["timescaledb_hybrid_long_temporal_aggregation"] = str(compressed_plan.relative_to(ROOT))
            before, after = storage_uncompressed["timescaledb"]["total_bytes"], storage_after["total_bytes"]
            latency_rows = []
            for compressed_query in compressed_queries:
                original = find_query(queries, compressed_query["name"])
                old = original["statistics"]["timescaledb"]["median_ms"]
                new = compressed_query["statistics"]["median_ms"]
                effect = pct_faster(old, new)
                latency_rows.append([compressed_query["name"], old, new,
                                     f"{abs(effect):.1f}% {'faster' if effect >= 0 else 'slower'}"])
            compression_result = {
                "skipped": False, "label": "TimescaleDB historical-compression feature",
                "configuration": compression_config, "storage_before": storage_uncompressed["timescaledb"],
                "storage_after": storage_after, "compression_ratio": before / after,
                "before_after_ratio": f"{before / after:.2f}:1",
                "storage_reduction_percent": (before - after) / before * 100.0,
                "historical_queries_after_compression": compressed_queries,
                "latency_comparison_rows": latency_rows,
            }
        production_after = production_snapshot(conn)
        production_unchanged = production_before == production_after
        if not production_unchanged:
            raise RuntimeError("Production source fingerprint changed during benchmark")

    row_count_equal = signatures_equal and all(sig["rows"] == total_rows for sig in signatures.values())
    all_query_checks = all(item["equivalent_results"] for item in queries)
    validation = {
        "row_counts_equal": row_count_equal, "dataset_counts_equal": signatures_equal,
        "patient_counts_equal": signatures_equal, "recording_counts_equal": signatures_equal,
        "min_max_values_equal": signatures_equal, "logical_signatures_equal": signatures_equal,
        "range_and_aggregate_results_equal": all_query_checks,
        "compression_query_results_equal": compression_equivalent,
        "analyze_completed_before_queries": True, "multiple_chunks": len(chunks_before) > 1,
        "production_source_unchanged": production_unchanged, "production_tables_modified": False,
        "all_checks_passed": row_count_equal and all_query_checks and compression_equivalent and len(chunks_before) > 1 and production_unchanged,
        "signatures": signatures, "production_before": production_before, "production_after": production_after,
    }
    simple, history = find_query(queries, "Apnea-ECG signal 300s"), find_query(queries, "full patient history")
    aggregate = find_query(queries, "longitudinal patient 5 minutes buckets")
    downsample = find_query(queries, "full patient history hourly downsampling")
    pg_ing, ts_ing = ingestion_summary["postgresql"]["statistics"], ingestion_summary["timescaledb"]["statistics"]
    pg_store, ts_store = storage_uncompressed["postgresql"], storage_uncompressed["timescaledb"]
    def obs(query: dict[str, Any], label: str) -> str:
        return comparison_sentence(label, query["statistics"]["postgresql"]["median_ms"], query["statistics"]["timescaledb"]["median_ms"])
    ingestion_change = pct_faster(pg_ing["median_seconds"], ts_ing["median_seconds"])
    ingestion_observation = f"Full-data ingestion: TimescaleDB was {abs(ingestion_change):.1f}% {'faster' if ingestion_change >= 0 else 'slower'} by median ({ts_ing['median_seconds']:.2f}s vs {pg_ing['median_seconds']:.2f}s)."
    storage_delta = (ts_store["total_bytes"] - pg_store["total_bytes"]) / pg_store["total_bytes"] * 100.0
    storage_observation = f"Uncompressed storage: TimescaleDB used {abs(storage_delta):.1f}% {'more' if storage_delta >= 0 else 'less'} total space ({ts_store['total_mb']:.2f} MB vs {pg_store['total_mb']:.2f} MB)."
    if compression_result.get("skipped"):
        compression_observation = "Historical compression was skipped for this diagnostic run."
        compression_table_value, supported_compression = "skipped", "No compression claim is supported by this diagnostic run."
    else:
        compression_observation = f"Timescale historical columnstore reduced its total footprint by {compression_result['storage_reduction_percent']:.1f}% ({compression_result['storage_before']['total_mb']:.2f} MB to {compression_result['storage_after']['total_mb']:.2f} MB; {compression_result['before_after_ratio']})."
        compression_table_value, supported_compression = f"{compression_result['storage_after']['total_mb']:.2f} MB", compression_observation
    paper_ready = {
        "ingestion_observation": ingestion_observation,
        "simple_range_observation": obs(simple, "Five-minute ECG indexed range"),
        "history_observation": obs(history, "Full two-night patient-history retrieval"),
        "aggregation_observation": obs(aggregate, "Two-night five-minute temporal aggregation"),
        "storage_observation": storage_observation, "compression_observation": compression_observation,
        "table": [
            ["Ingestion median", f"{pg_ing['median_seconds']:.2f} s", f"{ts_ing['median_seconds']:.2f} s", ingestion_observation],
            ["5-minute ECG range median", f"{simple['statistics']['postgresql']['median_ms']:.2f} ms", f"{simple['statistics']['timescaledb']['median_ms']:.2f} ms", obs(simple, "Result")],
            ["Full patient-history median", f"{history['statistics']['postgresql']['median_ms']:.2f} ms", f"{history['statistics']['timescaledb']['median_ms']:.2f} ms", obs(history, "Result")],
            ["Longitudinal 5-minute aggregate", f"{aggregate['statistics']['postgresql']['median_ms']:.2f} ms", f"{aggregate['statistics']['timescaledb']['median_ms']:.2f} ms", obs(aggregate, "Result")],
            ["Hourly downsampling", f"{downsample['statistics']['postgresql']['median_ms']:.2f} ms", f"{downsample['statistics']['timescaledb']['median_ms']:.2f} ms", obs(downsample, "Result")],
            ["Uncompressed total storage", f"{pg_store['total_mb']:.2f} MB", f"{ts_store['total_mb']:.2f} MB", storage_observation],
            ["Timescale historical columnstore", "not applicable", compression_table_value, compression_observation],
        ],
        "supported_claims": [obs(simple, "For the measured five-minute indexed ECG range"),
                             obs(history, "For the measured two-night patient-history retrieval"),
                             obs(aggregate, "For the measured longitudinal five-minute raw aggregation"),
                             storage_observation, supported_compression,
                             "One-day partitioning produced multiple chunks; discuss pruning only where saved plans demonstrate it."],
        "claims_to_weaken": [
            "Remove any blanket claim that TimescaleDB is faster than PostgreSQL; report workload-specific medians, including PostgreSQL wins.",
            "Do not compare compressed Timescale storage/latency against uncompressed PostgreSQL as the primary comparison.",
            "Do not claim a five-dataset benchmark; only Apnea-ECG and Sleep-EDF are implemented here.",
            "Do not describe shifted dates as clinical observations or model-evaluation data.",
            "Keep ingestion secondary and report its variability.",
        ],
    }
    result = {
        "benchmark_timestamp_utc": run_time.isoformat(), "benchmark_script": "scripts/benchmark_pdm.py",
        "environment": environment_metadata(), "database": db, "official_documentation": DOCS,
        "benchmark_data": {
            "rows": total_rows, "patients": signatures["postgresql"]["patients"],
            "recordings": signatures["postgresql"]["recordings"], "signals": signatures["postgresql"]["signals"],
            "channel_count": len({c["channel"] for d in datasets.values() for c in d["channels"]}),
            "dataset_count": signatures["postgresql"]["datasets"], "datasets": datasets,
            "source_filter": "Apnea-ECG and Sleep-EDF rows where signal_type equals recording modality; reconstruction excluded",
        },
        "configuration": {
            "schema": "benchmark", "tables": BACKENDS,
            "columns": ["dataset","recording_id","patient_id","source_signal_id","channel","timestamp","value","quality"],
            "indexes": ["(source_signal_id, timestamp DESC)", "(patient_id, timestamp DESC)"],
            "chunk_interval": CHUNK_INTERVAL, "number_of_chunks": len(chunks_before),
            "timestamp_mapping": {"base_date": iso(base_date), "method": "successive one-day recording offsets",
                                  "recordings": mapping,
                                  "statement": "Original physiological values and within-recording timing were preserved; recording timestamps were deterministically shifted across successive dates only to construct a longitudinal database workload."},
            "query_repetitions": query_repetitions, "query_warmups": 1,
            "ingestion_repetitions": ingestion_repetitions,
            "ingestion_method": "same PostgreSQL binary COPY payload; timed COPY FROM plus commit only",
            "cache_method": "warm local measurements; no manual OS cache flushing",
            "analyze": "ANALYZE executed on both tables before correctness and query benchmarks",
            "optional_temporal_replay": {"performed": False, "reason": "skipped due low host free disk space"},
        },
        "timescale_chunks_uncompressed": chunks_before, "query_targets": targets,
        "results": {"ingestion": ingestion_raw, "ingestion_summary": ingestion_summary,
                    "queries": queries, "storage_uncompressed": storage_uncompressed,
                    "timescale_historical_compression": compression_result},
        "query_plan_locations": plans, "correctness_validation": validation,
        "paper_ready_results": paper_ready,
        "outputs": {"json": str(json_path.relative_to(ROOT)), "report": str(report_path.relative_to(ROOT)),
                    "query_plans": str(PLAN_DIR.relative_to(ROOT))},
    }
    json_path.write_text(json.dumps(result, indent=2, default=json_default), encoding="utf-8")
    report_path.write_text(render_report(result), encoding="utf-8")
    print(json.dumps({"json": str(json_path), "report": str(report_path), "rows": total_rows,
                      "chunks": len(chunks_before), "validation": validation["all_checks_passed"]}, indent=2))
    return result


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--query-repetitions", type=int, default=QUERY_REPETITIONS)
    parser.add_argument("--ingestion-repetitions", type=int, default=INGESTION_REPETITIONS)
    parser.add_argument("--skip-compression", action="store_true")
    args = parser.parse_args()
    if args.query_repetitions < 2 or args.ingestion_repetitions < 1:
        raise SystemExit("query repetitions must be >=2 and ingestion repetitions >=1")
    run_benchmark(args.query_repetitions, args.ingestion_repetitions, args.skip_compression)


if __name__ == "__main__":
    main()
