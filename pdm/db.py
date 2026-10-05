"""PostgreSQL/TimescaleDB connectivity and non-destructive health helpers."""

import os
from contextlib import contextmanager

import psycopg2
from psycopg2 import sql


REQUIRED_PREDICTION_SERIES_COLUMNS = (
    "timestamp",
    "patient_id",
    "recording_id",
    "window_id",
    "prediction_id",
    "model_id",
    "reconstruction_error",
    "anomaly_threshold",
    "anomaly_margin",
    "normalized_margin",
    "is_anomaly",
    "predicted_class",
)

TEMPORAL_TABLES = (
    "signal_data",
    "signal_feature_data",
    "attention_data",
    "prediction_series",
)

MIGRATION_VERSION = "001_fusion_alignment"
CONTINUOUS_AGGREGATE_NAME = "pdm_prediction_5m"


def connection_kwargs():
    return {
        # 127.0.0.1 avoids Windows/Docker IPv6 localhost resolution quirks.
        "host": os.getenv("PDM_DB_HOST", "127.0.0.1"),
        "port": int(os.getenv("PDM_DB_PORT", "5433")),
        "dbname": os.getenv("PDM_DB_NAME", "pdm"),
        "user": os.getenv("PDM_DB_USER", "postgres"),
        "password": os.getenv("PDM_DB_PASSWORD", "123456"),
    }


@contextmanager
def connect():
    conn = psycopg2.connect(**connection_kwargs())
    try:
        yield conn
    finally:
        conn.close()


def _relation_exists(cur, schema: str, relation: str) -> bool:
    """Return whether a relation exists without failing the current transaction."""

    cur.execute("SELECT to_regclass(%s)", (f"{schema}.{relation}",))
    return cur.fetchone()[0] is not None


def _count_relation(cur, schema: str, relation: str):
    """Count a known relation, returning ``None`` when it is not present."""

    if not _relation_exists(cur, schema, relation):
        return None
    cur.execute(
        sql.SQL("SELECT COUNT(*) FROM {}.{}").format(
            sql.Identifier(schema), sql.Identifier(relation)
        )
    )
    return int(cur.fetchone()[0])


def _unavailable_health(error: Exception) -> dict:
    """Build a stable report when PostgreSQL cannot be reached."""

    required = {column: False for column in REQUIRED_PREDICTION_SERIES_COLUMNS}
    hypertables = {table: False for table in TEMPORAL_TABLES}
    return {
        "status": "unavailable",
        "database_available": False,
        "error": str(error),
        "postgres_version": None,
        "timescaledb_version": None,
        "migration_version": None,
        "migration_applied": False,
        "required_columns": required,
        "required_columns_present": False,
        "hypertables": [(table, False) for table in TEMPORAL_TABLES],
        "hypertable_status": hypertables,
        "continuous_aggregate_exists": False,
        "continuous_aggregate": {
            "name": CONTINUOUS_AGGREGATE_NAME,
            "exists": False,
        },
        "materialization_state": {
            "status": "unknown",
            "exists": False,
            "materialized": False,
            "refreshed": False,
        },
        "row_counts": {},
    }


def health_check() -> dict:
    """Return a read-only database/migration report.

    The command is deliberately useful before and after Phase 6 population.  A
    refused connection is represented as an ``unavailable`` report rather than
    being raised by the CLI, so local development and CI can inspect the state
    without requiring a running Docker stack.
    """

    try:
        with connect() as conn:
            conn.autocommit = True
            with conn.cursor() as cur:
                cur.execute("SELECT version()")
                postgres_version = cur.fetchone()[0]
                cur.execute(
                    "SELECT extversion FROM pg_extension WHERE extname = 'timescaledb'"
                )
                timescaledb_row = cur.fetchone()
                timescaledb_version = timescaledb_row[0] if timescaledb_row else None

                columns = set()
                if _relation_exists(cur, "public", "prediction_series"):
                    cur.execute(
                        """
                        SELECT column_name
                        FROM information_schema.columns
                        WHERE table_schema = 'public'
                          AND table_name = 'prediction_series'
                        """
                    )
                    columns = {row[0] for row in cur.fetchall()}
                required_columns = {
                    column: column in columns
                    for column in REQUIRED_PREDICTION_SERIES_COLUMNS
                }

                hypertable_status = {table: False for table in TEMPORAL_TABLES}
                if _relation_exists(cur, "timescaledb_information", "hypertables"):
                    cur.execute(
                        """
                        SELECT hypertable_name::text
                        FROM timescaledb_information.hypertables
                        WHERE hypertable_schema = 'public'
                          AND hypertable_name = ANY(%s)
                        """,
                        (list(TEMPORAL_TABLES),),
                    )
                    for (name,) in cur.fetchall():
                        hypertable_status[name] = True

                migration_version = None
                migration_table_exists = _relation_exists(
                    cur, "public", "pdm_schema_migrations"
                )
                if migration_table_exists:
                    cur.execute(
                        """
                        SELECT version
                        FROM pdm_schema_migrations
                        ORDER BY applied_at DESC, version DESC
                        LIMIT 1
                        """
                    )
                    migration_row = cur.fetchone()
                    migration_version = migration_row[0] if migration_row else None

                cagg = {
                    "name": CONTINUOUS_AGGREGATE_NAME,
                    "exists": False,
                }
                materialization = {
                    "status": "missing",
                    "exists": False,
                    "materialized": False,
                    "refreshed": False,
                }
                if _relation_exists(
                    cur, "timescaledb_information", "continuous_aggregates"
                ):
                    cur.execute(
                        """
                        SELECT view_schema::text,
                               view_name::text,
                               materialized_only,
                               materialization_hypertable_schema::text,
                               materialization_hypertable_name::text,
                               view_definition
                        FROM timescaledb_information.continuous_aggregates
                        WHERE view_schema = 'public'
                          AND view_name = %s
                        """,
                        (CONTINUOUS_AGGREGATE_NAME,),
                    )
                    cagg_row = cur.fetchone()
                    if cagg_row:
                        (
                            view_schema,
                            view_name,
                            materialized_only,
                            materialization_schema,
                            materialization_name,
                            view_definition,
                        ) = cagg_row
                        cagg.update(
                            {
                                "exists": True,
                                "view_schema": view_schema,
                                "view_name": view_name,
                                "materialized_only": materialized_only,
                                "view_uses_time_bucket": "time_bucket" in view_definition,
                                "materialization_hypertable": (
                                    f"{materialization_schema}.{materialization_name}"
                                ),
                            }
                        )
                        materialized_row_count = _count_relation(
                            cur, materialization_schema, materialization_name
                        )
                        aggregate_row_count = _count_relation(
                            cur, "public", CONTINUOUS_AGGREGATE_NAME
                        )
                        source_row_count = _count_relation(
                            cur, "public", "prediction_series"
                        )
                        materialization.update(
                            {
                                "exists": materialized_row_count is not None,
                                "materialized": materialized_row_count is not None,
                                "materialized_row_count": materialized_row_count,
                                "aggregate_row_count": aggregate_row_count,
                                "source_row_count": source_row_count,
                            }
                        )
                        if materialized_row_count is None:
                            materialization["status"] = "unknown"
                        elif materialized_row_count > 0:
                            materialization["status"] = "materialized"
                            materialization["refreshed"] = True
                        elif source_row_count:
                            materialization["status"] = "not_refreshed"
                        else:
                            materialization["status"] = "empty"
                        cagg["materialized_row_count"] = materialized_row_count
                        cagg["row_count"] = aggregate_row_count

                row_counts = {
                    "prediction_series": _count_relation(
                        cur, "public", "prediction_series"
                    ),
                    "pdm_prediction_5m": cagg.get("row_count"),
                    "pdm_prediction_5m_materialized": materialization.get(
                        "materialized_row_count"
                    ),
                    "prediction": _count_relation(cur, "public", "prediction"),
                    "analysis_window": _count_relation(
                        cur, "public", "analysis_window"
                    ),
                }

        return {
            "status": "ok",
            "database_available": True,
            "error": None,
            "postgres_version": postgres_version,
            "timescaledb_version": timescaledb_version,
            "migration_version": migration_version,
            "migration_applied": migration_version == MIGRATION_VERSION,
            "migration_table_exists": migration_table_exists,
            "required_columns": required_columns,
            "required_columns_present": all(required_columns.values()),
            "hypertables": [
                (table, hypertable_status[table]) for table in sorted(hypertable_status)
            ],
            "hypertable_status": hypertable_status,
            "continuous_aggregate_exists": cagg["exists"],
            "continuous_aggregate": cagg,
            "materialization_state": materialization,
            "row_counts": row_counts,
        }
    except Exception as exc:
        return _unavailable_health(exc)


if __name__ == "__main__":
    import json
    try:
        print(json.dumps(health_check(), indent=2, default=str))
    except Exception as exc:
        raise SystemExit(f"Database health check failed: {exc}")
