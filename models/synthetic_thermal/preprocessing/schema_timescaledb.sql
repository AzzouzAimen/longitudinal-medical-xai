-- =============================================================================
-- schema_timescaledb.sql
--  – T5 : Dual DB Ingestion
-- Topic M6 : Synthetic Thermal Time-Series
--
-- TimescaleDB schema
-- Extension timescaledb must already be loaded (CREATE EXTENSION timescaledb)
-- Run once before ingest_tsdb.py
-- =============================================================================

-- --------------------------------------------------------------------
-- 0. Extensions
-- --------------------------------------------------------------------
CREATE EXTENSION IF NOT EXISTS timescaledb;

-- --------------------------------------------------------------------
-- 1. subjects  (same as PostgreSQL — no time-series specifics)
-- --------------------------------------------------------------------
CREATE TABLE IF NOT EXISTS subjects (
    patient_id          SMALLINT        PRIMARY KEY,
    norm_mean_left      DOUBLE PRECISION NOT NULL,
    norm_std_left       DOUBLE PRECISION NOT NULL,
    norm_mean_right     DOUBLE PRECISION NOT NULL,
    norm_std_right      DOUBLE PRECISION NOT NULL,
    rows_input          INTEGER,
    rows_output         INTEGER,
    n_windows           INTEGER,
    n_anomaly_windows   INTEGER,
    n_segments          INTEGER,
    n_interpolated_rows INTEGER,
    created_at          TIMESTAMPTZ     DEFAULT NOW()
);

-- --------------------------------------------------------------------
-- 2. thermal_readings  (hypertable — replaces signals + recordings)
--    Partition by time + patient for efficient range scans.
-- --------------------------------------------------------------------
CREATE TABLE IF NOT EXISTS thermal_readings (
    timestamp               TIMESTAMPTZ     NOT NULL,         -- partition key (required)
    patient_id              SMALLINT        NOT NULL,
    left_temperature        REAL            NOT NULL,
    right_temperature       REAL            NOT NULL,
    left_temperature_norm   REAL            NOT NULL,
    right_temperature_norm  REAL            NOT NULL,
    temp_asymmetry          REAL            NOT NULL,
    anomaly_label           SMALLINT        NOT NULL,
    anomaly_type            VARCHAR(8)      NOT NULL,
    is_interpolated         BOOLEAN         NOT NULL DEFAULT FALSE,
    segment_id              INTEGER         NOT NULL DEFAULT 0,
    session_second          INTEGER         NOT NULL
);

-- Convert to hypertable
-- chunk_time_interval = 6h → each chunk ≈ 6h × 20 patients × 1 Hz = 432 000 rows
-- space_partitions = 4 (by patient_id) → parallel I/O on multi-core hosts
SELECT create_hypertable(
    'thermal_readings',
    'timestamp',
    partitioning_column   => 'patient_id',
    number_partitions     => 4,
    chunk_time_interval   => INTERVAL '6 hours',
    if_not_exists         => TRUE
);

COMMENT ON TABLE thermal_readings IS
    'Hypertable: 5 184 000 rows, partitioned by time (6h chunks) + patient_id (4 space partitions). '
    'is_interpolated / temp_asymmetry / segment_id required by .';

-- --------------------------------------------------------------------
-- 3. windows_tsdb  (same metadata as PostgreSQL, kept separate to avoid
--    hypertable overhead on a small dimension table)
-- --------------------------------------------------------------------
CREATE TABLE IF NOT EXISTS windows_tsdb (
    window_id           INTEGER         NOT NULL,
    patient_id          SMALLINT        NOT NULL REFERENCES subjects(patient_id),
    segment_id          INTEGER         NOT NULL,
    window_index        INTEGER         NOT NULL,
    window_start        TIMESTAMPTZ     NOT NULL,
    window_end          TIMESTAMPTZ     NOT NULL,
    label               SMALLINT        NOT NULL,
    anomaly_ratio       REAL            NOT NULL,
    is_interpolated     BOOLEAN         NOT NULL DEFAULT FALSE,
    PRIMARY KEY (patient_id, window_id)
);

-- --------------------------------------------------------------------
-- 4. Indexes (created AFTER bulk load)
-- --------------------------------------------------------------------
-- BM-R1 / BM-R2 : range scans
CREATE INDEX IF NOT EXISTS idx_tr_patient_ts
    ON thermal_readings (patient_id, timestamp DESC);

-- BM-A1 : time_bucket aggregation
CREATE INDEX IF NOT EXISTS idx_tr_ts
    ON thermal_readings (timestamp DESC);

-- BM-A2 : anomaly filter
CREATE INDEX IF NOT EXISTS idx_tr_anomaly
    ON thermal_readings (anomaly_label)
    WHERE anomaly_label = 1;

-- --------------------------------------------------------------------
-- 5. Compression policy
--    Compress chunks older than 12 hours.
--    segmentby patient_id → all data for one patient stays in one segment.
--    orderby timestamp DESC → optimizes range scans.
-- --------------------------------------------------------------------
ALTER TABLE thermal_readings SET (
    timescaledb.compress,
    timescaledb.compress_segmentby = 'patient_id',
    timescaledb.compress_orderby   = 'timestamp DESC'
);

-- Apply automatic compression: chunks older than 12h are compressed
SELECT add_compression_policy(
    'thermal_readings',
    INTERVAL '12 hours',
    if_not_exists => TRUE
);

-- --------------------------------------------------------------------
-- 6. Continuous aggregate for BM-A1 (avg temp per minute)
--    Pre-materializes the aggregation → much faster than on-the-fly grouping.
-- --------------------------------------------------------------------
CREATE MATERIALIZED VIEW IF NOT EXISTS avg_temp_per_minute
WITH (timescaledb.continuous) AS
SELECT
    time_bucket('1 minute', timestamp) AS bucket,
    patient_id,
    AVG(left_temperature)       AS avg_left,
    AVG(right_temperature)      AS avg_right,
    AVG(temp_asymmetry)         AS avg_asymmetry,
    COUNT(*)                    AS n_samples
FROM thermal_readings
GROUP BY bucket, patient_id
WITH NO DATA;

-- Refresh policy: materialize every hour, covering last 2h
SELECT add_continuous_aggregate_policy(
    'avg_temp_per_minute',
    start_offset => INTERVAL '2 hours',
    end_offset   => INTERVAL '1 minute',
    schedule_interval => INTERVAL '1 hour',
    if_not_exists => TRUE
);

-- --------------------------------------------------------------------
-- 7. Verify hypertable
-- --------------------------------------------------------------------
SELECT
    hypertable_name,
    num_chunks,
    pg_size_pretty(hypertable_size(hypertable_name::regclass)) AS total_size
FROM timescaledb_information.hypertables;