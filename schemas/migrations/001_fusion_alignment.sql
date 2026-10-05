-- Phase 5: align prediction_series with the fusion detector evidence contract.
-- This migration is intentionally data-preserving and safe to run more than once.

BEGIN;

CREATE EXTENSION IF NOT EXISTS timescaledb;

CREATE TABLE IF NOT EXISTS pdm_schema_migrations (
    version VARCHAR(128) PRIMARY KEY,
    applied_at TIMESTAMPTZ NOT NULL DEFAULT CURRENT_TIMESTAMP
);

ALTER TABLE prediction_series
    ADD COLUMN IF NOT EXISTS patient_id BIGINT,
    ADD COLUMN IF NOT EXISTS recording_id VARCHAR(64),
    ADD COLUMN IF NOT EXISTS model_id BIGINT,
    ADD COLUMN IF NOT EXISTS reconstruction_error DOUBLE PRECISION,
    ADD COLUMN IF NOT EXISTS anomaly_threshold DOUBLE PRECISION,
    ADD COLUMN IF NOT EXISTS anomaly_margin DOUBLE PRECISION,
    ADD COLUMN IF NOT EXISTS normalized_margin DOUBLE PRECISION,
    ADD COLUMN IF NOT EXISTS is_anomaly BOOLEAN;

-- Keep the legacy field available for old clients.  It is deliberately not
-- used as a margin or populated by this migration.
ALTER TABLE prediction_series
    ADD COLUMN IF NOT EXISTS probability DOUBLE PRECISION;

-- Recover relational provenance for rows written by the original schema.  The
-- new evidence fields remain nullable because old rows cannot have margins.
UPDATE prediction_series AS ps
SET recording_id = aw.recording_id,
    patient_id = r.patient_id
FROM analysis_window AS aw
JOIN recording AS r ON r.recording_id = aw.recording_id
WHERE ps.window_id = aw.window_id
  AND (ps.recording_id IS NULL OR ps.patient_id IS NULL);

UPDATE prediction_series AS ps
SET model_id = p.model_id,
    reconstruction_error = COALESCE(ps.reconstruction_error, p.reconstruction_error),
    anomaly_threshold = COALESCE(ps.anomaly_threshold, p.anomaly_threshold),
    is_anomaly = COALESCE(ps.is_anomaly, p.is_anomaly)
FROM prediction AS p
WHERE ps.prediction_id = p.prediction_id
  AND (ps.model_id IS NULL OR ps.reconstruction_error IS NULL
       OR ps.anomaly_threshold IS NULL OR ps.is_anomaly IS NULL);

CREATE INDEX IF NOT EXISTS idx_prediction_series_patient_time
    ON prediction_series (patient_id, timestamp DESC);
CREATE INDEX IF NOT EXISTS idx_prediction_series_model_time
    ON prediction_series (model_id, timestamp DESC);
CREATE INDEX IF NOT EXISTS idx_prediction_series_recording_time
    ON prediction_series (recording_id, timestamp DESC);

-- These provenance links are nullable to preserve legacy rows that cannot be
-- backfilled, while still validating every new non-null reference.
DO $$
BEGIN
    IF NOT EXISTS (
        SELECT 1 FROM pg_constraint c
        JOIN pg_class t ON t.oid = c.conrelid
        WHERE t.relname = 'prediction_series'
          AND c.contype = 'f'
          AND pg_get_constraintdef(c.oid) LIKE '%(patient_id)%'
    ) THEN
        ALTER TABLE prediction_series
            ADD CONSTRAINT prediction_series_patient_fk
            FOREIGN KEY (patient_id) REFERENCES patient(patient_id) NOT VALID;
    END IF;

    IF NOT EXISTS (
        SELECT 1 FROM pg_constraint c
        JOIN pg_class t ON t.oid = c.conrelid
        WHERE t.relname = 'prediction_series'
          AND c.contype = 'f'
          AND pg_get_constraintdef(c.oid) LIKE '%(recording_id)%'
    ) THEN
        ALTER TABLE prediction_series
            ADD CONSTRAINT prediction_series_recording_fk
            FOREIGN KEY (recording_id) REFERENCES recording(recording_id) NOT VALID;
    END IF;

    IF NOT EXISTS (
        SELECT 1 FROM pg_constraint c
        JOIN pg_class t ON t.oid = c.conrelid
        WHERE t.relname = 'prediction_series'
          AND c.contype = 'f'
          AND pg_get_constraintdef(c.oid) LIKE '%(model_id)%'
    ) THEN
        ALTER TABLE prediction_series
            ADD CONSTRAINT prediction_series_model_fk
            FOREIGN KEY (model_id) REFERENCES model(model_id) NOT VALID;
    END IF;
END
$$;

-- Fresh databases get a database-level guard.  If a pre-migration database
-- contains legacy labels (for example A/N or transition), leave those rows
-- untouched and rely on application validation until they are migrated.
DO $$
BEGIN
    IF NOT EXISTS (
        SELECT 1 FROM pg_constraint
        WHERE conrelid = 'prediction_series'::regclass
          AND conname = 'prediction_series_predicted_class_check'
    )
    AND NOT EXISTS (
        SELECT 1 FROM prediction_series
        WHERE predicted_class IS NOT NULL
          AND predicted_class NOT IN ('anomaly', 'normal')
    ) THEN
        ALTER TABLE prediction_series
            ADD CONSTRAINT prediction_series_predicted_class_check
            CHECK (predicted_class IS NULL OR predicted_class IN ('anomaly', 'normal'));
    END IF;
END
$$;

-- A five-minute continuous aggregate is the canonical historical trend source.
-- WITH NO DATA is intentional: Phase 5 does not populate experiment rows or
-- refresh historical materialization; Phase 6 performs the first full refresh.
DO $$
BEGIN
    IF to_regclass('public.pdm_prediction_5m') IS NULL THEN
        EXECUTE $view$
            CREATE MATERIALIZED VIEW pdm_prediction_5m
            WITH (timescaledb.continuous) AS
            SELECT
                time_bucket('5 minutes', timestamp) AS bucket,
                patient_id,
                model_id,
                AVG(reconstruction_error) AS mean_reconstruction_error,
                MAX(reconstruction_error) AS max_reconstruction_error,
                AVG(anomaly_margin) AS mean_anomaly_margin,
                COUNT(*) AS total_count,
                SUM(CASE WHEN is_anomaly THEN 1 ELSE 0 END) AS anomaly_count,
                SUM(CASE WHEN is_anomaly THEN 1 ELSE 0 END)::DOUBLE PRECISION
                    / NULLIF(COUNT(*), 0) AS anomaly_rate
            FROM prediction_series
            GROUP BY bucket, patient_id, model_id
            WITH NO DATA
        $view$;
    END IF;
END
$$;

INSERT INTO pdm_schema_migrations(version)
VALUES ('001_fusion_alignment')
ON CONFLICT (version) DO NOTHING;

COMMIT;
