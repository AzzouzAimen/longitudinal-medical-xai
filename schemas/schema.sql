-- JAISE Patient Digital Memory (PDM) schema
-- PostgreSQL + TimescaleDB
-- Relational PDM: patient/context/model/prediction/XAI metadata
-- TimescaleDB: high-volume temporal signal, feature, attention and prediction series

CREATE TABLE patient (
    patient_id BIGSERIAL PRIMARY KEY,
    patient_code VARCHAR(64) UNIQUE NOT NULL,
    date_of_birth DATE,
    sex VARCHAR(20),
    created_at TIMESTAMPTZ NOT NULL DEFAULT CURRENT_TIMESTAMP
);

CREATE TABLE clinical_visit (
    visit_id BIGSERIAL PRIMARY KEY,
    patient_id BIGINT NOT NULL REFERENCES patient(patient_id),
    visit_start TIMESTAMPTZ NOT NULL,
    visit_end TIMESTAMPTZ,
    visit_type VARCHAR(64),
    reason TEXT,
    clinical_notes TEXT,
    status VARCHAR(32)
);

CREATE TABLE recording (
    recording_id VARCHAR(64) PRIMARY KEY,
    patient_id BIGINT NOT NULL REFERENCES patient(patient_id),
    visit_id BIGINT REFERENCES clinical_visit(visit_id),
    source_recording_name VARCHAR(255),
    modality VARCHAR(64) NOT NULL,
    sampling_rate_hz DOUBLE PRECISION,
    length_seconds DOUBLE PRECISION,
    recording_start_time TIMESTAMPTZ,
    data_location TEXT
);

CREATE TABLE signal (
    signal_id BIGSERIAL PRIMARY KEY,
    recording_id VARCHAR(64) NOT NULL REFERENCES recording(recording_id),
    signal_type VARCHAR(64) NOT NULL,
    channel VARCHAR(128),
    unit VARCHAR(64),
    sampling_rate_hz DOUBLE PRECISION
);

-- RENAMED from "window": WINDOW is a reserved keyword in PostgreSQL
-- (used in window-function / OVER syntax). "CREATE TABLE window (...)"
-- fails with a syntax error unless every reference is double-quoted
-- everywhere forever, which is fragile. Renamed to analysis_window.
CREATE TABLE analysis_window (
    window_id BIGSERIAL PRIMARY KEY,
    recording_id VARCHAR(64) NOT NULL REFERENCES recording(recording_id),
    window_index INTEGER NOT NULL,
    start_sample BIGINT,
    end_sample BIGINT,
    start_time TIMESTAMPTZ NOT NULL,
    end_time TIMESTAMPTZ NOT NULL,
    duration_seconds DOUBLE PRECISION,
    UNIQUE(recording_id, window_index)
);

CREATE TABLE window_signal (
    window_id BIGINT NOT NULL REFERENCES analysis_window(window_id) ON DELETE CASCADE,
    signal_id BIGINT NOT NULL REFERENCES signal(signal_id) ON DELETE CASCADE,
    PRIMARY KEY (window_id, signal_id)
);

CREATE TABLE model (
    model_id BIGSERIAL PRIMARY KEY,
    model_name VARCHAR(255) NOT NULL,
    model_version VARCHAR(128),
    task VARCHAR(128) NOT NULL,
    architecture VARCHAR(255),
    training_dataset VARCHAR(255),
    input_configuration TEXT,
    created_at TIMESTAMPTZ NOT NULL DEFAULT CURRENT_TIMESTAMP,
    -- ADDED: prevents two rows silently representing the same
    -- registered model version.
    UNIQUE(model_name, model_version)
);

CREATE TABLE model_modality (
    model_id BIGINT NOT NULL REFERENCES model(model_id) ON DELETE CASCADE,
    modality VARCHAR(64) NOT NULL,
    PRIMARY KEY (model_id, modality)
);

CREATE TABLE prediction (
    prediction_id BIGSERIAL PRIMARY KEY,
    window_id BIGINT NOT NULL REFERENCES analysis_window(window_id),
    model_id BIGINT NOT NULL REFERENCES model(model_id),
    prediction_time TIMESTAMPTZ NOT NULL DEFAULT CURRENT_TIMESTAMP,
    predicted_class VARCHAR(128),
    prediction_probability DOUBLE PRECISION,
    reconstruction_error DOUBLE PRECISION,
    anomaly_threshold DOUBLE PRECISION,
    is_anomaly BOOLEAN,
    severity VARCHAR(32),
    status VARCHAR(32)
);

CREATE TABLE annotation (
    annotation_id BIGSERIAL PRIMARY KEY,
    recording_id VARCHAR(64) NOT NULL REFERENCES recording(recording_id),
    window_id BIGINT REFERENCES analysis_window(window_id),
    annotation_type VARCHAR(64),
    label VARCHAR(128),
    start_time TIMESTAMPTZ,
    end_time TIMESTAMPTZ,
    source VARCHAR(128),
    confidence DOUBLE PRECISION
);

CREATE TABLE xai_explanation (
    explanation_id BIGSERIAL PRIMARY KEY,
    prediction_id BIGINT NOT NULL REFERENCES prediction(prediction_id) ON DELETE CASCADE,
    xai_method VARCHAR(128) NOT NULL,
    explanation_type VARCHAR(128),
    explanation_text TEXT,
    generated_at TIMESTAMPTZ NOT NULL DEFAULT CURRENT_TIMESTAMP
);

CREATE TABLE xai_feature (
    xai_feature_id BIGSERIAL PRIMARY KEY,
    explanation_id BIGINT NOT NULL REFERENCES xai_explanation(explanation_id) ON DELETE CASCADE,
    signal_id BIGINT REFERENCES signal(signal_id),
    feature_name VARCHAR(255) NOT NULL,
    feature_value DOUBLE PRECISION,
    attribution_value DOUBLE PRECISION,
    importance_rank INTEGER
);

CREATE TABLE xai_location (
    location_id BIGSERIAL PRIMARY KEY,
    explanation_id BIGINT NOT NULL REFERENCES xai_explanation(explanation_id) ON DELETE CASCADE,
    signal_id BIGINT REFERENCES signal(signal_id),
    channel VARCHAR(128),
    start_sample BIGINT,
    end_sample BIGINT,
    temporal_start TIMESTAMPTZ,
    temporal_end TIMESTAMPTZ,
    spatial_location VARCHAR(255),
    frequency_band VARCHAR(128)
);

CREATE TABLE xai_artifact (
    artifact_id BIGSERIAL PRIMARY KEY,
    explanation_id BIGINT NOT NULL REFERENCES xai_explanation(explanation_id) ON DELETE CASCADE,
    artifact_type VARCHAR(128),
    artifact_location TEXT NOT NULL,
    format VARCHAR(32),
    description TEXT
);

CREATE TABLE patient_baseline (
    baseline_id BIGSERIAL PRIMARY KEY,
    patient_id BIGINT NOT NULL REFERENCES patient(patient_id) ON DELETE CASCADE,
    signal_id BIGINT REFERENCES signal(signal_id),
    baseline_type VARCHAR(128) NOT NULL,
    baseline_value JSONB,
    reference_start TIMESTAMPTZ,
    reference_end TIMESTAMPTZ,
    created_at TIMESTAMPTZ NOT NULL DEFAULT CURRENT_TIMESTAMP
);

CREATE TABLE diagnosis (
    diagnosis_id BIGSERIAL PRIMARY KEY,
    patient_id BIGINT NOT NULL REFERENCES patient(patient_id) ON DELETE CASCADE,
    visit_id BIGINT REFERENCES clinical_visit(visit_id),
    diagnosis_code VARCHAR(128),
    diagnosis_name VARCHAR(255) NOT NULL,
    diagnosis_status VARCHAR(64),
    diagnosis_time TIMESTAMPTZ,
    notes TEXT
);

CREATE TABLE clinical_assessment (
    assessment_id BIGSERIAL PRIMARY KEY,
    prediction_id BIGINT NOT NULL REFERENCES prediction(prediction_id) ON DELETE CASCADE,
    clinician_id VARCHAR(128),
    -- REMOVED: final_diagnosis VARCHAR(255)
    -- This duplicated the dedicated `diagnosis` table and re-introduced
    -- the problem fixed earlier: a diagnosis scoped to one prediction
    -- instead of living on the patient's longitudinal record. A
    -- clinician's review of *this* prediction should reference the
    -- patient's actual diagnosis record instead, if relevant.
    related_diagnosis_id BIGINT REFERENCES diagnosis(diagnosis_id),
    agreement VARCHAR(32),
    comments TEXT,
    assessment_time TIMESTAMPTZ NOT NULL DEFAULT CURRENT_TIMESTAMP
);

CREATE TABLE narrative (
    narrative_id BIGSERIAL PRIMARY KEY,
    patient_id BIGINT NOT NULL REFERENCES patient(patient_id) ON DELETE CASCADE,
    window_id BIGINT REFERENCES analysis_window(window_id),
    prediction_id BIGINT REFERENCES prediction(prediction_id),
    narrative_text TEXT NOT NULL,
    llm_model_version VARCHAR(128),
    generated_at TIMESTAMPTZ NOT NULL DEFAULT CURRENT_TIMESTAMP
);

CREATE TABLE narrative_evidence (
    narrative_evidence_id BIGSERIAL PRIMARY KEY,
    narrative_id BIGINT NOT NULL REFERENCES narrative(narrative_id) ON DELETE CASCADE,
    explanation_id BIGINT REFERENCES xai_explanation(explanation_id),
    annotation_id BIGINT REFERENCES annotation(annotation_id),
    prediction_id BIGINT REFERENCES prediction(prediction_id),
    source_narrative_id BIGINT REFERENCES narrative(narrative_id),
    evidence_role VARCHAR(64),
    -- ADDED: a "narrative_evidence" row that references nothing is a
    -- silent data-quality bug (it looks like traceability but proves
    -- nothing). Require at least one real reference.
    CONSTRAINT narrative_evidence_has_reference CHECK (
        explanation_id IS NOT NULL OR
        annotation_id IS NOT NULL OR
        prediction_id IS NOT NULL OR
        source_narrative_id IS NOT NULL
    )
);

-- =========================
-- TIMESCALEDB
-- =========================

CREATE TABLE signal_data (
    timestamp TIMESTAMPTZ NOT NULL,
    signal_id BIGINT NOT NULL REFERENCES signal(signal_id),
    channel VARCHAR(128),
    value DOUBLE PRECISION NOT NULL,
    quality DOUBLE PRECISION
);

CREATE TABLE signal_feature_data (
    timestamp TIMESTAMPTZ NOT NULL,
    window_id BIGINT NOT NULL REFERENCES analysis_window(window_id),
    signal_id BIGINT REFERENCES signal(signal_id),
    feature_name VARCHAR(255) NOT NULL,
    feature_value DOUBLE PRECISION,
    quality DOUBLE PRECISION
);

CREATE TABLE attention_data (
    timestamp TIMESTAMPTZ NOT NULL,
    window_id BIGINT NOT NULL REFERENCES analysis_window(window_id),
    signal_id BIGINT REFERENCES signal(signal_id),
    attention_value DOUBLE PRECISION NOT NULL
);

CREATE TABLE prediction_series (
    timestamp TIMESTAMPTZ NOT NULL,
    prediction_id BIGINT REFERENCES prediction(prediction_id),
    window_id BIGINT NOT NULL REFERENCES analysis_window(window_id),
    predicted_class VARCHAR(128),
    probability DOUBLE PRECISION
);

-- Run after enabling TimescaleDB, for ALL FOUR hypertables below
-- (the original only documented this for signal_data — the other
-- three were left as plain tables, which loses TimescaleDB's chunking
-- and continuous-aggregate benefits for exactly the workloads the
-- paper benchmarks: attention curves and derived temporal features).
--
-- CREATE EXTENSION IF NOT EXISTS timescaledb;
-- SELECT create_hypertable('signal_data', 'timestamp');
-- SELECT create_hypertable('signal_feature_data', 'timestamp');
-- SELECT create_hypertable('attention_data', 'timestamp');
-- SELECT create_hypertable('prediction_series', 'timestamp');

CREATE INDEX idx_visit_patient ON clinical_visit(patient_id);
CREATE INDEX idx_recording_patient ON recording(patient_id);
CREATE INDEX idx_recording_visit ON recording(visit_id);
CREATE INDEX idx_signal_recording ON signal(recording_id);
CREATE INDEX idx_window_recording_time ON analysis_window(recording_id, start_time, end_time);
CREATE INDEX idx_prediction_window ON prediction(window_id);
CREATE INDEX idx_prediction_model ON prediction(model_id);
CREATE INDEX idx_prediction_time ON prediction(prediction_time);
CREATE INDEX idx_annotation_recording_time ON annotation(recording_id, start_time, end_time);
CREATE INDEX idx_xai_prediction ON xai_explanation(prediction_id);
CREATE INDEX idx_narrative_patient_time ON narrative(patient_id, generated_at);
CREATE INDEX idx_diagnosis_patient ON diagnosis(patient_id);
CREATE INDEX idx_clinical_assessment_prediction ON clinical_assessment(prediction_id);

CREATE INDEX idx_signal_data_signal_time ON signal_data(signal_id, timestamp DESC);
CREATE INDEX idx_attention_window_time ON attention_data(window_id, timestamp DESC);
CREATE INDEX idx_feature_window_time ON signal_feature_data(window_id, timestamp DESC);
CREATE INDEX idx_prediction_series_window_time ON prediction_series(window_id, timestamp DESC);
