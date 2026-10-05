-- Viewer review notes; leaves the supplied PDM tables unchanged.
CREATE TABLE IF NOT EXISTS viewer_feedback (
    feedback_id BIGSERIAL PRIMARY KEY,
    prediction_id BIGINT NOT NULL REFERENCES prediction(prediction_id),
    narrative_id BIGINT REFERENCES narrative(narrative_id),
    subject VARCHAR(32) NOT NULL DEFAULT 'overall' CHECK (subject IN ('overall', 'narrative', 'channel')),
    channel VARCHAR(128),
    reviewer VARCHAR(128),
    notes TEXT NOT NULL CHECK (char_length(btrim(notes)) BETWEEN 1 AND 10000),
    created_at TIMESTAMPTZ NOT NULL DEFAULT CURRENT_TIMESTAMP
);
CREATE INDEX IF NOT EXISTS idx_viewer_feedback_event_time ON viewer_feedback(prediction_id, created_at DESC);
