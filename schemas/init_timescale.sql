CREATE EXTENSION IF NOT EXISTS timescaledb;
SELECT create_hypertable('signal_data', 'timestamp', if_not_exists => TRUE);
SELECT create_hypertable('signal_feature_data', 'timestamp', if_not_exists => TRUE);
SELECT create_hypertable('attention_data', 'timestamp', if_not_exists => TRUE);
SELECT create_hypertable('prediction_series', 'timestamp', if_not_exists => TRUE);

