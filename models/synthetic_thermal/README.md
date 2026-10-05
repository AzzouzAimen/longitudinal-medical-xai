# Synthetic thermal time series

`code/` implements a temporal-attention autoencoder (TAAE), evaluation, attention extraction, localization, robustness analysis, and supervised BiLSTM, feature K-means, and DTW K-means baselines. The model uses five channels per 60-sample window: raw and normalized left/right temperature, plus right-minus-left asymmetry. `code/train.py` trains from TimescaleDB and saves a checkpoint under `results/`; `code/evaluate.py` and `code/baselines.py` use `.npy` windows and labels. Use `code/requirements.txt` for model dependencies.

`preprocessing/` contains the thermal CSV cleaning and windowing pipeline, TimescaleDB ingestion code, schema, and dependencies. The [input CSV files](https://github.com/malik630/Data-Wrangling/tree/9293a0674f577ff7df419bc10782b7a2d1ab9be9/data/raw) are linked; no raw or processed data is bundled. From this folder, supply the paths explicitly:

```text
python preprocessing/etl_pipeline.py --data-dir PATH_TO_RAW_CSVS --out-dir PATH_TO_PROCESSED
python preprocessing/ingest_timescaledb.py --etl-dir PATH_TO_PROCESSED --dsn "host=HOST port=5433 dbname=m6_thermal_tsdb user=USER password=PASSWORD"
```

The processed windows are written under `PATH_TO_PROCESSED/npy`. Point evaluation and baseline commands to that directory with `--npy-dir`. The preprocessing code transforms the CSV files; it does not include their original synthetic-signal generator.
