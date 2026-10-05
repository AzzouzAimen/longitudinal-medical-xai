# Apnea-ECG

[PhysioNet Apnea-ECG](https://physionet.org/content/apnea-ecg/1.0.0/) supplies the recordings and event annotations. [training_and_baselines.ipynb](code/training_and_baselines.ipynb) contains the preprocessing, training, evaluation, and baseline workflow. `code/model.py`, `loss.py`, `train.py`, and `evaluate.py` are companion code extracts. The extracts use variables prepared in notebook cells and are not standalone entry points.

The workflow expects 60-second ECG windows with 6,000 samples, labels, recording IDs, window start times, and split labels. It reads a local `.npz` cache (`X`, `y`, `recording_ids`, `window_starts`, `db_splits`) or PostgreSQL table `ml.ecg_minute_windows`. Preprocessing produces 100 steps with seven morphology features. Baselines include supervised BiLSTM, feature K-means, DTW K-means, and an S4 example.

The raw recordings, cache, trained weights, and database ingestion process are not included. `requirements-reference.txt` lists dependencies inferred from the code; database access in the notebook uses Kaggle environment secrets.
