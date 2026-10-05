# CHB-MIT EEG

The [CHB-MIT Scalp EEG Database](https://physionet.org/content/chbmit/1.0.0/) provides the recordings and seizure annotations. `code/src/medattnaid/` contains the temporal-attention model, loss, data access, training, evaluation, localization, ablation, and baseline routines. Dependencies are listed in `code/requirements.txt`.

The model reads cleaned EEG signals from TimescaleDB (`eeg_signals`) or PostgreSQL tables `recordings`, `channels`, `seizure_annotations`, and `signal_samples`. A database dump and EDF ingestion routine are not included. Localization examples cover `chb01_03` and `chb01_04`.

From `code/`, put `src` on `PYTHONPATH` and use `medattnaid.train`, `medattnaid.evaluate`, or `medattnaid.train_ablations`. Their output paths are under `results/`.
