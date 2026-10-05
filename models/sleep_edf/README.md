# Sleep-EDF

`code/model.py` implements a BiLSTM temporal-attention autoencoder; `code/loss.py` combines fidelity, pattern, and trend reconstruction terms. `code/train.py` and `code/eval.py` expect `data/sleep_edf_processed.npy`, a single-channel `(N, T)` array, when run from `code/`. Training writes a checkpoint under `results/`; evaluation reads that checkpoint and writes reconstruction errors and predictions there.

The input preprocessing and checkpoint are not included. This training routine uses all supplied windows and evaluates with an 85th-percentile threshold, so its split differs from the separate patient-held-out PDM training in [pdm/training.py](../../pdm/training.py). `requirements-reference.txt` lists dependencies inferred from imports.

Data catalogs: [Sleep-EDF](https://physionet.org/content/sleep-edf/1.0.0/) and [Sleep-EDF Expanded](https://physionet.org/content/sleep-edfx/1.0.0/). The PDM signal preparation code is [prepare_sleep_edf.py](../../scripts/prepare_sleep_edf.py).
