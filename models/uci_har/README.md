# UCI HAR

The [UCI Human Activity Recognition dataset](https://archive.ics.uci.edu/dataset/240/human+activity+recognition+using+smartphones) contains 128-sample inertial-signal windows, activity labels, and subject IDs. `code/` contains temporal-attention model, loss, training, and evaluation code. Its training imports `encoder.py`, `decoder.py`, `mse_loss.py`, `pattern_loss.py`, and a data loader that are not in this folder. The `code/train.py` fallback generates random windows, so it must not be used as a real-data result.

For a complete real-data example, [reference/train.py](reference/train.py) reads the official UCI HAR inertial text files and trains a compact BiLSTM temporal-attention autoencoder. It separates validation by subject, fits normalization on training-normal windows, sets a validation-error threshold, and writes measured test metrics. Choose normal and anomaly activity IDs explicitly:

```text
python reference/train.py --dataset-root "path/to/UCI HAR Dataset" --normal-activities 1 --anomaly-activities 2,3,4,5,6 --output-dir results
```

Use `reference/requirements.txt` for that example. It is a training reference; no run output or paper result is supplied with it.
