# Dataset models

Each dataset has a `code/` folder containing its model and evaluation routines. The dataset READMEs describe the expected input, training code, baseline methods, and any missing dependencies.

| Dataset | Model code | Data source |
| --- | --- | --- |
| CHB-MIT EEG | [Temporal-attention anomaly detector](chb_mit/) | [PhysioNet CHB-MIT](https://physionet.org/content/chbmit/1.0.0/) |
| Apnea-ECG | [Temporal-attention ECG reconstruction](apnea_ecg/) | [PhysioNet Apnea-ECG](https://physionet.org/content/apnea-ecg/1.0.0/) |
| Sleep-EDF | [BiLSTM attention autoencoder](sleep_edf/) | [PhysioNet Sleep-EDF Expanded](https://physionet.org/content/sleep-edfx/1.0.0/) |
| UCI HAR | [Activity anomaly detection](uci_har/) | [UCI HAR](https://archive.ics.uci.edu/dataset/240/human+activity+recognition+using+smartphones) |
| Synthetic thermal | [Thermal attention autoencoder and baselines](synthetic_thermal/) | [Synthetic thermal CSVs](https://github.com/malik630/Data-Wrangling/tree/9293a0674f577ff7df419bc10782b7a2d1ab9be9/data/raw) |

Apnea-ECG and Sleep-EDF also have a PDM experiment pipeline in [pdm](../pdm/) and [scripts](../scripts/README.md). The five model folders have different input formats and do not share one training command. No raw signal arrays or trained model files are included.
