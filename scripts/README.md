# Research entry points

Run these modules from the repository root. Dataset acquisition and model training require the linked source data and a configured local database where applicable.

| Stage | Module | Purpose |
| --- | --- | --- |
| Prepare signals | `scripts.prepare_apnea_ecg`, `scripts.prepare_sleep_edf` | Download selected public records and inspect annotations |
| Train PDM detector | `scripts.train_fusion_models` | Train patient-held-out Apnea-ECG and Sleep-EDF folds |
| Populate PDM | `scripts.populate_fusion_pdm` | Store detector outputs, evidence, and attention in TimescaleDB |
| Select cases | `scripts.build_fusion_cases` | Freeze current-evidence and history conditions |
| Generate narratives | `scripts.run_fusion_experiment` | Save paired evidence packages and narratives |
| Evaluate narratives | `scripts.score_fusion_experiment`, `scripts.adjudicate_fusion_review` | Score grounding and record review decisions |
| Sleep example | `scripts.build_sleep_case` | Save a Sleep-EDF evidence and narrative case |
| Database measurements | `scripts.benchmark_pdm` | Record storage, query, and aggregation measurements |
| Inference measurements | `scripts.measure_fusion_latency` | Record local detector and persistence latency |
| Paper source bundle | `scripts.build_paper_evidence`, `scripts.build_apnea_figure_source` | Assemble result sources and an Apnea-ECG figure input |

The scripts write to the named folders in [outputs](../outputs/README.md). The saved results are records of completed experiments.
