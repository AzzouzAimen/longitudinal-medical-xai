# Saved experiment outputs

| Directory | Contents |
| --- | --- |
| `detector_training/` | Apnea-ECG and Sleep-EDF patient-held-out fold metadata, configurations, normalization, and evaluation decisions |
| `narrative_experiments/apnea_ecg/` | Frozen case selection, prompts, evidence packages, and paired narrative runs |
| `narrative_experiments/sleep_edf_case/` | Saved Sleep-EDF evidence package and narrative |
| `narrative_experiments/scoring/` | Case-level scoring and final result tables |
| `narrative_experiments/adjudication/` | Narrative review records and decisions |
| `database_benchmarks/` | Database measurements and query plans |
| `database_alignment/measurements/` | Stored alignment and aggregation measurements |
| `inference_latency/` | Local inference latency measurements |
| `paper_source_bundle/source_files/` | Paper result source files and their hash manifest |

The paired Apnea-ECG run is `apnea_paired_v7_contextual_freegen_20260917T200000Z`, scored with `fusion-scoring-v7`. The paper source files carry SHA-256 digests in `SOURCE_MANIFEST.json`. The detector folders hold metadata but no model binaries or signal arrays. Some frozen run records refer to paths from the original experiment layout.
