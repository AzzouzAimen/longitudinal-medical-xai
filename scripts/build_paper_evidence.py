"""Assemble the Phase 17 paper evidence pack from frozen raw artifacts."""

from __future__ import annotations

import hashlib
import json
import shutil
from pathlib import Path

from scripts.score_fusion_experiment import verify_reproduction


ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "outputs" / "paper_source_bundle"
RAW = OUT / "source_files"
RUN_ID = "apnea_paired_v7_contextual_freegen_20260917T200000Z"
SCORING_VERSION = "fusion-scoring-v7"
RESULTS = ROOT / "outputs" / "narrative_experiments" / "scoring" / RUN_ID / SCORING_VERSION
REVIEW = ROOT / "outputs" / "narrative_experiments" / "adjudication" / RUN_ID / SCORING_VERSION
SLEEP = ROOT / "outputs" / "narrative_experiments" / "sleep_edf_case"
ALIGN = ROOT / "outputs" / "database_alignment" / "measurements"
LATENCY = ROOT / "outputs" / "inference_latency"
SELECTION = ROOT / "outputs" / "narrative_experiments" / "apnea_ecg" / "v7" / "frozen" / "case_selection.json"
POLICY = ROOT / "outputs" / "narrative_experiments" / "apnea_ecg" / "generation_policy_v7.json"
PROMPT = ROOT / "outputs" / "narrative_experiments" / "apnea_ecg" / "prompt_v7.txt"


def _load(path: Path) -> dict:
    return json.loads(path.read_text(encoding="utf-8"))


def _pct(value: float) -> str:
    return f"{100.0 * value:.2f}%"


def require_final_scoring() -> None:
    """Prevent paper artifacts from being built from proposals or scoring v1."""

    manifest_path = REVIEW / "review_manifest.json"
    manifest = _load(manifest_path)
    required = (RESULTS / "table7_results.json", RESULTS / "approved_semantic_review.csv")
    if manifest.get("status") != "adjudication_complete" or not all(path.exists() for path in required):
        raise RuntimeError(
            "human review is not complete or AI adjudication is unavailable; "
            "final v7 paper results cannot be built"
        )
    table = _load(RESULTS / "table7_results.json")
    approved_hash = hashlib.sha256(
        (RESULTS / "approved_semantic_review.csv").read_bytes()
    ).hexdigest()
    if not (
        manifest.get("schema_version") == "fusion-semantic-review-v7"
        and manifest.get("scoring_version") == SCORING_VERSION
        and manifest.get("run_id") == RUN_ID
        and manifest.get("adjudication", {}).get("adjudication_type")
        == "ai_assisted_evidence_audit"
        and manifest.get("approved_semantic_review_sha256") == approved_hash
        and table.get("schema_version") == "fusion-table7-results-v7"
        and table.get("scoring_version") == SCORING_VERSION
        and table.get("run_id") == RUN_ID
        and table.get("run_contract_hash") == manifest.get("run_contract_hash")
    ):
        raise RuntimeError("v7 scoring provenance or hashes do not match the paper contract")
    verify_reproduction(
        ROOT / "outputs/narrative_experiments/apnea_ecg/runs" / RUN_ID,
        RESULTS,
    )


def _copy_sources() -> dict:
    sources = {
        "table7_results.json": RESULTS / "table7_results.json",
        "case_level_results.json": RESULTS / "case_level_results.json",
        "approved_semantic_review.csv": RESULTS / "approved_semantic_review.csv",
        "ai_adjudication_decisions.json": REVIEW / "ai_adjudication_decisions.json",
        "semantic_review_manifest.json": REVIEW / "review_manifest.json",
        "apnea_case_selection.json": SELECTION,
        "sleep_case_manifest.json": SLEEP / "manifest.json",
        "sleep_evidence_package.json": SLEEP / "evidence_package.json",
        "sleep_narrative.json": SLEEP / "parsed_response.json",
        "pdm_alignment.json": ALIGN / "alignment.json",
        "aggregate_query_output.json": ALIGN / "aggregate_query_output.json",
        "aggregate_query_plan.json": ALIGN / "aggregate_query_plan.json",
        "latency_summary.json": LATENCY / "summary.json",
        "latency_raw_measurements.csv": LATENCY / "raw_measurements.csv",
        "generation_policy_v7.json": POLICY,
        "prompt_v7.txt": PROMPT,
    }
    RAW.mkdir(parents=True, exist_ok=True)
    for stale in RAW.iterdir():
        if stale.is_file():
            stale.unlink()
    manifest = {}
    for name, source in sources.items():
        if not source.exists():
            raise FileNotFoundError(source)
        target = RAW / name
        shutil.copy2(source, target)
        manifest[name] = {
            "source": str(source.relative_to(ROOT).as_posix()),
            "sha256": hashlib.sha256(target.read_bytes()).hexdigest(),
            "bytes": target.stat().st_size,
        }
    (RAW / "SOURCE_MANIFEST.json").write_text(
        json.dumps(manifest, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    return manifest


def main() -> int:
    require_final_scoring()
    OUT.mkdir(parents=True, exist_ok=True)
    source_manifest = _copy_sources()
    table = _load(RESULTS / "table7_results.json")
    selection = _load(SELECTION)
    policy = _load(POLICY)
    sleep_package = _load(SLEEP / "evidence_package.json")
    sleep_manifest = _load(SLEEP / "manifest.json")
    sleep_narrative = _load(SLEEP / "parsed_response.json")
    alignment = _load(ALIGN / "alignment.json")
    latency = _load(LATENCY / "summary.json")
    proposed = _load(REVIEW / "review_manifest.json")

    cases = selection["selection_core"]["cases"]
    distribution = {category: sum(row["category"] == category for row in cases) for category in ("TP", "FP", "TN")}
    run_cases = ROOT / "outputs" / "narrative_experiments" / "apnea_ecg" / "runs" / RUN_ID / "cases"
    context = []
    for case in cases:
        package = _load(run_cases / f"prediction_{case['prediction_id']}" / "pdm_enabled" / "evidence_package.json")
        item = package["pdm_context"]
        context.append({
            "history": bool(item["detailed_history"]), "aggregate": bool(item["aggregate_trend"]),
            "visit": bool(item["clinical_context"]["visits"]),
            "diagnosis": bool(item["clinical_context"]["diagnoses"]),
        })
    context_counts = {key: sum(row[key] for row in context) for key in context[0]}

    implementation = f"""# Implementation summary

## Scope

The local implementation is a two-dataset research prototype: six patient-held-out Apnea-ECG folds and three subject-held-out Sleep-EDF folds. It does not replace professor-supplied results for other modalities.

## Detector and training contract

Both datasets use the fixed two-stage BiLSTM temporal-attention autoencoder: encoder widths 32 and 16, a 16-dimensional LayerNorm latent representation, and decoder widths 16 and 32. The implemented clinical-pattern loss approximation is `0.3 × pattern + 0.4 × trend + 0.3 × fidelity`, where pattern compares first temporal differences, trend compares least-squares slopes, and fidelity is mean squared reconstruction error.

LOSO separates test patients from training, normalization, early stopping, and threshold calibration. Apnea uses 60-second ECG windows transformed from 6,000 raw samples to 100×7 features. Sleep uses retained 30-second 300×3 windows from three modeled channels; unknown/movement epochs outside the meaningful interval are excluded.

## PDM and evidence design

Every analyzed window has a lightweight relational prediction and TimescaleDB history row. Dense raw/reconstruction/residual/attention evidence is materialized only for deterministic representative windows. Sleep channel residual thresholds come from validation-normal stable stage-2 windows. Attention is shared window-level temporal relevance with no channel owner.

The five-minute `pdm_prediction_5m` continuous aggregate is the source for historical trend retrieval. Synthetic Apnea clinical context is visibly marked `synthetic_prototype`, frozen before generation, and retains suspected rather than confirmed status.

## Narrative generation and persistence

The paired Apnea experiment used `{policy['model']}` with temperature {policy['generation_settings']['temperature']}, top-p {policy['generation_settings']['top_p']}, maximum {policy['generation_settings']['max_tokens']} tokens, and explicit non-thinking mode. Sixteen newly frozen cases generated 32 paired narratives. The LLM generated every narrative claim; no deterministic fact renderer, grounding correction, claim omission, or evaluator feedback was used. Only one identical retry for malformed JSON was permitted. One saved response required a recorded structural normalization that mapped two history subsection labels to the allowed history section; its claim text and citations were unchanged. Narratives persist through the existing `narrative` and `narrative_evidence` tables; Apnea links only current prediction/XAI evidence, while the separate Sleep demonstration additionally links its supplied sleep-stage annotation.

Raw source manifest: `source_files/SOURCE_MANIFEST.json`.
"""
    (OUT / "IMPLEMENTATION_SUMMARY.md").write_text(implementation, encoding="utf-8")

    current = table["aggregate"]["Current-Evidence"]
    enabled = table["aggregate"]["PDM-Enabled"]
    paired = table["paired_analysis"]["metrics"]
    current_represented = sum(row["correctly_represented_count"] for row in current["case_values"])
    enabled_represented = sum(row["correctly_represented_count"] for row in enabled["case_values"])
    metric_labels = [
        ("coverage", "Evidence coverage"), ("unsupported_claim_rate", "Unsupported claim rate"),
        ("omission_rate", "Omission rate"), ("incorrect_representation_rate", "Incorrect representation rate"),
        ("misattribution_rate", "Misattribution rate"), ("fabrication_rate", "Fabrication rate"),
        ("factual_fidelity", "Factual fidelity"),
    ]
    result_lines = [
        "# Evidence-fusion evaluation results", "",
        "The results are organized according to the corresponding locations in `JAISE2026-new.pdf`: Section 4.5.2, Section 4.5.4, and Table 7 on pages 13-15.", "",
        "## Section 4.5.2 - PDM Context Ablation (page 13)", "",
        "### Evaluation sample", "",
        "| Sample characteristic | Value |",
        "|---|---:|",
        f"| Monitoring events | **N = {len(cases)}** |",
        f"| True-positive cases | **{distribution['TP']}** |",
        f"| False-positive cases | **{distribution['FP']}** |",
        f"| True-negative cases | **{distribution['TN']}** |", "",
        "### Experimental description", "",
        f"The comparison is performed on N = {len(cases)} monitoring events sampled from the Apnea-ECG detection experiments, including {distribution['TP']} true-positive, {distribution['FP']} false-positive, and {distribution['TN']} true-negative cases. Each event is processed under both configurations, producing N paired interpretations.", "",
        "## Section 4.5.4 - Quantitative Results (page 14)", "",
        "### Main quantitative results", "",
        "| Measure | Result |",
        "|---|---:|",
        f"| PDM-Enabled evidence coverage | **{_pct(enabled['coverage']['mean'])}** |",
        f"| Current-Evidence coverage | **{_pct(current['coverage']['mean'])}** |",
        f"| Coverage difference | **{paired['coverage']['mean_difference'] * 100:.2f} percentage points** |",
        f"| Unsupported claim rate: PDM-Enabled | **{_pct(enabled['unsupported_claim_rate']['mean'])}** |",
        f"| Unsupported claim rate: Current-Evidence | **{_pct(current['unsupported_claim_rate']['mean'])}** |",
        f"| Fabrication rate: PDM-Enabled | **{_pct(enabled['fabrication_rate']['mean'])}** |",
        f"| Fabrication rate: Current-Evidence | **{_pct(current['fabrication_rate']['mean'])}** |", "",
        "### Results summary", "",
        f"The PDM-Enabled Fusion configuration achieved an evidence coverage of {_pct(enabled['coverage']['mean'])}, compared with {_pct(current['coverage']['mean'])} for Current-Evidence, corresponding to a difference of {paired['coverage']['mean_difference'] * 100:.2f} percentage points. The unsupported claim rate was {_pct(enabled['unsupported_claim_rate']['mean'])} and {_pct(current['unsupported_claim_rate']['mean'])}, respectively, while fabrication rates were {_pct(enabled['fabrication_rate']['mean'])} and {_pct(current['fabrication_rate']['mean'])}. The paired coverage difference had a case-bootstrap 95% confidence interval of [{paired['coverage']['bootstrap_95_ci'][0] * 100:.2f}, {paired['coverage']['bootstrap_95_ci'][1] * 100:.2f}] percentage points and a patient-cluster bootstrap 95% confidence interval of [{paired['coverage']['patient_cluster_bootstrap_95_ci'][0] * 100:.2f}, {paired['coverage']['patient_cluster_bootstrap_95_ci'][1] * 100:.2f}] percentage points.", "",
        "## Table 7 - Evidence-grounding comparison (page 15)", "",
        f"Evaluation sample: **N = {len(cases)} paired cases**.", "",
        "| Configuration | Coverage (%) | Unsupported (%) | Omission (%) | Misattrib. (%) | Fabrication (%) |",
        "|---|---:|---:|---:|---:|---:|",
        f"| Current-Evidence | {current['coverage']['mean'] * 100:.2f} ± {current['coverage']['sd'] * 100:.2f} | {current['unsupported_claim_rate']['mean'] * 100:.2f} | {current['omission_rate']['mean'] * 100:.2f} | {current['misattribution_rate']['mean'] * 100:.2f} | {current['fabrication_rate']['mean'] * 100:.2f} |",
        f"| PDM-Enabled Fusion | {enabled['coverage']['mean'] * 100:.2f} ± {enabled['coverage']['sd'] * 100:.2f} | {enabled['unsupported_claim_rate']['mean'] * 100:.2f} | {enabled['omission_rate']['mean'] * 100:.2f} | {enabled['misattribution_rate']['mean'] * 100:.2f} | {enabled['fabrication_rate']['mean'] * 100:.2f} |", "",
        "Table note: Coverage is reported as mean ± standard deviation across the 16 paired cases. The remaining columns report mean case-level rates.", "",
        "## Brief interpretation for the paper", "",
        f"PDM-enabled evidence fusion increased mean evidence coverage by {paired['coverage']['mean_difference'] * 100:.2f} percentage points and reduced the mean omission rate by {-paired['omission_rate']['mean_difference'] * 100:.2f} percentage points. This supports the conclusion that retrieved longitudinal context helped the LLM produce more complete evidence-grounded narratives. It does not show improved anomaly-detection accuracy, diagnostic performance, or clinical benefit.", "",
        "## Methodological note", "",
        "Both conditions used the same LLM, generation settings, output schema, citation rules, and safety constraints. The shared prompt asked Current-Evidence for a concise current-event narrative and PDM-Enabled for a broader contextual narrative covering the supplied history categories. The comparison therefore evaluates the complete PDM-enabled interpretation process, not a strict input-only ablation in which the generation instruction is identical between conditions.", "",
        "## Complete statistical results", "",
        f"The final experiment contains **{len(cases)} cases** ({distribution['TP']} TP, {distribution['FP']} FP, {distribution['TN']} TN) and 32 narratives. These prediction identities exclude all 64 identities used in preceding narrative-development and evaluation cohorts. Every PDM package has prior history and aggregate/recurrence summaries; {context_counts['visit']} have synthetic visits and {context_counts['diagnosis']} has a synthetic suspected diagnosis.", "",
        "| Metric | Current Evidence mean ± SD | PDM-Enabled mean ± SD | Paired difference | Case bootstrap 95% CI | Patient-cluster bootstrap 95% CI |",
        "|---|---:|---:|---:|---:|---:|",
    ]
    for key, label in metric_labels:
        cur, pdm, diff = current[key], enabled[key], paired[key]
        ci = diff["bootstrap_95_ci"]
        cluster_ci = diff.get("patient_cluster_bootstrap_95_ci")
        result_lines.append(
            f"| {label} | {_pct(cur['mean'])} ± {_pct(cur['sd'])} | {_pct(pdm['mean'])} ± {_pct(pdm['sd'])} | {_pct(diff['mean_difference'])} | [{_pct(ci[0])}, {_pct(ci[1])}] | [{_pct(cluster_ci[0])}, {_pct(cluster_ci[1])}] |"
        )
    result_lines.extend([
        "", "## Semantic adjudication", "",
        f"Exact facts were checked automatically and all {proposed['required_review_rows']} ambiguous row(s) received explicit AI-assisted evidence adjudication. This was not human review. The final decision CSV, adjudication provenance, and every case-level denominator are preserved under `source_files/`.",
        "", "## Observations", "",
        f"Coverage is the proportion of all relevant concepts supplied to each condition: up to seven available current-event concepts for Current-Evidence and those available concepts plus four longitudinal-context concepts for PDM-Enabled. Mean coverage was {_pct(current['coverage']['mean'])} and {_pct(enabled['coverage']['mean'])}, respectively, a paired change of {_pct(paired['coverage']['mean_difference'])}; both the case and patient-cluster bootstrap intervals were above zero. PDM-Enabled used all four context categories in every case ({_pct(enabled['context_utilization']['mean'])}) and represented {enabled_represented} grounded concepts versus {current_represented} for Current-Evidence because it had a larger evidence set. Its unsupported-claim, misattribution, and fabrication rates were {_pct(enabled['unsupported_claim_rate']['mean'])}, {_pct(enabled['misattribution_rate']['mean'])}, and {_pct(enabled['fabrication_rate']['mean'])}.",
        "", "## Limitations", "",
        "The sample is deliberately small and was selected deterministically by label strata, patient quotas, margin diversity, and time separation; it is not a random clinical sample. The identities are held out from 64 earlier narrative-engineering and evaluation cases, but the source patients are the same dataset patients, so this is case-held-out engineering evidence rather than patient-held-out clinical validation. PDM-Enabled had a larger coverage denominator, so its improvement concerns completeness relative to the richer evidence supplied, not detector accuracy. Semantic decisions were made by an AI-assisted evidence audit, not by human reviewers; this can introduce evaluator-model bias. Case bootstrap intervals treat windows as paired observations, while the patient-cluster bootstrap resamples patient-level means. Neither is a population-level clinical claim. The synthetic context is not clinical ground truth, and model outputs are not diagnoses.",
    ])
    (OUT / "EVIDENCE_FUSION_RESULTS.md").write_text("\n".join(result_lines) + "\n", encoding="utf-8")

    event = sleep_package["current_event"]
    detector = sleep_package["detector_evidence"]
    channels = sleep_package["channel_evidence"]
    prior = sleep_package["previous_night_context"]
    stage = sleep_package["recorded_sleep_stage_context"]
    flagged = [row["channel"] for row in channels if row["residual_threshold_flag"]]
    sleep_summary = f"""# Representative Sleep-EDF case

The selected out-of-fold event is prediction **{sleep_manifest['prediction_id']}** for `{event['patient_code']}` on the patient's second recording. It is a 30-second 300×3 modeled window with timestamp provenance `{event['timestamp_provenance']}`.

The shared model reported reconstruction error {detector['reconstruction_error']:.6f} against threshold {detector['anomaly_threshold']:.6f}. Two modeled channels were flagged as component evidence: {', '.join(flagged)}. These are not independent detectors. EOG horizontal contributed {next(row['contribution_share'] for row in channels if row['channel'] == 'EOG horizontal'):.2%} of squared residual evidence. Attention remains shared temporal relevance with no channel owner.

The previous same-patient night `{prior['recording_id']}` contains {prior['window_count']} retained windows and {prior['anomaly_count']} anomaly decisions ({prior['anomaly_rate']:.2%}). The current hypnogram context is stage `{stage['label']}` from `{stage['source']}`; it is context, not diagnosis or detector ground truth. Respiration, EMG, and temperature are auxiliary signals and were not modeled reconstruction inputs.

The generated narrative is persisted as PDM narrative **{sleep_manifest['narrative_id']}**. Package hash: `{sleep_manifest['package_hash']}`. All cited evidence IDs validate, and no claim describes shared attention as sensor-specific.

This is one representative demonstration, not part of the paired Table 7 experiment and not evidence of clinical utility.
"""
    (OUT / "SLEEP_CASE_SUMMARY.md").write_text(sleep_summary, encoding="utf-8")

    benchmark = alignment["benchmark"]["paper_ready_results"]
    hypertable_rows = "\n".join(
        f"- `{row['hypertable_name']}`: {row['num_chunks']} chunks, compression enabled = {row['compression_enabled']}"
        for row in alignment["hypertables"]
    )
    pdm_features = f"""# PDM features and benchmark evidence

## Storage model

Relational tables retain patients, visits, diagnoses, recordings, signals, windows, models, predictions, annotations, XAI metadata, narratives, and evidence links. TimescaleDB hypertables retain `signal_data`, `signal_feature_data`, `attention_data`, and `prediction_series`. Lightweight prediction history is complete; dense time series and XAI are selectively materialized.

{hypertable_rows}

Migration `001_fusion_alignment` adds patient/model/recording provenance, detector quantities, constraints/indexes, and the continuous aggregate. `pdm_prediction_5m` groups by five-minute bucket, patient, and model. It is explicitly refreshed after historical population and is read directly by `get_prediction_trend`.

The current aggregate contains {alignment['continuous_aggregate']['row_count']} rows representing all {alignment['continuous_aggregate']['prediction_series_rows']} source rows, with {alignment['continuous_aggregate']['full_recalculation_mismatch_count']} independent-calculation mismatches. Query output and an analyzed plan are preserved under `source_files/`.

## Existing two-dataset benchmark

- {benchmark['ingestion_observation']}
- {benchmark['simple_range_observation']}
- {benchmark['history_observation']}
- {benchmark['aggregation_observation']}
- {benchmark['storage_observation']}
- {benchmark['compression_observation']}

The primary comparison remains uncompressed PostgreSQL versus uncompressed TimescaleDB; columnstore is a separate historical-storage result. Results are workload-specific, include PostgreSQL wins, and cover only Apnea-ECG and Sleep-EDF. Shifted benchmark timestamps are never used as patient evidence.
"""
    (OUT / "PDM_FEATURES.md").write_text(pdm_features, encoding="utf-8")

    latency_lines = [
        "# Local fusion latency", "",
        "Ten warm-ups preceded 100 independent real held-out windows per modality. Model loading and network LLM generation are excluded from per-window latency.", "",
        "| Dataset | Input | Mean ± SD total | Median total | p95 total | Streaming budget | p95 safety factor |",
        "|---|---:|---:|---:|---:|---:|---:|",
    ]
    for dataset in ("apnea", "sleep"):
        item = latency[dataset]
        total = item["total_local"]
        latency_lines.append(
            f"| {dataset.title()} | {'100×7' if dataset == 'apnea' else '300×3'} | {total['mean_ms']:.3f} ± {total['sd_ms']:.3f} ms | {total['median_ms']:.3f} ms | {total['p95_ms']:.3f} ms | {item['streaming_budget_seconds']:.0f} s | {item['p95_safety_factor']:.1f}× |"
        )
    latency_lines.extend([
        "", "Inference, residual/channel-evidence preparation, and rollback-safe PDM persistence are separately recorded in `source_files/latency_raw_measurements.csv`. These measurements support only the local Apnea and Sleep prototypes and do not replace professor-supplied modality results.",
    ])
    (OUT / "LATENCY_RESULTS.md").write_text("\n".join(latency_lines) + "\n", encoding="utf-8")

    print(json.dumps({"status": "complete", "output": str(OUT), "sources": len(source_manifest)}, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
