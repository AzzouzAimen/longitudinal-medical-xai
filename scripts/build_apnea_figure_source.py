"""Create the traceable representative Apnea PDM–LLM figure source."""

from __future__ import annotations

import hashlib
import json
import shutil
from pathlib import Path

from pdm.evidence import canonical_json_hash
from pdm.llm import _has_private_evaluation_key


ROOT = Path(__file__).resolve().parents[1]
RUN_ID = "apnea_paired_v7_contextual_freegen_20260917T200000Z"
RUN = ROOT / "outputs" / "narrative_experiments" / "apnea_ecg" / "runs" / RUN_ID
OUT = ROOT / "outputs" / "narrative_experiments" / "apnea_figure_source"


def _read(path: Path) -> dict:
    return json.loads(path.read_text(encoding="utf-8"))


def _write(path: Path, payload: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2, sort_keys=True, ensure_ascii=False) + "\n", encoding="utf-8")


def select_case() -> tuple[int, list[dict]]:
    candidates = []
    for case_dir in RUN.glob("cases/prediction_*"):
        package = _read(case_dir / "pdm_enabled" / "evidence_package.json")
        identity = package["current_evidence"]["identity"]
        context = package["pdm_context"]
        candidates.append({
            "prediction_id": int(identity["prediction_id"]),
            "patient_code": identity["patient_code"],
            "dense_current_evidence": package["current_evidence"]["provenance"]["dense_evidence_materialized"],
            "history_count": len(context["detailed_history"]),
            "aggregate_bucket_count": len(context["aggregate_trend"]),
            "clinical_visit_count": len(context["clinical_context"]["visits"]),
            "diagnosis_count": len(context["clinical_context"]["diagnoses"]),
        })
    candidates.sort(key=lambda row: (
        -int(row["dense_current_evidence"]), -row["diagnosis_count"],
        -row["clinical_visit_count"], -row["aggregate_bucket_count"],
        -row["history_count"], row["prediction_id"],
    ))
    selected = candidates[0]
    if not (
        selected["dense_current_evidence"] and selected["history_count"] == 5
        and selected["aggregate_bucket_count"] == 12
        and selected["clinical_visit_count"] and selected["diagnosis_count"]
    ):
        raise ValueError("no frozen Apnea case meets the representative context criteria")
    return selected["prediction_id"], candidates


def main() -> int:
    prediction_id, audit = select_case()
    source = RUN / "cases" / f"prediction_{prediction_id}"
    current = _read(source / "current_evidence" / "evidence_package.json")
    enabled = _read(source / "pdm_enabled" / "evidence_package.json")
    narrative = _read(source / "pdm_enabled" / "parsed_response.json")
    persistence = _read(source / "pdm_enabled" / "persistence.json")
    if _has_private_evaluation_key(current) or _has_private_evaluation_key(enabled):
        raise ValueError("representative public packages contain private evaluation data")
    if current["current_evidence_hash"] != enabled["current_evidence_hash"]:
        raise ValueError("representative pair current-evidence hashes differ")
    OUT.mkdir(parents=True, exist_ok=True)
    current_target = OUT / "current_evidence_package.json"
    enabled_target = OUT / "pdm_enabled_package.json"
    shutil.copy2(source / "current_evidence" / "evidence_package.json", current_target)
    shutil.copy2(source / "pdm_enabled" / "evidence_package.json", enabled_target)
    shutil.copy2(source / "pdm_enabled" / "parsed_response.json", OUT / "pdm_enabled_narrative.json")
    selection = {
        "schema_version": "apnea-representative-selection-v1",
        "selected_prediction_id": prediction_id,
        "selection_basis": "public evidence only: dense current evidence, a complete five-event history, twelve aggregate buckets, a synthetic visit, and a suspected diagnosis; narrative score change was not used",
        "candidate_audit": audit,
    }
    _write(OUT / "selection.json", selection)
    event = enabled["current_evidence"]["event"]
    model = enabled["current_evidence"]["model_evidence"]
    attention = enabled["current_evidence"]["attention_evidence"]
    xai = enabled["current_evidence"]["xai_evidence"]
    context = enabled["pdm_context"]
    artifact = {
        "schema_version": "apnea-pdm-llm-figure-source-v1",
        "run_id": RUN_ID,
        "prediction_id": prediction_id,
        "patient": enabled["current_evidence"]["identity"],
        "current_ecg_window": {
            "event": event,
            "raw_signal_summary": enabled["current_evidence"]["signal_evidence"],
        },
        "anomaly_decision": model["predicted_class"],
        "detector_quantities": {
            key: model[key] for key in (
                "reconstruction_error", "anomaly_threshold", "anomaly_margin", "normalized_margin", "semantics"
            )
        },
        "temporal_xai": {"attention": attention, "explanation": xai},
        "previous_events": context["detailed_history"],
        "aggregate_trend": context["aggregate_trend"],
        "recurrence_summary": context["recurrence_summary"],
        "synthetic_context": context["clinical_context"],
        "narrative": narrative,
        "pdm_persistence": {
            **persistence,
            "prediction_id": prediction_id,
            "explanation_ids": [row["explanation_id"] for row in xai["explanations"]],
        },
        "traceability": {
            "current_evidence_hash": current["current_evidence_hash"],
            "current_package_hash": current["package_hash"],
            "pdm_enabled_package_hash": enabled["package_hash"],
            "current_package_file_sha256": hashlib.sha256(current_target.read_bytes()).hexdigest(),
            "pdm_package_file_sha256": hashlib.sha256(enabled_target.read_bytes()).hexdigest(),
            "narrative_hash": canonical_json_hash(narrative),
            "source_run_manifest": str((RUN / "run_manifest.json").relative_to(ROOT).as_posix()),
        },
        "privacy_check": {
            "official_annotation_in_request": False,
            "private_case_category_in_request": False,
        },
    }
    artifact["artifact_hash"] = canonical_json_hash(artifact)
    _write(OUT / "figure_source.json", artifact)
    sections = narrative["narrative_sections"]
    diagnosis = context["clinical_context"]["diagnoses"][0]
    markdown = f"""# Representative Apnea PDM–LLM artifact

**Frozen run:** `{RUN_ID}`  
**Prediction:** {prediction_id}  
**Narrative ID:** {persistence['narrative_id']}  
**Artifact hash:** `{artifact['artifact_hash']}`

## Why this case

The case was selected using public input richness, not outcome improvement: it has dense ECG evidence, five prior events, twelve aggregate buckets, a synthetic visit, and a synthetic suspected diagnosis. The current detector decision is **{model['predicted_class']}**, which is retained even though the contextual diagnosis status is **{diagnosis['status']}**.

## Current ECG window and decision

{sections['current_event']}

## Detector and temporal XAI

{sections['model_and_xai_evidence']}

## Previous monitoring history

{sections['previous_monitoring_history']}

## Synthetic context

{sections['recorded_clinical_context']}

## Evidence-grounded summary

{sections['evidence_grounded_summary']}

## Traceability and limits

- Current evidence hash: `{current['current_evidence_hash']}`
- Current package hash: `{current['package_hash']}`
- PDM-enabled package hash: `{enabled['package_hash']}`
- The LLM request contains no official Apnea annotation or private TP/FP/TN category.
- Detector margins are not probabilities; attention is shared temporal relevance.
- Clinical material is visibly synthetic and the diagnosis remains suspected.
"""
    (OUT / "APNEA_REPRESENTATIVE.md").write_text(markdown, encoding="utf-8")
    print(json.dumps({"status": "complete", "prediction_id": prediction_id, "output": str(OUT)}, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
