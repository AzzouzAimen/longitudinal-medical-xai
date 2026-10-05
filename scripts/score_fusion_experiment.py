"""Prepare and finalize the paired Apnea evidence-grounding evaluation."""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import random
import shutil
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from pdm.evaluation import METRICS, aggregate_table7, export_semantic_review, score_exact_facts
from pdm.evidence import canonical_json_hash


ROOT = Path(__file__).resolve().parents[1]
EXPERIMENT_ROOT = ROOT / "outputs" / "narrative_experiments" / "apnea_ecg"
RUNS_ROOT = EXPERIMENT_ROOT / "runs"
REVIEW_ROOT = ROOT / "outputs" / "narrative_experiments" / "adjudication"
RESULTS_ROOT = ROOT / "outputs" / "narrative_experiments" / "scoring"
SCORING_VERSION = "fusion-scoring-v2"
SCORING_VERSION_V3 = "fusion-scoring-v3"
SCORING_VERSION_V4 = "fusion-scoring-v4"
SCORING_VERSION_V5 = "fusion-scoring-v5"
SCORING_VERSION_V6 = "fusion-scoring-v6"
SCORING_VERSION_V7 = "fusion-scoring-v7"
CLAIM_CLASSES = {"supported", "unsupported-other", "misattributed", "fabricated"}
EVIDENCE_CLASSES = {"correctly_represented", "incorrectly_represented", "omitted"}


def review_dir_for(run_id: str, scoring_version: str = SCORING_VERSION) -> Path:
    return REVIEW_ROOT / run_id / scoring_version


def result_dir_for(run_id: str, scoring_version: str = SCORING_VERSION) -> Path:
    return RESULTS_ROOT / run_id / scoring_version


def _scoring_version(manifest: dict) -> str:
    if manifest.get("schema_version") == "fusion-paired-run-v7":
        return SCORING_VERSION_V7
    if manifest.get("schema_version") == "fusion-paired-run-v6":
        return SCORING_VERSION_V6
    if manifest.get("schema_version") == "fusion-paired-run-v5":
        return SCORING_VERSION_V5
    if manifest.get("schema_version") == "fusion-paired-run-v4":
        return SCORING_VERSION_V4
    return (
        SCORING_VERSION_V3
        if int(manifest.get("evidence_package_version", 2)) == 3
        else SCORING_VERSION
    )


def _write_json(path: Path, value: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(
        json.dumps(value, indent=2, sort_keys=True, ensure_ascii=False) + "\n",
        encoding="utf-8",
    )
    temporary.replace(path)


def _load_complete_run(run_dir: Path) -> tuple[dict, dict]:
    manifest = json.loads((run_dir / "run_manifest.json").read_text(encoding="utf-8"))
    progress = json.loads((run_dir / "progress.json").read_text(encoding="utf-8"))
    if progress.get("status") != "complete":
        raise ValueError("scoring requires a complete 32-request run")
    if len(progress.get("completed_request_keys", [])) != manifest["expected_request_count"]:
        raise ValueError("run completion count differs from the run contract")
    return manifest, progress


def prepare_review(run_dir: Path) -> Path:
    """Create deterministic proposals, then stop at the semantic-review gate."""

    run_dir = Path(run_dir).resolve()
    manifest, _ = _load_complete_run(run_dir)
    scoring_version = _scoring_version(manifest)
    review_dir = review_dir_for(manifest["run_id"], scoring_version)
    scores, review_rows = [], []
    for request in manifest["requests"]:
        case_id = int(request["prediction_id"])
        condition = request["condition"]
        case_dir = run_dir / "cases" / f"prediction_{case_id}" / request["condition_slug"]
        package = json.loads((case_dir / "evidence_package.json").read_text(encoding="utf-8"))
        narrative = json.loads((case_dir / "parsed_response.json").read_text(encoding="utf-8"))
        score = score_exact_facts(
            package,
            narrative,
            include_context_in_coverage=(
                manifest.get("schema_version") in {
                    "fusion-paired-run-v5", "fusion-paired-run-v6",
                    "fusion-paired-run-v7",
                }
            ),
        )
        narrative_hash = canonical_json_hash(narrative)
        score_record = {
            "case_id": case_id,
            "patient_id": int(package["current_evidence"]["identity"]["patient_id"]),
            "condition": condition,
            "package_hash": package["package_hash"],
            "narrative_hash": narrative_hash,
            **score,
        }
        scores.append(score_record)

        for assessment in score["claim_assessments"]:
            if assessment["requires_review"]:
                review_rows.append({
                "review_id": f"{case_id}:{request['condition_slug']}:claim:{assessment['claim_id']}",
                "record_type": "factual_claim",
                "case_id": case_id,
                "condition": condition,
                "claim_id": assessment["claim_id"],
                "claim_text": assessment["text"],
                "evidence_item_id": "",
                "evidence_ids": assessment["evidence_ids"],
                "proposed_class": assessment["proposed_class"],
                "requires_review": assessment["requires_review"],
                "issue_codes": assessment["issue_codes"],
                "package_hash": package["package_hash"],
                "narrative_hash": narrative_hash,
                "reviewer_class": "",
                "reviewer_note": "",
            })
        for assessment in score["evidence_assessments"]:
            needs_review = (
                assessment["proposed_status"] == "incorrectly_represented"
                or bool(assessment.get("semantic_review_required"))
            )
            if needs_review:
                review_rows.append({
                "review_id": f"{case_id}:{request['condition_slug']}:evidence:{assessment['id']}",
                "record_type": "evidence_item",
                "case_id": case_id,
                "condition": condition,
                "claim_id": "",
                "claim_text": "",
                "evidence_item_id": assessment["id"],
                "evidence_ids": [assessment["evidence_id"]],
                "proposed_class": assessment["proposed_status"],
                "requires_review": needs_review,
                "issue_codes": ["semantic_confirmation"] if needs_review else [],
                "package_hash": package["package_hash"],
                "narrative_hash": narrative_hash,
                "reviewer_class": "",
                "reviewer_note": "",
            })

    _write_json(review_dir / "proposed_case_scores.json", {
        "schema_version": (
            "fusion-proposed-scores-v7" if scoring_version == SCORING_VERSION_V7
            else "fusion-proposed-scores-v6" if scoring_version == SCORING_VERSION_V6
            else "fusion-proposed-scores-v5" if scoring_version == SCORING_VERSION_V5
            else "fusion-proposed-scores-v4" if scoring_version == SCORING_VERSION_V4
            else "fusion-proposed-scores-v3" if scoring_version == SCORING_VERSION_V3
            else "fusion-proposed-scores-v2"
        ),
        "scoring_version": scoring_version,
        "run_id": manifest["run_id"],
        "run_contract_hash": manifest["run_contract_hash"],
        "case_condition_count": len(scores),
        "records": scores,
    })
    review_path = review_dir / "semantic_review.csv"
    export_semantic_review(review_rows, review_path)
    required = sum(bool(row["requires_review"]) for row in review_rows)
    instructions = f"""# Semantic adjudication instructions

This file is the required semantic-adjudication gate for run `{manifest['run_id']}`.

1. Open `semantic_review.csv` and filter `requires_review` to `True`.
2. For each flagged `factual_claim`, enter one of: supported, unsupported-other, misattributed, fabricated.
3. For each flagged `evidence_item`, enter one of: correctly_represented, incorrectly_represented, omitted.
4. Put the decision in `reviewer_class`; optionally explain changes in `reviewer_note`.
5. Leave the IDs, hashes, proposed values, and unflagged rows unchanged.

There are {required} required decisions. Final Table 7 generation remains blocked until they are explicitly classified. Record whether the adjudicator is human or AI-assisted; never describe AI decisions as human review.
"""
    (review_dir / "semantic_review_instructions.md").write_text(instructions, encoding="utf-8")
    _write_json(review_dir / "review_manifest.json", {
        "schema_version": (
            "fusion-semantic-review-v7" if scoring_version == SCORING_VERSION_V7
            else "fusion-semantic-review-v6" if scoring_version == SCORING_VERSION_V6
            else "fusion-semantic-review-v5" if scoring_version == SCORING_VERSION_V5
            else "fusion-semantic-review-v4" if scoring_version == SCORING_VERSION_V4
            else "fusion-semantic-review-v3" if scoring_version == SCORING_VERSION_V3
            else "fusion-semantic-review-v2"
        ),
        "scoring_version": scoring_version,
        "run_id": manifest["run_id"],
        "run_contract_hash": manifest["run_contract_hash"],
        "total_rows": len(review_rows),
        "required_review_rows": required,
        "semantic_review_sha256": __import__("hashlib").sha256(review_path.read_bytes()).hexdigest(),
        "status": "awaiting_semantic_adjudication",
    })
    return review_dir


def _read_review(path: Path) -> list[dict]:
    with Path(path).open(encoding="utf-8-sig", newline="") as handle:
        return list(csv.DictReader(handle))


def approve_proposals(review_path: Path, decisions_path: Path, output_path: Path) -> None:
    """Materialize explicit review decisions without auto-approving proposals."""

    rows = _read_review(review_path)
    decisions = json.loads(Path(decisions_path).read_text(encoding="utf-8"))
    overrides = decisions.get("overrides", {})
    known_ids = {row["review_id"] for row in rows}
    unknown = sorted(set(overrides) - known_ids)
    if unknown:
        raise ValueError(f"decision overrides contain unknown review IDs: {unknown}")
    missing = sorted(known_ids - set(overrides))
    if missing:
        raise ValueError(f"missing explicit reviewer decisions: {missing}")
    for row in rows:
        decision = overrides[row["review_id"]]
        reviewer_class = str(decision.get("reviewer_class", "")).strip()
        allowed = CLAIM_CLASSES if row["record_type"] == "factual_claim" else EVIDENCE_CLASSES
        if reviewer_class not in allowed:
            raise ValueError(
                f"invalid explicit reviewer class for {row['review_id']}: {reviewer_class}"
            )
        row["reviewer_class"] = reviewer_class
        row["reviewer_note"] = decision.get(
            "reviewer_note",
            decisions.get("default_reviewer_note", ""),
        )
    export_semantic_review(rows, output_path)


def _as_bool(value: Any) -> bool:
    return str(value).strip().casefold() in {"true", "1", "yes"}


def _percentile(values: list[float], probability: float) -> float:
    ordered = sorted(values)
    index = (len(ordered) - 1) * probability
    lower = int(index)
    upper = min(lower + 1, len(ordered) - 1)
    fraction = index - lower
    return ordered[lower] * (1 - fraction) + ordered[upper] * fraction


def _paired_summary(case_records: list[dict]) -> dict:
    by_case: dict[int, dict[str, dict]] = {}
    for row in case_records:
        by_case.setdefault(int(row["case_id"]), {})[row["condition"]] = row
    if any(set(pair) != {"Current-Evidence", "PDM-Enabled"} for pair in by_case.values()):
        raise ValueError("each case must contain both frozen conditions")
    rng = random.Random(2026)
    result = {"difference_direction": "PDM-Enabled minus Current-Evidence", "metrics": {}}
    for metric in METRICS:
        eligible_pairs = [
            pair for pair in by_case.values()
            if pair["PDM-Enabled"].get(metric) is not None
            and pair["Current-Evidence"].get(metric) is not None
        ]
        differences = [
            pair["PDM-Enabled"][metric] - pair["Current-Evidence"][metric]
            for pair in eligible_pairs
        ]
        if not differences:
            result["metrics"][metric] = {"n": 0, "mean_difference": None, "bootstrap_95_ci": None}
            continue
        means = []
        for _ in range(10000):
            sample = [differences[rng.randrange(len(differences))] for _ in differences]
            means.append(sum(sample) / len(sample))
        metric_result = {
            "n": len(differences),
            "mean_difference": sum(differences) / len(differences),
            "bootstrap_95_ci": [_percentile(means, 0.025), _percentile(means, 0.975)],
            "case_differences": differences,
        }
        patient_values: dict[str, list[float]] = {}
        for pair, difference in zip(eligible_pairs, differences):
            patient_id = pair["Current-Evidence"].get("patient_id")
            if patient_id is None:
                continue
            patient_values.setdefault(str(patient_id), []).append(difference)
        if patient_values:
            patient_means = {
                patient: sum(values) / len(values)
                for patient, values in sorted(patient_values.items())
            }
            cluster_values = list(patient_means.values())
            cluster_bootstrap = []
            for _ in range(10000):
                sample = [
                    cluster_values[rng.randrange(len(cluster_values))]
                    for _ in cluster_values
                ]
                cluster_bootstrap.append(sum(sample) / len(sample))
            metric_result.update({
                "patient_cluster_count": len(cluster_values),
                "patient_mean_differences": patient_means,
                "patient_cluster_bootstrap_95_ci": [
                    _percentile(cluster_bootstrap, 0.025),
                    _percentile(cluster_bootstrap, 0.975),
                ],
            })
        result["metrics"][metric] = metric_result
    return result


def calculate_approved_results(run_dir: Path, approved_review: Path) -> dict:
    manifest, _ = _load_complete_run(Path(run_dir))
    scoring_version = _scoring_version(manifest)
    rows = _read_review(approved_review)
    expected_review = review_dir_for(manifest["run_id"], scoring_version) / "semantic_review.csv"
    expected_rows = _read_review(expected_review)
    expected_by_id = {row["review_id"]: row for row in expected_rows}
    seen = set()
    for row in rows:
        if not row.get("review_id") or row["review_id"] in seen:
            raise ValueError("review IDs must be present and unique")
        seen.add(row["review_id"])
        final_class = row.get("reviewer_class", "").strip() or row["proposed_class"].strip()
        allowed = CLAIM_CLASSES if row["record_type"] == "factual_claim" else EVIDENCE_CLASSES
        if _as_bool(row.get("requires_review")) and not row.get("reviewer_class", "").strip():
            raise ValueError(f"required reviewer decision is blank: {row['review_id']}")
        if final_class not in allowed:
            raise ValueError(f"invalid final class for {row['review_id']}: {final_class}")
        expected = expected_by_id.get(row["review_id"])
        if expected is None:
            raise ValueError(f"unexpected review row: {row['review_id']}")
        immutable_fields = set(expected) - {"reviewer_class", "reviewer_note"}
        if any(row.get(field, "") != expected.get(field, "") for field in immutable_fields):
            raise ValueError(f"frozen review fields changed: {row['review_id']}")
        row["final_class"] = final_class

    expected_ids = set(expected_by_id)
    if seen != expected_ids:
        raise ValueError("approved review rows differ from the frozen proposal")

    reviewed_by_id = {row["review_id"]: row for row in rows}
    proposal = json.loads(
        (review_dir_for(manifest["run_id"], scoring_version) / "proposed_case_scores.json").read_text(encoding="utf-8")
    )
    case_records = []
    for proposed in proposal["records"]:
        case_id, condition = int(proposed["case_id"]), proposed["condition"]
        slug = "current_evidence" if condition == "Current-Evidence" else "pdm_enabled"
        claims = []
        supported_context_refs: set[str] = set()
        v3_context_ids = set(proposed.get("context_evidence_ids", []))
        for claim in proposed["claim_assessments"]:
            review_id = f"{case_id}:{slug}:claim:{claim['claim_id']}"
            final_class = reviewed_by_id.get(review_id, {}).get("final_class", claim["proposed_class"])
            claims.append(final_class)
            if v3_context_ids and final_class == "supported":
                supported_context_refs.update(
                    ref for ref in claim["evidence_ids"] if ref in v3_context_ids
                )
        evidence = []
        for item in proposed["evidence_assessments"]:
            review_id = f"{case_id}:{slug}:evidence:{item['id']}"
            final_class = reviewed_by_id.get(review_id, {}).get("final_class", item["proposed_status"])
            evidence.append(final_class)
        claim_counts = {name: claims.count(name) for name in CLAIM_CLASSES}
        evidence_counts = {name: evidence.count(name) for name in EVIDENCE_CLASSES}
        factual_count, relevant_count = len(claims), len(evidence)
        unsupported = sum(claim_counts[name] for name in CLAIM_CLASSES - {"supported"})
        case_record = {
            "case_id": case_id,
            "condition": condition,
            "factual_claim_count": factual_count,
            "relevant_evidence_count": relevant_count,
            **{f"{name}_count": count for name, count in claim_counts.items()},
            **{f"{name}_count": count for name, count in evidence_counts.items()},
            "coverage": evidence_counts["correctly_represented"] / relevant_count if relevant_count else None,
            "omission_rate": evidence_counts["omitted"] / relevant_count if relevant_count else None,
            "incorrect_representation_rate": evidence_counts["incorrectly_represented"] / relevant_count if relevant_count else None,
            "unsupported_claim_rate": unsupported / factual_count if factual_count else None,
            "misattribution_rate": claim_counts["misattributed"] / factual_count if factual_count else None,
            "fabrication_rate": claim_counts["fabricated"] / factual_count if factual_count else None,
            "factual_fidelity": claim_counts["supported"] / factual_count if factual_count else None,
            "coverage_scope": proposed["coverage_scope"],
            "context_relevant_evidence_count": proposed["context_relevant_evidence_count"],
            "context_cited_evidence_count": (
                len(supported_context_refs) if v3_context_ids
                else proposed["context_cited_evidence_count"]
            ),
            "context_utilization": (
                len(supported_context_refs) / len(v3_context_ids) if v3_context_ids
                else proposed["context_utilization"]
            ),
            "format_failure": factual_count == 0,
        }
        if proposed.get("patient_id") is not None:
            case_record["patient_id"] = int(proposed["patient_id"])
        case_records.append(case_record)
    if len(case_records) != manifest["expected_request_count"]:
        raise ValueError("approved review does not cover all case-condition records")
    return {
        "schema_version": (
            "fusion-table7-results-v7" if scoring_version == SCORING_VERSION_V7
            else "fusion-table7-results-v6" if scoring_version == SCORING_VERSION_V6
            else "fusion-table7-results-v5" if scoring_version == SCORING_VERSION_V5
            else "fusion-table7-results-v4" if scoring_version == SCORING_VERSION_V4
            else "fusion-table7-results-v3" if scoring_version == SCORING_VERSION_V3
            else "fusion-table7-results-v2"
        ),
        "scoring_version": scoring_version,
        "run_id": manifest["run_id"],
        "run_contract_hash": manifest["run_contract_hash"],
        "case_records": case_records,
        "aggregate": aggregate_table7(case_records),
        "paired_analysis": _paired_summary(case_records),
    }


def finalize_review(
    run_dir: Path,
    approved_review: Path,
    adjudication: dict | None = None,
) -> Path:
    results = calculate_approved_results(run_dir, approved_review)
    result_dir = result_dir_for(results["run_id"], results["scoring_version"])
    result_dir.mkdir(parents=True, exist_ok=True)
    shutil.copy2(approved_review, result_dir / "approved_semantic_review.csv")
    _write_json(result_dir / "case_level_results.json", {
        "schema_version": results["schema_version"],
        "scoring_version": results["scoring_version"],
        "run_id": results["run_id"],
        "run_contract_hash": results["run_contract_hash"],
        "records": results["case_records"],
    })
    _write_json(result_dir / "table7_results.json", {
        key: value for key, value in results.items() if key != "case_records"
    })
    manifest_path = review_dir_for(results["run_id"], results["scoring_version"]) / "review_manifest.json"
    review_manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    review_manifest.update({
        "status": "adjudication_complete",
        "completed_at_utc": datetime.now(timezone.utc).isoformat(),
        "approved_semantic_review_sha256": hashlib.sha256(
            (result_dir / "approved_semantic_review.csv").read_bytes()
        ).hexdigest(),
        "result_dir": str(result_dir.relative_to(ROOT).as_posix()),
        "adjudication": adjudication or {
            "adjudication_type": "external_unspecified",
            "disclosure": "Adjudicator provenance was not supplied.",
        },
    })
    _write_json(manifest_path, review_manifest)
    return result_dir


def verify_reproduction(run_dir: Path, result_dir: Path) -> None:
    approved = result_dir / "approved_semantic_review.csv"
    regenerated = calculate_approved_results(run_dir, approved)
    saved_case = json.loads((result_dir / "case_level_results.json").read_text(encoding="utf-8"))
    saved_table = json.loads((result_dir / "table7_results.json").read_text(encoding="utf-8"))
    expected_case = {
        "schema_version": regenerated["schema_version"],
        "scoring_version": regenerated["scoring_version"], "run_id": regenerated["run_id"],
        "run_contract_hash": regenerated["run_contract_hash"], "records": regenerated["case_records"],
    }
    expected_table = {key: value for key, value in regenerated.items() if key != "case_records"}
    if saved_case != expected_case or saved_table != expected_table:
        raise ValueError("saved Table 7 results do not reproduce from the approved review")


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run-dir", type=Path)
    mode = parser.add_mutually_exclusive_group()
    mode.add_argument("--approved-review", type=Path)
    mode.add_argument("--approve-proposals", action="store_true")
    mode.add_argument("--verify-reproduction", action="store_true")
    parser.add_argument("--result-dir", type=Path)
    parser.add_argument("--decision-overrides", type=Path)
    return parser


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    if args.verify_reproduction and args.run_dir is None:
        candidates = sorted(
            (path.parent for path in RESULTS_ROOT.rglob("table7_results.json")),
            key=lambda path: path.stat().st_mtime,
        )
        if not candidates:
            raise ValueError("no saved result is available for reproduction")
        args.result_dir = args.result_dir or candidates[-1]
        args.run_dir = RUNS_ROOT / args.result_dir.parent.name
    if args.run_dir is None:
        raise ValueError("--run-dir is required for review preparation or finalization")
    if args.approve_proposals:
        if not args.decision_overrides:
            raise ValueError("--decision-overrides is required with --approve-proposals")
        manifest, _ = _load_complete_run(args.run_dir)
        review_dir = review_dir_for(manifest["run_id"], _scoring_version(manifest))
        approved = review_dir / "approved_semantic_review.csv"
        approve_proposals(
            review_dir / "semantic_review.csv", args.decision_overrides, approved
        )
        decision_document = json.loads(args.decision_overrides.read_text(encoding="utf-8"))
        adjudication = {
            key: value for key, value in decision_document.items() if key != "overrides"
        }
        result_dir = finalize_review(args.run_dir, approved, adjudication)
        print(json.dumps({"status": "complete", "result_dir": str(result_dir)}, indent=2))
    elif args.verify_reproduction:
        if not args.result_dir:
            raise ValueError("--result-dir is required with --verify-reproduction")
        verify_reproduction(args.run_dir, args.result_dir)
        print(json.dumps({"status": "reproduced", "result_dir": str(args.result_dir)}, indent=2))
    elif args.approved_review:
        result_dir = finalize_review(args.run_dir, args.approved_review)
        print(json.dumps({"status": "complete", "result_dir": str(result_dir)}, indent=2))
    else:
        review_dir = prepare_review(args.run_dir)
        print(json.dumps({"status": "awaiting_semantic_adjudication", "review_dir": str(review_dir)}, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
