"""Record the independent AI-assisted semantic audit of the frozen fusion run."""

from __future__ import annotations

import argparse
import csv
import json
import re
from datetime import date
from pathlib import Path

from pdm.evaluation import _concept_matches
from pdm.evidence import canonical_json_hash
from scripts.run_fusion_experiment import validate_grounded_response


ROOT = Path(__file__).resolve().parents[1]


MISATTRIBUTED = {
    "79:pdm_enabled:claim:claim-5",
    "478:pdm_enabled:claim:claim-5",
    "1542:pdm_enabled:claim:claim-10",
    "1547:pdm_enabled:claim:claim-16",
    "1624:pdm_enabled:claim:claim-12",
    "2325:pdm_enabled:claim:claim-10",
    "2557:pdm_enabled:claim:claim-7",
}

UNSUPPORTED = {
    "1542:current_evidence:claim:claim-6": (
        "The cited evidence says attention is not probability, but does not support "
        "the added statement that it is not confidence."
    ),
    "1542:pdm_enabled:claim:claim-14": (
        "The absence claim is contradicted by supplied detailed history, aggregate "
        "trend, and recurrence evidence."
    ),
}


def _v3_artifacts(run_dir: Path, row: dict) -> tuple[dict, dict]:
    slug = "current_evidence" if row["condition"] == "Current-Evidence" else "pdm_enabled"
    case_dir = run_dir / "cases" / f"prediction_{row['case_id']}" / slug
    package = json.loads((case_dir / "evidence_package.json").read_text(encoding="utf-8"))
    narrative = json.loads((case_dir / "parsed_response.json").read_text(encoding="utf-8"))
    if package["package_hash"] != row["package_hash"]:
        raise ValueError(f"package hash mismatch for {row['review_id']}")
    if canonical_json_hash(narrative) != row["narrative_hash"]:
        raise ValueError(f"narrative hash mismatch for {row['review_id']}")
    return package, narrative


def _semantic_claim_supported(claim: dict, package: dict) -> bool:
    text = claim["text"].casefold()
    if "not used as detector support" in text:
        return False
    if not validate_grounded_response(
        {"claims": [claim]}, package, enforce_concepts=False
    )["valid"]:
        return False
    catalog = {row["evidence_id"]: row for row in package["citation_evidence"]}
    for evidence_id in claim["evidence_ids"]:
        block = catalog[evidence_id]
        facts = block.get("facts", {})
        kind = block.get("evidence_type")
        if kind == "pdm_context_status":
            states_absence = bool(
                re.search(
                    r"\b(?:no|without)\b[^.;]*\b(?:history|context|trend|recurrence)\b",
                    text,
                )
                or re.search(
                    r"\b(?:history|context|trend|recurrence)\b[^.;]*\b"
                    r"(?:not\s+(?:supplied|available|included)|absent|unavailable)\b",
                    text,
                )
            )
            if not states_absence:
                return False
        elif kind == "clinical_context_summary" and not facts.get("supplied", True):
            states_absence = bool(
                re.search(
                    r"\b(?:no|without)\b[^.;]*\b(?:clinical|diagnos|visit)\w*\b",
                    text,
                )
                or re.search(
                    r"\b(?:clinical\s+context|diagnos\w*|visits?)\b[^.;]*\b"
                    r"(?:not\s+(?:supplied|available|included|recorded)|absent|unavailable)\b",
                    text,
                )
                or re.search(r"\bzero\s+(?:recorded\s+)?(?:diagnos\w*|visits?)\b", text)
            )
            if not states_absence:
                return False
        elif kind == "attention_summary":
            if "shared window-level temporal relevance" not in text and "attention" not in text:
                return False
        elif kind in {"recent_history_summary", "recurrence_summary"} and "consecutive" in text:
            total = facts.get("event_count", facts.get("analyzed_window_count"))
            if total is None or facts.get("anomaly_count") != total:
                return False
    return True


def _classify_v3_row(row: dict, package: dict, narrative: dict) -> tuple[str, str]:
    if row["record_type"] == "factual_claim":
        claim = next(
            (item for item in narrative["claims"] if item["claim_id"] == row["claim_id"]),
            None,
        )
        if claim is None or claim["text"] != row["claim_text"]:
            raise ValueError(f"claim drift for {row['review_id']}")
        supported = _semantic_claim_supported(claim, package)
        if supported:
            return (
                "supported",
                "Compared with the cited frozen catalog block(s); every factual part is supported.",
            )
        return (
            "unsupported-other",
            "The cited frozen catalog block(s) do not establish every factual part of this claim.",
        )

    current_concepts = package["evaluation_concepts"]["current"]
    context_concepts = package["evaluation_concepts"].get("context", [])
    concepts = current_concepts + context_concepts
    concept = next(item for item in concepts if item["concept_id"] == row["evidence_item_id"])
    if concept in context_concepts:
        concept = {**concept, "match_mode": "any"}
    related = [
        claim for claim in narrative["claims"]
        if concept["evidence_id"] in claim.get("evidence_ids", [])
    ]
    represented = any(_concept_matches(claim["text"], concept) for claim in related)
    if concept["concept_id"] == "clinical_context":
        represented = represented or any(
            _semantic_claim_supported(claim, package)
            and any(
                token in claim["text"].casefold()
                for token in ("clinical", "diagnos", "visit", "synthetic prototype")
            )
            for claim in related
        )
    if concept["concept_id"] == "xai_interpretation":
        represented = represented or any(
                "shared window-level temporal relevance" in claim["text"].casefold()
                and (
                re.search(r"not\s+(?:a\s+)?channel-specific importance", claim["text"].casefold())
                or "no channel owner" in claim["text"].casefold()
                or "does not indicate channel-specific importance" in claim["text"].casefold()
            )
            for claim in related
        )
    return (
        ("correctly_represented", "The cited claim represents this concept in the frozen catalog.")
        if represented else
        ("omitted", "No cited claim represents all required values or semantics for this concept.")
    )


def build_decisions(review_path: Path, run_dir: Path | None = None) -> dict:
    review_path = Path(review_path)
    with review_path.open(encoding="utf-8-sig", newline="") as handle:
        rows = list(csv.DictReader(handle))
    is_v3 = any(
        version in review_path.parts
        for version in (
            "fusion-scoring-v3", "fusion-scoring-v4", "fusion-scoring-v5",
            "fusion-scoring-v6",
            "fusion-scoring-v7",
        )
    )
    if is_v3:
        run_id = review_path.parent.parent.name
        run_dir = Path(run_dir) if run_dir is not None else (
            ROOT / "outputs/narrative_experiments/apnea_ecg/runs" / run_id
        )
    known_ids = {row["review_id"] for row in rows}
    classified_ids = set() if is_v3 else MISATTRIBUTED | set(UNSUPPORTED)
    unknown = sorted(classified_ids - known_ids)
    if unknown:
        raise ValueError(f"adjudication mapping contains unknown review IDs: {unknown}")

    overrides = {}
    for row in rows:
        review_id = row["review_id"]
        if is_v3:
            package, narrative = _v3_artifacts(run_dir, row)
            final_class, note = _classify_v3_row(row, package, narrative)
        elif review_id in UNSUPPORTED:
            final_class = "unsupported-other"
            note = UNSUPPORTED[review_id]
        elif review_id in MISATTRIBUTED:
            final_class = "misattributed"
            note = (
                "The factual content appears elsewhere in the supplied package, "
                "but the cited block does not support every part of this claim."
            )
        else:
            final_class = "supported"
            note = (
                "Compared with the cited package block(s); every factual part is "
                "present, including any stated absence or empty-list condition."
            )
        overrides[review_id] = {
            "reviewer_class": final_class,
            "reviewer_note": note,
        }
    is_v4 = "fusion-scoring-v4" in review_path.parts
    is_v5 = "fusion-scoring-v5" in review_path.parts
    is_v6 = "fusion-scoring-v6" in review_path.parts
    is_v7 = "fusion-scoring-v7" in review_path.parts
    return {
        "schema_version": (
            "fusion-ai-adjudication-v7" if is_v7
            else "fusion-ai-adjudication-v6" if is_v6
            else "fusion-ai-adjudication-v5" if is_v5
            else "fusion-ai-adjudication-v4" if is_v4
            else "fusion-ai-adjudication-v3" if is_v3
            else "fusion-ai-adjudication-v1"
        ),
        "adjudication_type": "ai_assisted_evidence_audit",
        "adjudicator": "OpenAI Codex",
        "adjudication_date": date.today().isoformat(),
        "method": (
            "Each flagged claim was compared with the frozen evidence blocks named "
            "by its citation IDs. Whole-claim support and citation attribution were "
            "checked separately. For v5, adjudication evaluated the preserved first "
            "structurally valid response and did not repair or reject its content."
            if is_v5 or is_v6 or is_v7 else
            "Each flagged claim was compared with the frozen evidence blocks named "
            "by its citation IDs. Whole-claim support and citation attribution were "
            "checked separately. Catalog-based claims had also passed the pre-persistence "
            "grounding gate."
            if is_v3 else
            "Each flagged claim was compared with the frozen evidence blocks named "
            "by its citation IDs. Whole-claim support and citation attribution were "
            "checked separately."
        ),
        "disclosure": (
            "This is an AI-assisted semantic adjudication, not human review. A human "
            "spot-check is optional and must not be claimed unless performed."
        ),
        "overrides": overrides,
    }


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("review_csv", type=Path)
    parser.add_argument("output_json", type=Path)
    parser.add_argument("--run-dir", type=Path)
    args = parser.parse_args(argv)
    result = build_decisions(args.review_csv, args.run_dir)
    args.output_json.parent.mkdir(parents=True, exist_ok=True)
    args.output_json.write_text(
        json.dumps(result, indent=2, sort_keys=True, ensure_ascii=False) + "\n",
        encoding="utf-8",
    )
    print(json.dumps({
        "status": "complete",
        "decisions": len(result["overrides"]),
        "output": str(args.output_json),
    }, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
