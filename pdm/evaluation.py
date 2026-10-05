"""Reproducible evidence-grounding evaluation for the paired experiment."""

from __future__ import annotations

import csv
import json
import math
import re
import statistics
from pathlib import Path
from typing import Any


METRICS = (
    "coverage", "unsupported_claim_rate", "omission_rate",
    "incorrect_representation_rate", "misattribution_rate",
    "fabrication_rate", "factual_fidelity", "context_utilization",
)


def _item(evidence_id: str, kind: str, path: str, field: str, value: Any) -> dict:
    result = {
        "id": f"{evidence_id}::{field}", "evidence_id": evidence_id,
        "kind": kind, "package_path": path, "field": field, "value": value,
        "supplied": True, "relevant": True,
    }
    return result


def _append_fields(rows: list[dict], block: dict, kind: str, path: str, fields) -> None:
    evidence_id = block.get("evidence_id")
    if not evidence_id:
        return
    for field in fields:
        if field in block and block[field] is not None:
            rows.append(_item(evidence_id, kind, path, field, block[field]))


def _scalar_values(value: Any) -> list[Any]:
    """Return the scalar facts contained in one evidence block."""

    values: list[Any] = []
    if isinstance(value, dict):
        for key, child in value.items():
            if key != "evidence_id":
                values.extend(_scalar_values(child))
    elif isinstance(value, list):
        for child in value:
            values.extend(_scalar_values(child))
    elif value is not None:
        values.append(value)
    return values


def collect_evidence_blocks(package: dict) -> dict[str, dict]:
    """Create the shared registry used by generation validation and scoring.

    Coverage is intentionally handled by :func:`build_evidence_checklist`; this
    registry is broader because a narrative may legitimately cite provenance,
    identity, or the explicit absence of PDM context without those facts becoming
    part of the coverage denominator.
    """

    registry: dict[str, dict] = {}

    def add(
        evidence_id: str,
        path: str,
        value: Any,
        scope: str,
        citation_kind: str = "native",
    ) -> None:
        if evidence_id in registry and registry[evidence_id]["path"] != path:
            label = "context " if scope == "context" else ""
            raise ValueError(f"duplicate {label}evidence IDs: {evidence_id}")
        registry[evidence_id] = {
            "evidence_id": evidence_id,
            "path": path,
            "scope": scope,
            "citation_kind": citation_kind,
            "values": _scalar_values(value),
        }

    def visit(value: Any, path: str, scope: str) -> None:
        if isinstance(value, dict):
            evidence_id = value.get("evidence_id")
            if isinstance(evidence_id, str) and evidence_id:
                add(evidence_id, path, value, scope)
            for key, child in value.items():
                if (
                    key != "evidence_id"
                    and isinstance(child, str)
                    and child
                    and (str(key).endswith("evidence_id") or (str(key).endswith("_id") and ":" in child))
                ):
                    add(child, f"{path}.{key}", value, scope, "alias")
                visit(child, f"{path}.{key}" if path else str(key), scope)
        elif isinstance(value, list):
            for index, child in enumerate(value):
                visit(child, f"{path}[{index}]", scope)

    # V3 generation exposes only this compact catalog to the model. Treat it as
    # the authoritative scoring registry so legacy copies of the same IDs do not
    # create duplicate entries or accidentally broaden what a citation supports.
    citation_catalog = package.get("citation_evidence", [])
    if citation_catalog:
        for index, block in enumerate(citation_catalog):
            if not isinstance(block, dict) or not block.get("evidence_id"):
                continue
            add(
                str(block["evidence_id"]),
                f"citation_evidence[{index}]",
                block.get("facts", {}),
                str(block.get("scope", "package")),
                "native",
            )
        return registry

    current = package.get("current_evidence", {})
    visit(current, "current_evidence", "current")
    context = package.get("pdm_context", {})
    visit(context, "pdm_context", "context")
    # Identity has no native evidence_id in the frozen packages, but both aliases
    # were explicitly offered to the LLM in the evidence instructions.
    identity = current.get("identity", {}) if isinstance(current, dict) else {}
    if isinstance(identity, dict) and identity:
        add("identity", "current_evidence.identity", identity, "package", "alias")
        prediction_id = identity.get("prediction_id")
        if prediction_id is not None:
            add(
                f"window:{prediction_id}:identity",
                "current_evidence.identity",
                identity,
                "package",
                "alias",
            )

    # This virtual block makes an explicit statement such as "PDM context was
    # not included" citable. Concrete history blocks remain separate.
    included = bool(context.get("included")) if isinstance(context, dict) else False
    add("pdm_context", "pdm_context", context or {"included": included}, "package", "virtual")
    clinical_context = context.get("clinical_context", {}) if isinstance(context, dict) else {}
    if isinstance(clinical_context, dict):
        add(
            "clinical_context",
            "pdm_context.clinical_context",
            clinical_context,
            "package",
            "virtual",
        )

    # Retain the compact explicit-checklist contract used by fixtures and tools.
    for index, row in enumerate(package.get("checklist", [])):
        if not isinstance(row, dict):
            continue
        evidence_id = str(row.get("evidence_id", row.get("id", "")))
        if evidence_id:
            add(evidence_id, f"checklist[{index}]", row.get("value"), "current")
    return registry


def build_evidence_checklist(
    package: dict, *, include_context: bool = False
) -> list[dict]:
    """Build the coverage denominator from supplied evaluation concepts.

    Legacy experiments compare the shared current-event concepts only. The v5
    free-generation experiment additionally includes each supplied PDM context
    concept, so extra context expands both the numerator opportunity and the
    denominator rather than creating free coverage credit.
    """

    concepts = package.get("evaluation_concepts")
    if isinstance(concepts, dict) and isinstance(concepts.get("current"), list):
        rows = []
        selected = list(concepts["current"])
        if include_context:
            selected.extend(concepts.get("context", []))
        for concept in selected:
            if not isinstance(concept, dict):
                continue
            concept_id = str(concept["concept_id"])
            rows.append({
                "id": concept_id,
                "concept_id": concept_id,
                "evidence_id": str(concept["evidence_id"]),
                "kind": "concept",
                "package_path": "evaluation_concepts.current",
                "field": concept_id,
                "value": list(concept.get("match_values", [])),
                "match_values": list(concept.get("match_values", [])),
                "match_fields": list(concept.get("match_fields", [])),
                "match_mode": (
                    "any" if concept in concepts.get("context", []) else "all"
                ),
                "supplied": True,
                "relevant": True,
            })
        identifiers = [row["id"] for row in rows]
        if len(identifiers) != len(set(identifiers)):
            raise ValueError("evidence checklist contains duplicate concept IDs")
        return rows

    if isinstance(package.get("checklist"), list):
        return [{
            "evidence_id": row.get("evidence_id", row["id"]),
            "kind": row.get("kind", "explicit"),
            "package_path": row.get("package_path", "checklist"),
            "field": row.get("field", "value"), "supplied": True,
            "relevant": True, **row,
        } for row in package["checklist"]]
    current = package.get("current_evidence")
    if not isinstance(current, dict):
        return []
    # Backwards-compatible fallback for the small contract fixture.
    if not any(isinstance(value, dict) for value in current.values()):
        return [_item("current_evidence", "current", f"current_evidence.{field}", field, value)
                for field, value in current.items()]

    rows: list[dict] = []
    _append_fields(rows, current.get("event", {}), "current_event", "current_evidence.event", ("start_time", "duration_seconds"))
    _append_fields(rows, current.get("model_evidence", {}), "detector", "current_evidence.model_evidence", (
        "predicted_class", "reconstruction_error", "anomaly_threshold", "anomaly_margin", "normalized_margin",
    ))
    for name in ("signal_evidence", "reconstruction_evidence", "residual_evidence"):
        block = current.get(name, {})
        if block.get("available"):
            _append_fields(rows, block, name, f"current_evidence.{name}", ("stored_value_count", "available"))
    attention = current.get("attention_evidence", {})
    if attention.get("available"):
        _append_fields(rows, attention, "attention", "current_evidence.attention_evidence", ("peak_time", "maximum", "semantics"))
    xai = current.get("xai_evidence", {})
    if xai.get("available"):
        _append_fields(rows, xai, "xai", "current_evidence.xai_evidence", ("channel_semantics",))
        for explanation in xai.get("explanations", []):
            if explanation.get("description"):
                rows.append(_item(xai["evidence_id"], "xai", "current_evidence.xai_evidence.explanations",
                                  f"explanation_{explanation['explanation_id']}", explanation["description"]))

    identifiers = [row["id"] for row in rows]
    if len(identifiers) != len(set(identifiers)):
        raise ValueError("evidence checklist contains duplicate item IDs")
    return rows


def _value_matches(text: str, value: Any) -> bool:
    lowered = text.casefold()
    if isinstance(value, bool):
        return str(value).casefold() in lowered
    if isinstance(value, (int, float)) and not isinstance(value, bool):
        if not math.isfinite(float(value)):
            return False
        candidates = {str(value), f"{float(value):.2f}", f"{float(value):.3f}", f"{float(value):.4f}"}
        for candidate in candidates:
            candidate = candidate.rstrip("0").rstrip(".") if "." in candidate else candidate
            if candidate and re.search(rf"(?<![\d.]){re.escape(candidate)}(?!\d)", text):
                return True
        return False
    candidate = str(value).casefold()
    return bool(candidate) and candidate in lowered


def _concept_matches(text: str, item: dict) -> bool:
    values = item.get("match_values")
    if not isinstance(values, list):
        values = item.get("value")
        values = values if isinstance(values, list) else [values]
    matches = [_value_matches(text, value) for value in values]
    return bool(matches) and (
        any(matches) if item.get("match_mode") == "any" else all(matches)
    )


def score_exact_facts(
    package: dict, narrative: dict, *, include_context_in_coverage: bool = False
) -> dict:
    """Propose exact classifications and case metrics without semantic guessing."""

    checklist = build_evidence_checklist(
        package, include_context=include_context_in_coverage
    )
    claims = narrative.get("claims", []) if isinstance(narrative, dict) else []
    if not isinstance(claims, list):
        raise ValueError("narrative claims must be a list")
    registry = collect_evidence_blocks(package)
    for item in checklist:
        registry.setdefault(item["id"], {
            "evidence_id": item["id"],
            "path": item["package_path"],
            "scope": "current",
            "citation_kind": "native",
            "values": [item["value"]],
        })
        parent = registry.setdefault(item["evidence_id"], {
                "evidence_id": item["evidence_id"],
                "path": item["package_path"],
                "scope": "current",
                "citation_kind": "native",
                "values": [],
            })
        if item["value"] not in parent["values"]:
            parent["values"].append(item["value"])
    by_evidence: dict[str, list[dict]] = {}
    for item in checklist:
        by_evidence.setdefault(item["evidence_id"], []).append(item)
        by_evidence.setdefault(item["id"], []).append(item)
    concepts = package.get("evaluation_concepts")
    is_v3 = isinstance(concepts, dict) and concepts.get("schema_version") == "fusion-evaluation-concepts-v3"
    if is_v3:
        context_ids = {
            str(row["evidence_id"])
            for row in concepts.get("context", [])
            if isinstance(row, dict) and row.get("evidence_id")
        }
    else:
        context_ids = {
            evidence_id for evidence_id, block in registry.items()
            if block["scope"] == "context"
        }
    assessments, correctly, incorrectly = [], set(), set()
    cited_evidence_ids: set[str] = set()
    cited_context_ids: set[str] = set()
    for index, claim in enumerate(claims, start=1):
        text = str(claim.get("text", ""))
        refs = claim.get("evidence_ids", [])
        refs = refs if isinstance(refs, list) else []
        cited_items = []
        for ref in refs:
            ref = str(ref)
            cited_items.extend(by_evidence.get(ref, []))
        cited_items = list({item["id"]: item for item in cited_items}.values())
        cited_blocks = [registry[str(ref)] for ref in refs if str(ref) in registry]
        matched_cited_blocks = [
            block for block in cited_blocks
            if any(_value_matches(text, value) for value in block["values"])
        ]
        matched_cited_items = [item for item in cited_items if _concept_matches(text, item)]
        cited_ids = {str(ref) for ref in refs}
        cited_evidence_ids.update(cited_ids)
        matched_other = [
            block for evidence_id, block in registry.items()
            if evidence_id not in cited_ids
            and any(_value_matches(text, value) for value in block["values"])
        ]
        issues, requires_review = [], False
        if not refs or any(str(ref) not in registry for ref in refs):
            proposed, issues, requires_review = "unsupported-other", ["invalid_evidence_id"], True
        elif matched_cited_blocks:
            proposed = "supported"
            if is_v3:
                correctly.update(item["id"] for item in matched_cited_items)
            else:
                correctly.update(item["id"] for item in matched_cited_items)
            cited_context_ids.update(str(ref) for ref in refs if str(ref) in context_ids)
            requires_review = any(token in text.casefold() for token in (" because ", " therefore ", "suggests ", "indicates "))
            if any(block["citation_kind"] != "native" for block in matched_cited_blocks):
                issues.append("alias_citation_requires_review")
                requires_review = True
        elif matched_other:
            proposed, issues, requires_review = "misattributed", ["fact_matches_different_evidence"], True
            incorrectly.update(item["id"] for item in cited_items)
        else:
            proposed, issues, requires_review = "unsupported-other", ["semantic_or_unmatched_claim"], True
        assessments.append({
            "claim_id": claim.get("claim_id", f"claim-{index}"), "section": claim.get("section"),
            "text": text, "evidence_ids": [str(ref) for ref in refs], "proposed_class": proposed,
            "issue_codes": issues, "requires_review": requires_review,
        })
    evidence_assessments = []
    for item in checklist:
        status = ("incorrectly_represented" if item["id"] in incorrectly else
                  "correctly_represented" if item["id"] in correctly else "omitted")
        evidence_assessments.append({
            **item,
            "proposed_status": status,
            "semantic_review_required": (
                is_v3 and status != "correctly_represented"
                and item["evidence_id"] in cited_evidence_ids
            ),
        })
    total = len(checklist)
    correct_count = sum(row["proposed_status"] == "correctly_represented" for row in evidence_assessments)
    incorrect_count = sum(row["proposed_status"] == "incorrectly_represented" for row in evidence_assessments)
    omitted_count = sum(row["proposed_status"] == "omitted" for row in evidence_assessments)
    counts = {name: sum(row["proposed_class"] == name for row in assessments)
              for name in ("supported", "unsupported-other", "misattributed", "fabricated")}
    factual_count = len(assessments)
    unsupported = counts["unsupported-other"] + counts["misattributed"] + counts["fabricated"]
    context_count = len(context_ids)
    result = {
        "checklist": checklist, "evidence_assessments": evidence_assessments, "claim_assessments": assessments,
        "relevant_evidence_count": total, "correctly_represented_count": correct_count,
        "incorrectly_represented_count": incorrect_count, "omitted_count": omitted_count,
        "factual_claim_count": factual_count, "supported_claim_count": counts["supported"],
        "unsupported_other_count": counts["unsupported-other"], "misattributed_count": counts["misattributed"],
        "fabricated_count": counts["fabricated"], "coverage": correct_count / total if total else None,
        "omission_rate": omitted_count / total if total else None,
        "incorrect_representation_rate": incorrect_count / total if total else None,
        "unsupported_claim_rate": unsupported / factual_count if factual_count else None,
        "misattribution_rate": counts["misattributed"] / factual_count if factual_count else None,
        "fabrication_rate": counts["fabricated"] / factual_count if factual_count else None,
        "factual_fidelity": counts["supported"] / factual_count if factual_count else None,
        "coverage_scope": (
            "all_supplied_concepts"
            if is_v3 and include_context_in_coverage
            else "shared_current_concepts" if is_v3
            else "shared_current_evidence"
        ),
        "context_relevant_evidence_count": context_count,
        "context_cited_evidence_count": len(cited_context_ids),
        "context_utilization": (
            len(cited_context_ids) / context_count if context_count else None
        ),
        "format_failure": factual_count == 0, "zero_claim_failure": factual_count == 0,
    }
    if is_v3:
        result["context_evidence_ids"] = sorted(context_ids)
    return result


REVIEW_FIELDS = (
    "review_id", "record_type", "case_id", "condition", "claim_id", "claim_text",
    "evidence_item_id", "evidence_ids", "proposed_class", "requires_review", "issue_codes",
    "package_hash", "narrative_hash", "reviewer_class", "reviewer_note",
)


def export_semantic_review(records: list[dict], path: Path) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    ordered = sorted(records, key=lambda row: (
        int(row["case_id"]), 0 if row["condition"] == "Current-Evidence" else 1,
        row.get("record_type", ""), row.get("claim_id", ""), row.get("evidence_item_id", ""),
    ))
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=REVIEW_FIELDS, extrasaction="ignore")
        writer.writeheader()
        for row in ordered:
            value = {field: row.get(field, "") for field in REVIEW_FIELDS}
            for field in ("evidence_ids", "issue_codes"):
                if isinstance(value[field], list):
                    value[field] = json.dumps(value[field], ensure_ascii=False)
            writer.writerow(value)


def aggregate_table7(approved_records: list[dict]) -> dict:
    grouped: dict[str, list[dict]] = {}
    for record in approved_records:
        grouped.setdefault(record["condition"], []).append(record)
    result = {}
    for condition, records in grouped.items():
        summary = {"case_count": len(records), "case_values": records}
        for metric in METRICS:
            values = [float(row[metric]) for row in records if row.get(metric) is not None]
            summary[metric] = {
                "mean": statistics.fmean(values) if values else None,
                "sd": statistics.stdev(values) if len(values) > 1 else (0.0 if values else None),
                "n": len(values),
            }
        result[condition] = summary
    return result
