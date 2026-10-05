"""Hybrid deterministic/LLM narrative generation for the v4 experiment.

All measured facts are rendered from the frozen evidence catalog.  The LLM is
allowed to contribute one optional synthesis claim, which can be omitted
without making the factual narrative incomplete.
"""

from __future__ import annotations

import json
import re
import time
from pathlib import Path

from .llm import BASE_URL, MODEL_ID, NARRATIVE_SECTIONS, render_narrative


ROOT = Path(__file__).resolve().parents[1]
EXPERIMENT_ROOT = ROOT / "outputs" / "narrative_experiments" / "apnea_ecg"
PROMPT_V4_PATH = EXPERIMENT_ROOT / "prompt_v4.txt"
POLICY_V4_PATH = EXPERIMENT_ROOT / "generation_policy_v4.json"


def _catalog(package: dict) -> dict[str, dict]:
    rows = package.get("citation_evidence", [])
    return {str(row["evidence_type"]): row for row in rows}


def _claim(claim_id: str, section: str, text: str, block: dict) -> dict:
    return {
        "claim_id": claim_id,
        "section": section,
        "text": text,
        "evidence_ids": [str(block["evidence_id"])],
    }


def build_deterministic_claims(package: dict) -> list[dict]:
    """Build the complete factual layer directly from one frozen package."""

    by_type = _catalog(package)
    event = by_type["current_event_summary"]
    detector = by_type["detector_summary"]
    signal = by_type.get("signal_summary")
    attention = by_type.get("attention_summary")
    ef, df = event["facts"], detector["facts"]
    claims = [
        _claim(
            "fact-event-timing", "current_event",
            f"The current event starts at {ef['start_time']} and lasts {ef['duration_seconds']} seconds.",
            event,
        ),
        _claim(
            "fact-detector-decision", "model_and_xai_evidence",
            f"The detector classified the current window as {df['predicted_class']}.",
            detector,
        ),
        _claim(
            "fact-reconstruction-comparison", "model_and_xai_evidence",
            f"The reconstruction error is {df['reconstruction_error']} and the anomaly threshold is {df['anomaly_threshold']}.",
            detector,
        ),
        _claim(
            "fact-detector-margin", "model_and_xai_evidence",
            f"The detector margin is {df['anomaly_margin']} and the normalized margin is {df['normalized_margin']}; these are not probabilities or calibrated confidence.",
            detector,
        ),
    ]
    if signal and signal["facts"].get("available"):
        sf = signal["facts"]
        claims.append(_claim(
            "fact-signal", "model_and_xai_evidence",
            f"The primary signal channel is {sf['primary_channel']} with {sf['primary_sample_count']} samples.",
            signal,
        ))
    if attention and attention["facts"].get("available"):
        af = attention["facts"]
        claims.extend([
            _claim(
                "fact-attention-result", "model_and_xai_evidence",
                f"Temporal attention peaks at {af['peak_time']} with a maximum value of {af['maximum']}.",
                attention,
            ),
            _claim(
                "fact-xai-meaning", "model_and_xai_evidence",
                f"Attention semantics are: {af['semantics']}.",
                attention,
            ),
        ])

    history = by_type.get("recent_history_summary")
    if history and history.get("scope") == "context":
        f = history["facts"]
        anomaly_count = "zero" if f["anomaly_count"] == 0 else str(f["anomaly_count"])
        claims.append(_claim(
            "fact-recent-history", "previous_monitoring_history",
            f"The recent history contains {f['event_count']} events: {anomaly_count} anomaly and {f['normal_count']} normal decisions, from {f['earliest_time']} through {f['most_recent_time']}; reconstruction error ranges from {f['minimum_reconstruction_error']} to {f['maximum_reconstruction_error']}.",
            history,
        ))
    aggregate = by_type.get("aggregate_trend_summary")
    if aggregate and aggregate.get("scope") == "context":
        f = aggregate["facts"]
        claims.append(_claim(
            "fact-aggregate-trend", "previous_monitoring_history",
            f"Across {f['bucket_count']} aggregate buckets, mean reconstruction error {f['endpoint_direction']} from {f['first_mean_reconstruction_error']} at {f['first_bucket']} to {f['last_mean_reconstruction_error']} at {f['last_bucket']}.",
            aggregate,
        ))
    recurrence = by_type.get("recurrence_summary")
    if recurrence and recurrence.get("scope") == "context":
        f = recurrence["facts"]
        claims.append(_claim(
            "fact-recurrence", "previous_monitoring_history",
            f"Before the current event, the recurrence summary covers {f['analyzed_window_count']} analyzed windows with {f['anomaly_count']} anomalies and an anomaly rate of {f['anomaly_rate']}.",
            recurrence,
        ))
    clinical = by_type.get("clinical_context_summary")
    if clinical and clinical.get("scope") == "context":
        f = clinical["facts"]
        details = ""
        if f.get("diagnoses"):
            diagnosis = f["diagnoses"][0]
            details = (
                f" The recorded diagnosis is {diagnosis['name']} with status "
                f"{diagnosis['status']}."
            )
        claims.append(_claim(
            "fact-clinical-context", "recorded_clinical_context",
            f"Synthetic prototype clinical context is supplied with {f['visit_count']} visit and {f['diagnosis_count']} diagnosis record.{details}",
            clinical,
        ))
    return claims


def build_hybrid_narrative(package: dict, optional_claim: dict | None = None) -> dict:
    claims = build_deterministic_claims(package)
    if optional_claim is not None:
        claims.append(optional_claim)
    return {
        "schema_version": "fusion_narrative_v4",
        "source_schema_version": "fusion_hybrid_v4",
        "claims": claims,
        "narrative_sections": render_narrative(claims),
    }


def load_generation_contract() -> tuple[str, dict]:
    import hashlib

    prompt = PROMPT_V4_PATH.read_text(encoding="utf-8")
    policy = json.loads(POLICY_V4_PATH.read_text(encoding="utf-8"))
    if policy.get("prompt_sha256") != hashlib.sha256(prompt.encode("utf-8")).hexdigest():
        raise ValueError("frozen v4 prompt hash does not match generation policy")
    if policy.get("model") != MODEL_ID or policy.get("base_url") != BASE_URL:
        raise ValueError("v4 model endpoint differs from the approved contract")
    return prompt, policy


def build_synthesis_request(package: dict, correction_issues: list[str] | None = None) -> dict:
    prompt, policy = load_generation_contract()
    payload = {
        "condition": "PDM-Enabled" if package.get("pdm_context", {}).get("included") else "Current-Evidence",
        "package_hash": package.get("package_hash"),
        "citation_evidence": package.get("citation_evidence", []),
    }
    settings = policy["generation_settings"]
    messages = [
        {"role": "system", "content": prompt},
        {"role": "user", "content": json.dumps(payload, sort_keys=True, separators=(",", ":"), ensure_ascii=False)},
    ]
    if correction_issues:
        messages.append({
            "role": "user",
            "content": json.dumps({
                "instruction": "Replace the whole response and correct the listed issues using only exact supplied evidence.",
                "validation_issue_codes": sorted(set(correction_issues)),
            }, sort_keys=True, separators=(",", ":")),
        })
    return {
        "model": policy["model"], "messages": messages,
        "temperature": settings["temperature"], "top_p": settings["top_p"],
        "max_tokens": settings["max_tokens"],
        "extra_body": {"chat_template_kwargs": settings["chat_template_kwargs"]},
    }


def parse_synthesis(raw_text: str) -> dict:
    clean = re.sub(r"^```(?:json)?\s*|\s*```$", "", str(raw_text).strip(), flags=re.I).strip()
    value = json.loads(clean)
    if not isinstance(value, dict) or set(value) != {"schema_version", "claim"}:
        raise ValueError("synthesis response has the wrong top-level schema")
    if value.get("schema_version") != "fusion_summary_v4":
        raise ValueError("synthesis response has the wrong schema version")
    claim = value.get("claim")
    if not isinstance(claim, dict) or set(claim) != {"claim_id", "section", "text", "evidence_ids"}:
        raise ValueError("synthesis claim has the wrong schema")
    if claim.get("claim_id") != "llm-summary" or claim.get("section") != "evidence_grounded_summary":
        raise ValueError("synthesis claim identity or section is invalid")
    if not isinstance(claim.get("text"), str) or not claim["text"].strip():
        raise ValueError("synthesis text is empty")
    refs = claim.get("evidence_ids")
    if not isinstance(refs, list) or not refs or not all(isinstance(ref, str) and ref for ref in refs):
        raise ValueError("synthesis evidence IDs are invalid")
    return claim


def _scalar_values(value):
    if isinstance(value, dict):
        for child in value.values():
            yield from _scalar_values(child)
    elif isinstance(value, list):
        for child in value:
            yield from _scalar_values(child)
    elif value is not None and not isinstance(value, bool):
        yield value


def validate_optional_synthesis(parsed: dict, package: dict, *, base_validation_fn) -> dict:
    """Require every cited block to contribute an explicit, meaningful fact."""

    base = base_validation_fn(parsed, package, enforce_concepts=False)
    issues = list(base.get("issue_codes", []))
    catalog = {row["evidence_id"]: row for row in package.get("citation_evidence", [])}
    for claim in parsed.get("claims", []):
        text = str(claim.get("text", ""))
        lowered = text.casefold()
        for ref in claim.get("evidence_ids", []):
            block = catalog.get(ref)
            if block is None:
                continue
            facts = block.get("facts", {})
            kind = block.get("evidence_type")
            if kind == "clinical_context_summary" and not facts.get("supplied", True):
                anchored = (
                    "not supplied" in lowered
                    or "no clinical" in lowered
                    or (facts.get("diagnosis_count") == 0 and "no diagnos" in lowered)
                    or (facts.get("visit_count") == 0 and "no visit" in lowered)
                )
            else:
                anchored = False
                for value in _scalar_values(facts):
                    candidate = str(value).casefold()
                    if len(candidate) >= 4 and candidate in lowered:
                        anchored = True
                        break
                    if isinstance(value, (int, float)) and re.search(
                        rf"(?<![\d.]){re.escape(str(value))}(?!\d)", text
                    ):
                        anchored = True
                        break
            if not anchored:
                issues.append(f"{claim.get('claim_id')}:cited_block_has_no_explicit_fact:{ref}")
    return {
        **base,
        "valid": not issues,
        "issue_codes": sorted(set(issues)),
    }


def execute_optional_synthesis(client, package: dict, validation_fn, sleep_fn=time.sleep) -> dict:
    """Try once, allow one correction, then omit safely while retaining audit."""

    _, policy = load_generation_contract()
    attempts, issues = [], None
    transport = policy["transport_retry"]
    retryable = set(transport["retryable_http_statuses"])
    for response_number in (1, 2):
        request = build_synthesis_request(package, issues)
        response = None
        for transport_attempt in range(1, transport["maximum_attempts"] + 1):
            try:
                response = client.chat.completions.create(**request)
                break
            except Exception as exc:
                attempts.append({
                    "response_number": response_number, "transport_attempt": transport_attempt,
                    "transport_error": f"{type(exc).__name__}: {exc}", "raw_response": None,
                })
                status = getattr(exc, "status_code", None)
                retryable_error = status in retryable or any(
                    token in type(exc).__name__.casefold()
                    for token in ("timeout", "connection", "ratelimit")
                )
                if transport_attempt >= transport["maximum_attempts"] or not retryable_error:
                    return {"status": "omitted_transport_failure", "attempts": attempts, "accepted_claim": None}
                sleep_fn(transport["backoff_seconds"][transport_attempt - 1])
        raw = response.choices[0].message.content
        record = {
            "response_number": response_number, "transport_attempt": transport_attempt,
            "transport_error": None, "raw_response": raw,
            "response_metadata": {"id": getattr(response, "id", None), "model": getattr(response, "model", None)},
        }
        attempts.append(record)
        try:
            claim = parse_synthesis(raw)
            parsed = {"claims": [claim]}
            validation = validation_fn(parsed, package, enforce_concepts=False)
            record["grounding_validation"] = validation
            if validation.get("valid"):
                return {"status": "accepted", "attempts": attempts, "accepted_claim": claim}
            issues = list(validation.get("issue_codes", []))
        except (ValueError, json.JSONDecodeError) as exc:
            record["parse_error"] = f"{type(exc).__name__}: {exc}"
            issues = ["invalid_summary_json_or_schema"]
    return {
        "status": "omitted_after_correction", "attempts": attempts,
        "accepted_claim": None, "validation_issue_codes": issues or [],
    }
