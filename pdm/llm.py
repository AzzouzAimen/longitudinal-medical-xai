"""Evidence-grounded Apnea narrative experiment.

This module deliberately keeps retrieval, prompting, and persistence small.  It
does not alter the anomaly models or the professor-provided database schema.
"""
import argparse
import hashlib
import json
import os
import re
import time
from decimal import Decimal
from datetime import date, datetime, timezone
from pathlib import Path

import requests
from dotenv import load_dotenv

from .db import connect
from .evidence import build_paired_packages

MODEL_ID = "deepseek-ai/DeepSeek-V4-Flash-0731"
BASE_URL = "https://api.featherless.ai/v1"
ROOT = Path(__file__).resolve().parents[1]
EXPERIMENT_ROOT = ROOT / "outputs" / "narrative_experiments" / "apnea_ecg"
PROMPT_PATH = EXPERIMENT_ROOT / "prompt_v1.txt"
POLICY_PATH = EXPERIMENT_ROOT / "generation_policy_v2.json"
PROMPT_V3_PATH = EXPERIMENT_ROOT / "prompt_v3.txt"
POLICY_V3_PATH = EXPERIMENT_ROOT / "generation_policy_v3.json"
PROMPT_V5_PATH = EXPERIMENT_ROOT / "prompt_v5.txt"
POLICY_V5_PATH = EXPERIMENT_ROOT / "generation_policy_v5.json"
PROMPT_V6_PATH = EXPERIMENT_ROOT / "prompt_v6.txt"
POLICY_V6_PATH = EXPERIMENT_ROOT / "generation_policy_v6.json"
PROMPT_V7_PATH = EXPERIMENT_ROOT / "prompt_v7.txt"
POLICY_V7_PATH = EXPERIMENT_ROOT / "generation_policy_v7.json"
CONTEXT_FILE = Path(__file__).resolve().parents[1] / "config" / "synthetic_clinical_context.json"
SYNTHETIC_PROVENANCE = "synthetic_prototype"

def _jsonable(v):
    if isinstance(v, (datetime, date)): return v.isoformat()
    if isinstance(v, Decimal): return float(v)
    return v

def validate_synthetic_context(records):
    """Validate the frozen prototype fixture before any database writes."""
    if not isinstance(records, list) or len(records) < 3:
        raise ValueError("synthetic context requires at least three patient records")
    patient_codes = []
    visit_count = diagnosis_count = suspected_count = 0
    for rec in records:
        if not isinstance(rec, dict) or rec.get("provenance") != SYNTHETIC_PROVENANCE:
            raise ValueError("patient provenance must be synthetic_prototype")
        patient_code = rec.get("patient_code")
        if not isinstance(patient_code, str) or not patient_code.startswith("APNEA_"):
            raise ValueError("synthetic context patient_code must identify an Apnea patient")
        if patient_code in patient_codes:
            raise ValueError(f"duplicate synthetic patient: {patient_code}")
        patient_codes.append(patient_code)
        visits = rec.get("clinical_visits")
        if not isinstance(visits, list) or not visits:
            raise ValueError(f"{patient_code} requires at least one clinical visit")
        for visit in visits:
            if not isinstance(visit, dict) or visit.get("provenance") != SYNTHETIC_PROVENANCE:
                raise ValueError(f"visit provenance must be synthetic_prototype: {patient_code}")
            if SYNTHETIC_PROVENANCE not in str(visit.get("notes", "")):
                raise ValueError(f"visit notes must visibly mark synthetic_prototype: {patient_code}")
            if not visit.get("visit_date") or not visit.get("visit_type") or not visit.get("status"):
                raise ValueError(f"visit date, type, and status are required: {patient_code}")
            symptoms = visit.get("symptoms", [])
            if not isinstance(symptoms, list):
                raise ValueError(f"visit symptoms must be a list: {patient_code}")
            for symptom in symptoms:
                if not isinstance(symptom, dict) or symptom.get("provenance") != SYNTHETIC_PROVENANCE:
                    raise ValueError(f"symptom provenance must be synthetic_prototype: {patient_code}")
                if symptom.get("status") not in {"reported", "not_reported"} or not symptom.get("name"):
                    raise ValueError(f"symptom name/status is invalid: {patient_code}")
            visit_count += 1
        diagnoses = rec.get("diagnoses", [])
        if not isinstance(diagnoses, list):
            raise ValueError(f"diagnoses must be a list: {patient_code}")
        for diagnosis in diagnoses:
            if not isinstance(diagnosis, dict) or diagnosis.get("provenance") != SYNTHETIC_PROVENANCE:
                raise ValueError(f"diagnosis provenance must be synthetic_prototype: {patient_code}")
            if diagnosis.get("status") != "suspected":
                raise ValueError(f"synthetic diagnoses must remain suspected: {patient_code}")
            if SYNTHETIC_PROVENANCE not in str(diagnosis.get("notes", "")):
                raise ValueError(f"diagnosis notes must visibly mark synthetic_prototype: {patient_code}")
            if not diagnosis.get("diagnosis_name"):
                raise ValueError(f"diagnosis name is required: {patient_code}")
            diagnosis_count += 1
            suspected_count += 1
    if not any(code.startswith("APNEA_a") for code in patient_codes) or not any(
        code.startswith("APNEA_c") for code in patient_codes
    ):
        raise ValueError("synthetic patients must span the a and c record groups")
    return {
        "patient_codes": sorted(patient_codes),
        "patient_count": len(patient_codes),
        "visit_count": visit_count,
        "diagnosis_count": diagnosis_count,
        "suspected_diagnosis_count": suspected_count,
    }

def load_synthetic_context(path=CONTEXT_FILE):
    records = json.loads(Path(path).read_text(encoding="utf-8"))
    validate_synthetic_context(records)
    inserted = {"visits": 0, "diagnoses": 0}
    with connect() as conn, conn.cursor() as c:
        for rec in records:
            c.execute("SELECT patient_id FROM patient WHERE patient_code=%s", (rec["patient_code"],))
            row = c.fetchone()
            if not row: raise ValueError(f"patient_code not found: {rec['patient_code']}")
            pid = row[0]
            for visit in rec.get("clinical_visits", []):
                c.execute("SELECT visit_id FROM clinical_visit WHERE patient_id=%s AND visit_start=%s AND visit_type=%s AND clinical_notes=%s", (pid, visit["visit_date"], visit.get("visit_type"), visit.get("notes")))
                vr = c.fetchone()
                if vr: vid = vr[0]
                else:
                    c.execute("INSERT INTO clinical_visit(patient_id,visit_start,visit_type,reason,clinical_notes,status) VALUES(%s,%s,%s,%s,%s,%s) RETURNING visit_id", (pid, visit["visit_date"], visit.get("visit_type"), visit.get("reason"), visit.get("notes"), visit.get("status")))
                    vid = c.fetchone()[0]; inserted["visits"] += 1
                for dx in rec.get("diagnoses", []):
                    c.execute("SELECT diagnosis_id FROM diagnosis WHERE patient_id=%s AND visit_id=%s AND diagnosis_code=%s", (pid, vid, dx.get("diagnosis_code")))
                    if not c.fetchone():
                        c.execute("INSERT INTO diagnosis(patient_id,visit_id,diagnosis_code,diagnosis_name,diagnosis_status,diagnosis_time,notes) VALUES(%s,%s,%s,%s,%s,%s,%s)", (pid, vid, dx.get("diagnosis_code"), dx["diagnosis_name"], dx.get("status"), visit["visit_date"], dx.get("notes", f"Provenance: {rec.get('provenance')}")))
                        inserted["diagnoses"] += 1
        conn.commit()
    return inserted

def select_case(prediction_id=None):
    with connect() as conn, conn.cursor() as c:
        if prediction_id is None:
            c.execute("""SELECT pr.prediction_id FROM prediction pr JOIN analysis_window w USING(window_id)
                JOIN recording r USING(recording_id) JOIN annotation a ON a.window_id=w.window_id
                WHERE r.modality='apnea' AND pr.is_anomaly AND a.source='official' AND a.label='A'
                ORDER BY (SELECT count(*) FROM prediction p2 JOIN analysis_window w2 USING(window_id)
                          WHERE w2.recording_id=w.recording_id AND w2.start_time<w.start_time AND p2.is_anomaly) DESC,
                         pr.prediction_id DESC LIMIT 1""")
            prediction_id = c.fetchone()[0]
        c.execute("""SELECT pr.prediction_id,pr.window_id,p.patient_id,p.patient_code,r.recording_id,
            w.window_index,w.start_time,w.end_time,w.duration_seconds,pr.reconstruction_error,pr.anomaly_threshold,
            pr.is_anomaly,pr.predicted_class,m.model_name,m.model_version,x.explanation_id,
            a.annotation_id,a.label,a.source
            FROM prediction pr JOIN analysis_window w USING(window_id) JOIN recording r USING(recording_id)
            JOIN patient p USING(patient_id) JOIN model m USING(model_id)
            LEFT JOIN xai_explanation x ON x.prediction_id=pr.prediction_id
            LEFT JOIN annotation a ON a.window_id=w.window_id AND a.source='official'
            WHERE pr.prediction_id=%s ORDER BY a.annotation_id LIMIT 1""", (prediction_id,))
        row=c.fetchone()
        if not row: raise ValueError(f"prediction not found: {prediction_id}")
        cols=[d[0] for d in c.description]; return dict(zip(cols,row))

def build_evidence_package(prediction_id=None):
    """Return the canonical PDM-enabled package without evaluation data."""

    if prediction_id is None:
        raise ValueError("an explicit prediction_id is required until Phase 9 freezes case selection")
    _, pdm_enabled, _ = build_paired_packages(prediction_id)
    return pdm_enabled

def load_generation_contract(
    version: str = "v3",
    prompt_path: Path | None = None,
    policy_path: Path | None = None,
) -> tuple[str, dict]:
    """Load a versioned generation contract and fail closed on drift."""

    if version not in {"v2", "v3", "v5", "v6", "v7"}:
        raise ValueError(f"unsupported generation contract version: {version}")
    if prompt_path is None:
        prompt_path = {
            "v2": PROMPT_PATH,
            "v3": PROMPT_V3_PATH,
            "v5": PROMPT_V5_PATH,
            "v6": PROMPT_V6_PATH,
            "v7": PROMPT_V7_PATH,
        }[version]
    if policy_path is None:
        policy_path = {
            "v2": POLICY_PATH,
            "v3": POLICY_V3_PATH,
            "v5": POLICY_V5_PATH,
            "v6": POLICY_V6_PATH,
            "v7": POLICY_V7_PATH,
        }[version]

    prompt = Path(prompt_path).read_text(encoding="utf-8")
    policy = json.loads(Path(policy_path).read_text(encoding="utf-8"))
    prompt_hash = hashlib.sha256(prompt.encode("utf-8")).hexdigest()
    if policy.get("prompt_sha256") != prompt_hash:
        raise ValueError("frozen prompt hash does not match generation policy")
    expected_prompt = {
        "v2": "fusion_prompt_v1", "v3": "fusion_prompt_v3", "v5": "fusion_prompt_v5",
        "v6": "fusion_prompt_v6",
        "v7": "fusion_prompt_v7",
    }[version]
    expected_policy = {
        "v2": "generation_policy_v2", "v3": "generation_policy_v3", "v5": "generation_policy_v5",
        "v6": "generation_policy_v6",
        "v7": "generation_policy_v7",
    }[version]
    expected_schema = {
        "v2": "fusion_narrative_v1", "v3": "fusion_claims_v3", "v5": "fusion_claims_v5",
        "v6": "fusion_claims_v6",
        "v7": "fusion_claims_v7",
    }[version]
    if policy.get("prompt_version") != expected_prompt:
        raise ValueError("unsupported frozen prompt version")
    if policy.get("policy_version") != expected_policy:
        raise ValueError("unsupported frozen generation policy version")
    if policy.get("response_schema_version") != expected_schema:
        raise ValueError("unsupported frozen response schema version")
    if policy.get("provider") != "Featherless":
        raise ValueError("frozen provider must be Featherless")
    if policy.get("model") != MODEL_ID or policy.get("base_url") != BASE_URL:
        raise ValueError("frozen model endpoint differs from the approved contract")
    if version in {"v5", "v6", "v7"}:
        expected_settings = {
            "temperature": 0.2,
            "top_p": 0.9,
            "max_tokens": 1400 if version == "v7" else 1200,
            "chat_template_kwargs": {"thinking": False},
        }
    else:
        expected_settings = {
            "temperature": 0.1,
            "top_p": 0.9,
            "max_tokens": 16000,
            "chat_template_kwargs": {"thinking": False},
        }
    if policy.get("generation_settings") != expected_settings:
        raise ValueError("frozen generation settings differ from the approved contract")
    return prompt, policy


def load_frozen_generation_contract(
    prompt_path: Path = PROMPT_PATH,
    policy_path: Path = POLICY_PATH,
) -> tuple[str, dict]:
    """Backward-compatible loader for the immutable v2 experiment."""

    return load_generation_contract("v2", prompt_path, policy_path)


def _has_private_evaluation_key(value) -> bool:
    prohibited = {
        "official_label", "official_labels", "official_annotation",
        "case_category", "ground_truth", "dataset_annotation",
    }
    if isinstance(value, dict):
        return any(
            str(key).lower() in prohibited or _has_private_evaluation_key(child)
            for key, child in value.items()
        )
    if isinstance(value, list):
        return any(_has_private_evaluation_key(child) for child in value)
    return False


def build_generation_request(
    evidence: dict,
    version: str | None = None,
    correction_issues: list[str] | None = None,
) -> dict:
    """Build the byte-stable request shared by both experiment conditions."""

    if _has_private_evaluation_key(evidence):
        raise ValueError("private evaluation data cannot enter an LLM request")
    if version is None:
        version = (
            "v3"
            if evidence.get("package_metadata", {}).get("evidence_package_version") == 3
            else "v2"
        )
    prompt, policy = load_generation_contract(version)
    settings = policy["generation_settings"]
    if version in {"v3", "v5", "v6", "v7"}:
        user_payload = {
            "condition": (
                "PDM-Enabled"
                if evidence.get("pdm_context", {}).get("included")
                else "Current-Evidence"
            ),
            "evidence_package_version": 3,
            "package_hash": evidence.get("package_hash"),
            "citation_evidence": evidence.get("citation_evidence", []),
        }
    else:
        user_payload = evidence
    user_content = json.dumps(
        user_payload,
        sort_keys=True,
        separators=(",", ":"),
        ensure_ascii=False,
        allow_nan=False,
        default=_jsonable,
    )
    request = {
        "model": policy["model"],
        "messages": [
            {"role": "system", "content": prompt},
            {"role": "user", "content": user_content},
        ],
        "temperature": settings["temperature"],
        "top_p": settings["top_p"],
        "max_tokens": settings["max_tokens"],
    }
    if correction_issues:
        request["messages"].append({
            "role": "user",
            "content": json.dumps({
                "instruction": "Replace the whole response. Correct every listed validation issue without changing or inventing evidence.",
                "validation_issue_codes": sorted(set(correction_issues)),
            }, sort_keys=True, separators=(",", ":")),
        })
    request["extra_body"] = {
        "chat_template_kwargs": settings["chat_template_kwargs"]
    }
    return request


NARRATIVE_SECTIONS = (
    "current_event",
    "model_and_xai_evidence",
    "previous_monitoring_history",
    "recorded_clinical_context",
    "evidence_grounded_summary",
)


def render_narrative(claims: list[dict]) -> dict[str, str]:
    """Render reader-visible sections exclusively from structured claims."""

    grouped = {section: [] for section in NARRATIVE_SECTIONS}
    for claim in claims:
        section = claim["section"]
        citation = "; ".join(claim["evidence_ids"])
        grouped[section].append(f"{claim['text'].strip()} [evidence: {citation}]")
    return {section: " ".join(grouped[section]) for section in NARRATIVE_SECTIONS}


def parse_structured_response(raw_text: str) -> dict:
    """Parse and validate the frozen narrative response schema."""

    clean = re.sub(
        r"^```(?:json)?\s*|\s*```$", "", str(raw_text).strip(), flags=re.I
    ).strip()
    try:
        result = json.loads(clean)
    except json.JSONDecodeError as exc:
        raise ValueError("provider response is not valid JSON") from exc
    if not isinstance(result, dict):
        raise ValueError("provider response has the wrong narrative schema version")
    if result.get("schema_version") in {
        "fusion_claims_v3", "fusion_claims_v5", "fusion_claims_v6",
        "fusion_claims_v7",
    }:
        source_schema = result["schema_version"]
        if set(result) != {"schema_version", "claims"}:
            raise ValueError(f"{source_schema} is claims-only; narrative sections are rendered locally")
        claims = result.get("claims")
        if not isinstance(claims, list) or not claims:
            raise ValueError("provider response claims must be a non-empty list")
        claim_ids = []
        for claim in claims:
            if not isinstance(claim, dict):
                raise ValueError("each claim must be an object")
            if set(claim) != {"claim_id", "section", "text", "evidence_ids"}:
                raise ValueError("claim fields do not match the frozen schema")
            if claim.get("section") in {"aggregate_trend", "recurrence_summary"}:
                claim["section"] = "previous_monitoring_history"
            if (
                not isinstance(claim["claim_id"], str)
                or not claim["claim_id"].strip()
                or claim["section"] not in NARRATIVE_SECTIONS
                or not isinstance(claim["text"], str)
                or not claim["text"].strip()
                or not isinstance(claim["evidence_ids"], list)
                or not claim["evidence_ids"]
                or not all(isinstance(item, str) and item for item in claim["evidence_ids"])
            ):
                raise ValueError("claim content does not match the frozen schema")
            claim_ids.append(claim["claim_id"])
        if len(claim_ids) != len(set(claim_ids)):
            raise ValueError("claim IDs must be unique")
        copied_claims = json.loads(json.dumps(claims, ensure_ascii=False))
        return {
            "schema_version": (
                "fusion_narrative_v7" if source_schema == "fusion_claims_v7"
                else "fusion_narrative_v6" if source_schema == "fusion_claims_v6"
                else "fusion_narrative_v5" if source_schema == "fusion_claims_v5"
                else "fusion_narrative_v3"
            ),
            "source_schema_version": source_schema,
            "claims": copied_claims,
            "narrative_sections": render_narrative(copied_claims),
        }
    if (
        set(result) != {"schema_version", "narrative_sections", "claims"}
        or result.get("schema_version") != "fusion_narrative_v1"
    ):
        raise ValueError("provider response has the wrong narrative schema version")
    sections = result.get("narrative_sections")
    if not isinstance(sections, dict) or set(sections) != set(NARRATIVE_SECTIONS):
        raise ValueError("provider response has invalid narrative sections")
    if not all(isinstance(sections[name], str) for name in NARRATIVE_SECTIONS):
        raise ValueError("narrative sections must be strings")
    claims = result.get("claims")
    if not isinstance(claims, list):
        raise ValueError("provider response claims must be a list")
    for claim in claims:
        if not isinstance(claim, dict):
            raise ValueError("each claim must be an object")
        if set(claim) != {"claim_id", "section", "text", "evidence_ids"}:
            raise ValueError("claim fields do not match the frozen schema")
        if (
            not isinstance(claim["claim_id"], str)
            or claim["section"] not in NARRATIVE_SECTIONS
            or not isinstance(claim["text"], str)
            or not isinstance(claim["evidence_ids"], list)
            or not claim["evidence_ids"]
            or not all(isinstance(item, str) for item in claim["evidence_ids"])
        ):
            raise ValueError("claim content does not match the frozen schema")
    return result


def _retryable_transport_error(exc: Exception, retryable_statuses: set[int]) -> bool:
    status = getattr(exc, "status_code", None)
    if status in retryable_statuses:
        return True
    name = type(exc).__name__.lower()
    return "timeout" in name or "connection" in name or "ratelimit" in name


def execute_frozen_request(
    client,
    evidence: dict,
    sleep_fn=time.sleep,
    version: str | None = None,
    validation_fn=None,
) -> dict:
    """Execute the frozen retry/parsing policy while retaining all attempts."""

    if version is None:
        version = (
            "v3"
            if evidence.get("package_metadata", {}).get("evidence_package_version") == 3
            else "v2"
        )
    request = build_generation_request(evidence, version=version)
    _, policy = load_generation_contract(version)
    transport = policy["transport_retry"]
    retryable_statuses = set(transport["retryable_http_statuses"])
    attempts = []
    parse_failures = 0
    grounding_failures = 0
    malformed_budget = int(policy["malformed_json_retry"]["maximum_additional_requests"])
    grounding_budget = int(policy.get("grounding_retry", {}).get("maximum_additional_requests", 0))
    response_number = 0
    correction_issues = None
    while True:
        response_number += 1
        active_request = (
            build_generation_request(
                evidence, version=version, correction_issues=correction_issues
            )
            if correction_issues
            else request
        )
        response = None
        for transport_attempt in range(1, transport["maximum_attempts"] + 1):
            try:
                response = client.chat.completions.create(**active_request)
                break
            except Exception as exc:
                attempts.append(
                    {
                        "response_number": response_number,
                        "transport_attempt": transport_attempt,
                        "transport_error": f"{type(exc).__name__}: {exc}",
                        "raw_response": None,
                    }
                )
                if (
                    transport_attempt >= transport["maximum_attempts"]
                    or not _retryable_transport_error(exc, retryable_statuses)
                ):
                    raise
                sleep_fn(transport["backoff_seconds"][transport_attempt - 1])
        raw = response.choices[0].message.content
        response_metadata = {
            "id": getattr(response, "id", None),
            "model": getattr(response, "model", None),
            "usage": (
                response.usage.model_dump()
                if getattr(response, "usage", None) is not None
                and hasattr(response.usage, "model_dump")
                else None
            ),
        }
        attempt_record = {
            "response_number": response_number,
            "transport_attempt": transport_attempt,
            "transport_error": None,
            "raw_response": raw,
            "response_metadata": response_metadata,
        }
        attempts.append(attempt_record)
        try:
            parsed = parse_structured_response(raw)
            grounding = validation_fn(parsed, evidence) if validation_fn else {"valid": True, "issue_codes": []}
            attempt_record["grounding_validation"] = grounding
            if not grounding.get("valid", False):
                grounding_failures += 1
                if grounding_failures > grounding_budget:
                    return {
                        "request": request,
                        "attempts": attempts,
                        "parsed": parsed,
                        "parse_failure": False,
                        "grounding_failure": True,
                        "manual_parse_required": False,
                        "validation_issue_codes": grounding.get("issue_codes", []),
                    }
                correction_issues = list(grounding.get("issue_codes", []))
                continue
            return {
                "request": request,
                "attempts": attempts,
                "parsed": parsed,
                "parse_failure": False,
                "grounding_failure": False,
                "manual_parse_required": False,
            }
        except ValueError as exc:
            parse_failures += 1
            attempt_record["parse_error"] = str(exc)
            correction_issues = None
            if parse_failures > malformed_budget:
                return {
                    "request": request,
                    "attempts": attempts,
                    "parsed": None,
                    "parse_failure": True,
                    "grounding_failure": False,
                    "manual_parse_required": True,
                    "parse_failure_count": parse_failures,
                }

def verify_model(api_key):
    r=requests.get(BASE_URL+"/models",params={"q":MODEL_ID,"available_on_current_plan":"true"},headers={"Authorization":"Bearer "+api_key},timeout=30); r.raise_for_status()
    data=r.json().get("data",[]); match=next((m for m in data if m.get("id")==MODEL_ID),None)
    if not match: raise RuntimeError(f"{MODEL_ID} is not available on the current Featherless plan")
    return match

def generate_narrative(evidence):
    key=os.getenv("FEATHERLESS_API_KEY");
    if not key: raise RuntimeError("FEATHERLESS_API_KEY is not set")
    verify_model(key)
    from openai import OpenAI
    client=OpenAI(base_url=BASE_URL,api_key=key,default_headers={"X-Title":"JAISE2026 PDM evidence prototype"})
    return execute_frozen_request(client, evidence)

def persist(evidence, result, settings):
    cse=evidence["case"]
    with connect() as conn, conn.cursor() as c:
        c.execute("INSERT INTO narrative(patient_id,window_id,prediction_id,narrative_text,llm_model_version) VALUES(%s,%s,%s,%s,%s) RETURNING narrative_id",(cse["patient_id"],cse["window_id"],cse["prediction_id"],result["narrative"],MODEL_ID)); nid=c.fetchone()[0]
        refs={"prediction":cse["prediction_id"],"annotation":evidence.get("dataset_annotation",{}).get("annotation_id"),"xai":evidence.get("xai",{}).get("explanation_id")}
        for role,val in refs.items():
            if val: c.execute("INSERT INTO narrative_evidence(narrative_id,prediction_id,annotation_id,explanation_id,evidence_role) VALUES(%s,%s,%s,%s,%s)",(nid,val if role=="prediction" else None,val if role=="annotation" else None,val if role=="xai" else None,role))
        conn.commit()
    return nid

def main(argv=None):
    """Inspect the frozen Phase 10 contract without generating narratives."""

    load_dotenv(dotenv_path=ROOT / ".env")
    parser = argparse.ArgumentParser(description=__doc__)
    mode = parser.add_mutually_exclusive_group(required=True)
    mode.add_argument("--verify-model", action="store_true")
    mode.add_argument("--show-policy", action="store_true")
    args = parser.parse_args(argv)
    prompt, policy = load_frozen_generation_contract()
    if args.verify_model:
        key = os.getenv("FEATHERLESS_API_KEY")
        if not key:
            raise RuntimeError("FEATHERLESS_API_KEY is not set")
        model = verify_model(key)
        print(json.dumps({
            "id": model.get("id"),
            "available_on_current_plan": model.get("available_on_current_plan"),
            "context_length": model.get("context_length"),
            "prompt_sha256": policy["prompt_sha256"],
        }, indent=2, sort_keys=True))
        return
    print(json.dumps({
        "policy": policy,
        "policy_sha256": hashlib.sha256(POLICY_PATH.read_bytes()).hexdigest(),
        "prompt_character_count": len(prompt),
    }, indent=2, sort_keys=True))

if __name__ == "__main__": main()
