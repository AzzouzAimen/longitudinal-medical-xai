"""Run the frozen paired Apnea narrative experiment exactly once per pair."""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import re
import shutil
import subprocess
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from pdm.evaluation import _concept_matches, collect_evidence_blocks

from dotenv import load_dotenv

from pdm.db import connect
from pdm.evidence import canonical_json_hash
from pdm.hybrid import (
    POLICY_V4_PATH,
    build_hybrid_narrative,
    build_synthesis_request,
    execute_optional_synthesis,
    load_generation_contract as load_generation_contract_v4,
    validate_optional_synthesis,
)
from pdm.llm import (
    BASE_URL,
    MODEL_ID,
    POLICY_PATH,
    POLICY_V3_PATH,
    POLICY_V5_PATH,
    POLICY_V6_PATH,
    POLICY_V7_PATH,
    build_generation_request,
    execute_frozen_request,
    load_generation_contract,
    load_frozen_generation_contract,
    verify_model,
)


ROOT = Path(__file__).resolve().parents[1]
EXPERIMENT_ROOT = ROOT / "outputs" / "narrative_experiments" / "apnea_ecg"
SELECTION_PATH = EXPERIMENT_ROOT / "v7" / "frozen" / "case_selection.json"
RUNS_ROOT = EXPERIMENT_ROOT / "runs"
CONDITIONS = (
    ("Current-Evidence", "current_package_path", "current_evidence"),
    ("PDM-Enabled", "pdm_package_path", "pdm_enabled"),
)


def _write_json(path: Path, payload: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(
        json.dumps(payload, indent=2, sort_keys=True, ensure_ascii=False) + "\n",
        encoding="utf-8",
    )
    temporary.replace(path)


def _sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _git_state() -> dict:
    def run(*args: str) -> str:
        result = subprocess.run(
            ["git", *args], cwd=ROOT, capture_output=True, text=True, check=True
        )
        return result.stdout.strip()

    diff = subprocess.run(
        ["git", "diff", "--binary", "HEAD"], cwd=ROOT,
        capture_output=True, check=True,
    ).stdout
    dirty_entries = run("status", "--short").splitlines()
    return {
        "revision": run("rev-parse", "HEAD"),
        "branch": run("branch", "--show-current"),
        "dirty": bool(dirty_entries),
        "dirty_entries": dirty_entries,
        "tracked_diff_sha256": hashlib.sha256(diff).hexdigest(),
    }


def _load_selection(selection_path: Path) -> dict:
    manifest = json.loads(Path(selection_path).read_text(encoding="utf-8"))
    core = manifest.get("selection_core")
    if manifest.get("selection_hash") != canonical_json_hash(core):
        raise ValueError("selection hash mismatch")
    without_hash = {k: v for k, v in manifest.items() if k != "manifest_hash"}
    if manifest.get("manifest_hash") != canonical_json_hash(without_hash):
        raise ValueError("case manifest hash mismatch")
    if len(core.get("cases", [])) != 16:
        raise ValueError("frozen selection must contain 16 cases")
    return manifest


def build_run_contract(
    selection_path: Path = SELECTION_PATH,
    run_id: str = "",
    generation_version: str = "v4",
) -> dict:
    """Build the immutable contract that must be saved before API calls."""

    selection_path = Path(selection_path).resolve()
    manifest = _load_selection(selection_path)
    if generation_version == "v4":
        prompt, policy = load_generation_contract_v4()
        policy_path = POLICY_V4_PATH
    else:
        prompt, policy = load_generation_contract(generation_version)
        policy_path = {
            "v2": POLICY_PATH,
            "v3": POLICY_V3_PATH,
            "v5": POLICY_V5_PATH,
            "v6": POLICY_V6_PATH,
            "v7": POLICY_V7_PATH,
        }[generation_version]
    derived_by_id = {
        int(row["prediction_id"]): row
        for row in manifest["derived_outputs"]["cases"]
    }
    requests = []
    package_version = None
    expected_package_version = 2 if generation_version == "v2" else 3
    for case in manifest["selection_core"]["cases"]:
        prediction_id = int(case["prediction_id"])
        derived = derived_by_id[prediction_id]
        for condition, path_key, slug in CONDITIONS:
            package_path = selection_path.parent / derived[path_key]
            package = json.loads(package_path.read_text(encoding="utf-8"))
            encountered_version = package["package_metadata"]["evidence_package_version"]
            if encountered_version != expected_package_version:
                raise ValueError(
                    f"package version {encountered_version} does not match generation "
                    f"version {generation_version}"
                )
            if package_version is not None and encountered_version != package_version:
                raise ValueError("mixed package versions in one run contract")
            package_version = encountered_version
            declared_hash = package.get("package_hash")
            package_without_hash = dict(package)
            package_without_hash.pop("package_hash", None)
            if declared_hash != canonical_json_hash(package_without_hash):
                raise ValueError(f"canonical package hash mismatch: {package_path}")
            selection_hash_key = (
                "current_package_hash" if slug == "current_evidence"
                else "pdm_package_hash"
            )
            if declared_hash != derived.get(selection_hash_key):
                raise ValueError(
                    f"package hash differs from frozen selection: {package_path}"
                )
            requests.append({
                "sequence": len(requests) + 1,
                "prediction_id": prediction_id,
                "condition": condition,
                "condition_slug": slug,
                "package_path": str(package_path.relative_to(ROOT).as_posix()),
                "package_hash": package["package_hash"],
                "package_file_sha256": _sha256(package_path),
                "current_evidence_hash": package["current_evidence_hash"],
                "private_evaluation_path": derived["private_evaluation_path"],
                "private_evaluation_sha256": _sha256(
                    selection_path.parent / derived["private_evaluation_path"]
                ),
            })
    contract = {
        "schema_version": (
            "fusion-paired-run-v1" if generation_version == "v2"
            else "fusion-paired-run-v4" if generation_version == "v4"
            else "fusion-paired-run-v5" if generation_version == "v5"
            else "fusion-paired-run-v6" if generation_version == "v6"
            else "fusion-paired-run-v7" if generation_version == "v7"
            else "fusion-paired-run-v3"
        ),
        "run_id": run_id,
        "created_at_utc": datetime.now(timezone.utc).isoformat(),
        "selection_path": str(selection_path.relative_to(ROOT).as_posix()),
        "selection_hash": manifest["selection_hash"],
        "case_manifest_hash": manifest["manifest_hash"],
        "evidence_package_version": package_version,
        "prompt_version": policy["prompt_version"],
        "prompt_sha256": policy["prompt_sha256"],
        "generation_policy_version": policy["policy_version"],
        "generation_policy_sha256": _sha256(policy_path),
        "provider": policy["provider"],
        "model": policy["model"],
        "generation_settings": policy["generation_settings"],
        "expected_case_count": 16,
        "expected_request_count": 32,
        "requests": requests,
        "code_state": _git_state(),
        "prompt_character_count": len(prompt),
    }
    contract["run_contract_hash"] = canonical_json_hash(contract)
    return contract


def _binding(contract: dict) -> dict:
    keys = (
        "selection_hash", "case_manifest_hash", "prompt_sha256",
        "generation_policy_sha256", "provider", "model",
        "generation_settings", "expected_case_count", "expected_request_count",
        "requests",
    )
    return {key: contract[key] for key in keys}


def initialize_run(run_dir: Path, contract: dict) -> tuple[dict, dict]:
    run_dir.mkdir(parents=True, exist_ok=True)
    contract_path = run_dir / "run_manifest.json"
    progress_path = run_dir / "progress.json"
    if contract_path.exists():
        existing = json.loads(contract_path.read_text(encoding="utf-8"))
        if _binding(existing) != _binding(contract):
            raise ValueError("existing run contract differs from frozen inputs")
        contract = existing
    else:
        _write_json(contract_path, contract)
    if progress_path.exists():
        progress = json.loads(progress_path.read_text(encoding="utf-8"))
    else:
        progress = {
            "run_id": contract["run_id"],
            "status": "initialized",
            "completed_request_keys": [],
            "narrative_ids": {},
            "updated_at_utc": datetime.now(timezone.utc).isoformat(),
        }
        _write_json(progress_path, progress)
    if progress.get("status") == "superseded":
        raise RuntimeError("run was superseded and cannot be resumed")
    return contract, progress


def _condition_key(prediction_id: int, slug: str) -> str:
    return f"{prediction_id}:{slug}"


def _narrative_text(parsed: dict) -> str:
    return json.dumps(parsed, sort_keys=True, separators=(",", ":"), ensure_ascii=False)


def validate_response_evidence(parsed: dict, package: dict) -> dict:
    """Report citation/identifier defects for scoring without regenerating output."""

    valid_ids = set(collect_evidence_blocks(package))
    claim_ids = [claim["claim_id"] for claim in parsed["claims"]]
    duplicate_claim_ids = sorted({item for item in claim_ids if claim_ids.count(item) > 1})
    unknown = sorted({
        evidence_id
        for claim in parsed["claims"]
        for evidence_id in claim["evidence_ids"]
        if evidence_id not in valid_ids
    })
    return {
        "valid_evidence_ids": not unknown,
        "unknown_evidence_ids": unknown,
        "unique_claim_ids": not duplicate_claim_ids,
        "duplicate_claim_ids": duplicate_claim_ids,
    }


def _claim_has_catalog_anchor(text: str, block: dict) -> bool:
    """Conservatively check that a claim refers to the kind of block it cites."""

    lowered = text.casefold()
    facts = block.get("facts", {})
    predicted_class = facts.get("predicted_class")
    claims_anomaly_class = bool(re.search(
        r"\b(?:flagged|classified|detected)(?:\s+the\s+window)?\s+(?:as\s+)?an\s+anomaly\b",
        lowered,
    ))
    claims_normal_class = bool(re.search(
        r"\b(?:flagged|classified|detected)(?:\s+the\s+window)?\s+(?:as\s+)?normal\b",
        lowered,
    ))
    if predicted_class == "normal" and claims_anomaly_class:
        return False
    if predicted_class == "anomaly" and claims_normal_class:
        return False
    claims_positive_history_anomalies = bool(re.search(
        r"\b(?:history|recurrence|summary)\b[^.;]*\b"
        r"(?:shows?|includes?|contains?|indicates?)\b[^.;]*\banomal(?:y|ies)\b",
        lowered,
    ))
    if facts.get("anomaly_count") == 0 and claims_positive_history_anomalies:
        if not re.search(r"\b(?:no|zero)\s+anomal(?:y|ies)\b", lowered):
            return False
    if facts.get("available") is False:
        positive_available = bool(re.search(
            r"\b(?:is|are|was|were)\s+(?:available|present|provided|supplied)\b",
            lowered,
        ))
        negative_available = bool(re.search(
            r"\b(?:not\s+(?:available|present|provided|supplied)|unavailable|absent|"
            r"no\s+[^.;]*(?:available|present|provided|supplied))\b",
            lowered,
        ))
        if positive_available and not negative_available:
            return False

    def scalar_values(value):
        if isinstance(value, dict):
            for child in value.values():
                yield from scalar_values(child)
        elif isinstance(value, list):
            for child in value:
                yield from scalar_values(child)
        elif value is not None:
            yield value

    for value in scalar_values(facts):
        if isinstance(value, bool):
            continue
        candidate = str(value).casefold()
        if candidate and candidate in lowered:
            return True

    keywords = {
        "current_event_summary": ("event", "window", "recording", "second"),
        "detector_summary": ("detector", "reconstruction", "threshold", "margin", "anomaly", "normal"),
        "model_provenance": ("model", "bilstm", "autoencoder", "fold"),
        "signal_summary": ("signal", "ecg", "sample", "channel"),
        "attention_summary": ("attention", "temporal relevance", "channel owner"),
        "pdm_context_status": ("pdm", "history", "context"),
        "recent_history_summary": ("history", "prior", "previous", "preceding"),
        "aggregate_trend_summary": ("aggregate", "trend", "bucket"),
        "recurrence_summary": (
            "recurrence", "analyzed window", "prior window", "sustained pattern",
        ),
        "clinical_context_summary": ("clinical", "diagnos", "visit", "synthetic"),
    }
    return any(token in lowered for token in keywords.get(block.get("evidence_type"), ()))


def validate_grounded_response(
    parsed: dict, package: dict, *, enforce_concepts: bool = True
) -> dict:
    """Validate the v3 claim/citation boundary before persistence."""

    catalog = package.get("citation_evidence", [])
    by_id = {
        row["evidence_id"]: row
        for row in catalog
        if isinstance(row, dict) and isinstance(row.get("evidence_id"), str)
    }
    issues = []
    claims = parsed.get("claims", [])
    claim_ids = [str(claim.get("claim_id", "")) for claim in claims]
    for duplicate in sorted({item for item in claim_ids if claim_ids.count(item) > 1}):
        issues.append(f"{duplicate}:duplicate_claim_id")
    included = bool(package.get("pdm_context", {}).get("included"))
    for claim in claims:
        claim_id = str(claim.get("claim_id", ""))
        text = str(claim.get("text", ""))
        lowered = text.casefold()
        refs = claim.get("evidence_ids", [])
        for ref in refs:
            block = by_id.get(str(ref))
            if block is None:
                issues.append(f"{claim_id}:unknown_evidence_id:{ref}")
            elif not _claim_has_catalog_anchor(text, block):
                issues.append(f"{claim_id}:citation_owner_mismatch:{ref}")
        denies_history = any(phrase in lowered for phrase in (
            "no pdm history", "no previous monitoring history", "no prior monitoring history",
            "no history was supplied", "no history or clinical context",
        ))
        if included and denies_history:
            issues.append(f"{claim_id}:pdm_history_denied_when_included")
        if "attention" in lowered:
            attention_as_support = bool(
                re.search(
                    r"\b(?:anomal(?:y|ous)|normal|detector|decision|classified|flagged)\b"
                    r".*\bsupport(?:ed|s|ing)?\s+by\b.*\battention\b",
                    lowered,
                )
                or re.search(
                    r"\battention\b.*\bsupport(?:s|ed|ing)?\b.*"
                    r"\b(?:anomal(?:y|ous)|normal|detector|decision|classification)\b",
                    lowered,
                )
            )
            if attention_as_support:
                issues.append(f"{claim_id}:attention_described_as_detector_support")
            negated_probability = bool(
                re.search(r"\bnot\s+(?:(?:a|anomaly)\s+)?probability\b", lowered)
            )
            negated_confidence = bool(
                re.search(r"\bnot\s+(?:a\s+)?(?:calibrated\s+)?confidence\b", lowered)
                or re.search(
                    r"\bnot\s+(?:(?:a|anomaly)\s+)?probability\s*(?:,|or)\s*"
                    r"(?:calibrated\s+)?confidence\b",
                    lowered,
                )
            )
            if "probability" in lowered and not negated_probability:
                issues.append(f"{claim_id}:attention_described_as_probability")
            if "confidence" in lowered and not negated_confidence:
                issues.append(f"{claim_id}:attention_described_as_confidence")
    concepts = (
        package.get("evaluation_concepts", {}).get("current", [])
        if enforce_concepts else []
    )
    for concept in concepts:
        evidence_id = str(concept.get("evidence_id", ""))
        related_text = " ".join(
            str(claim.get("text", ""))
            for claim in claims
            if evidence_id in claim.get("evidence_ids", [])
        )
        represented = _concept_matches(related_text, concept)
        if concept.get("concept_id") == "xai_interpretation":
            lowered_related = related_text.casefold()
            represented = represented or (
                "shared window-level temporal relevance" in lowered_related
                and (
                    bool(re.search(
                        r"not\s+(?:a\s+)?channel-specific importance",
                        lowered_related,
                    ))
                    or "no channel owner" in lowered_related
                    or "no channel-level evidence" in lowered_related
                    or "does not indicate channel-specific importance" in lowered_related
                )
            )
        if not represented:
            issues.append(f"missing_current_concept:{concept.get('concept_id')}")
    return {
        "schema_version": "fusion-response-validation-v3",
        "valid": not issues,
        "issue_codes": sorted(set(issues)),
        "claim_count": len(claims),
        "catalog_evidence_count": len(by_id),
    }


def require_valid_response(validation: dict, request_key: str) -> None:
    """Fail closed before any v3 narrative can cross the persistence boundary."""

    if not validation.get("valid", False):
        issues = validation.get("issue_codes", [])
        raise RuntimeError(
            f"grounding validation failed for {request_key}: {','.join(issues)}"
        )


def _verify_narrative_links(cur, narrative_id: int, prediction_id: int,
                            explanation_ids: list[int]) -> None:
    cur.execute(
        """
        SELECT prediction_id,annotation_id,explanation_id,evidence_role
        FROM narrative_evidence WHERE narrative_id=%s
        ORDER BY evidence_role,prediction_id,explanation_id
        """,
        (narrative_id,),
    )
    actual = {
        (row[0], row[1], row[2], row[3])
        for row in cur.fetchall()
    }
    expected = {(prediction_id, None, None, "current_prediction")}
    expected.update((None, None, item, "current_xai") for item in explanation_ids)
    if actual != expected:
        raise ValueError(f"persisted evidence links differ for narrative {narrative_id}")


def persist_narrative(package: dict, parsed: dict, run_id: str, condition_slug: str) -> int:
    """Persist once and link only current prediction/XAI evidence."""

    identity = package["current_evidence"]["identity"]
    marker = f"{MODEL_ID}|{run_id}|{condition_slug}"
    if len(marker) > 128:
        raise ValueError("run marker exceeds narrative.llm_model_version capacity")
    narrative_text = _narrative_text(parsed)
    explanation_ids = sorted({
        int(row["explanation_id"])
        for row in package["current_evidence"]["xai_evidence"]["explanations"]
    })
    with connect() as conn, conn.cursor() as cur:
        cur.execute(
            "SELECT narrative_id,narrative_text FROM narrative WHERE prediction_id=%s AND llm_model_version=%s ORDER BY narrative_id",
            (identity["prediction_id"], marker),
        )
        existing = cur.fetchall()
        if len(existing) > 1:
            raise ValueError("duplicate persisted narrative for run condition")
        if existing:
            narrative_id, stored_text = int(existing[0][0]), existing[0][1]
            if stored_text != narrative_text:
                raise ValueError("persisted narrative content differs for run condition")
            _verify_narrative_links(cur, narrative_id, identity["prediction_id"], explanation_ids)
            return narrative_id
        cur.execute(
            """
            INSERT INTO narrative(patient_id,window_id,prediction_id,narrative_text,llm_model_version)
            VALUES(%s,%s,%s,%s,%s) RETURNING narrative_id
            """,
            (
                identity["patient_id"], identity["window_id"],
                identity["prediction_id"], narrative_text, marker,
            ),
        )
        narrative_id = int(cur.fetchone()[0])
        cur.execute(
            """
            INSERT INTO narrative_evidence(narrative_id,prediction_id,evidence_role)
            VALUES(%s,%s,'current_prediction')
            """,
            (narrative_id, identity["prediction_id"]),
        )
        for explanation_id in explanation_ids:
            cur.execute(
                """
                INSERT INTO narrative_evidence(narrative_id,explanation_id,evidence_role)
                VALUES(%s,%s,'current_xai')
                """,
                (narrative_id, explanation_id),
            )
        conn.commit()
        _verify_narrative_links(cur, narrative_id, identity["prediction_id"], explanation_ids)
    return narrative_id


def _save_response_artifacts(condition_dir: Path, audit: dict) -> None:
    _write_json(condition_dir / "response_audit.json", audit)
    for index, attempt in enumerate(audit["attempts"], start=1):
        raw = attempt.get("raw_response")
        if raw is not None:
            (condition_dir / f"raw_response_{index}.txt").write_text(raw, encoding="utf-8")
    if audit.get("parsed") is not None:
        _write_json(condition_dir / "parsed_response.json", audit["parsed"])


def _database_snapshot() -> dict:
    counts = {}
    with connect() as conn, conn.cursor() as cur:
        for table in ("narrative", "narrative_evidence", "prediction", "xai_explanation"):
            cur.execute(f"SELECT count(*) FROM {table}")
            counts[table] = int(cur.fetchone()[0])
        cur.execute(
            "SELECT count(*) FROM narrative WHERE llm_model_version LIKE %s",
            (f"{MODEL_ID}|%",),
        )
        counts["experiment_narratives"] = int(cur.fetchone()[0])
    return {"captured_at_utc": datetime.now(timezone.utc).isoformat(), "counts": counts}


def run_experiment(
    run_id: str,
    selection_path: Path = SELECTION_PATH,
    generation_version: str = "v4",
) -> Path:
    load_dotenv(dotenv_path=ROOT / ".env")
    key = os.getenv("FEATHERLESS_API_KEY")
    if not key:
        raise RuntimeError("FEATHERLESS_API_KEY is not set")
    run_dir = RUNS_ROOT / run_id
    run_dir.mkdir(parents=True, exist_ok=True)
    lock_path = run_dir / "run.lock"
    try:
        lock_fd = os.open(lock_path, os.O_CREAT | os.O_EXCL | os.O_WRONLY)
    except FileExistsError as exc:
        raise RuntimeError(f"run is already active or has a stale lock: {lock_path}") from exc
    os.write(lock_fd, json.dumps({
        "pid": os.getpid(), "started_at_utc": datetime.now(timezone.utc).isoformat()
    }).encode("utf-8"))
    os.close(lock_fd)
    try:
        return _run_experiment_locked(
            run_id, selection_path, run_dir, key, generation_version
        )
    finally:
        lock_path.unlink(missing_ok=True)


def _run_experiment_locked(
    run_id: str,
    selection_path: Path,
    run_dir: Path,
    key: str,
    generation_version: str,
) -> Path:
    contract_candidate = build_run_contract(selection_path, run_id, generation_version)
    contract, progress = initialize_run(run_dir, contract_candidate)
    snapshot_path = run_dir / "database_before.json"
    if not snapshot_path.exists():
        _write_json(snapshot_path, _database_snapshot())
    selection = _load_selection(Path(selection_path))
    private_by_id = {
        int(row["prediction_id"]): row["private_evaluation_path"]
        for row in selection["derived_outputs"]["cases"]
    }
    verify_model(key)
    from openai import OpenAI
    client = OpenAI(
        base_url=BASE_URL,
        api_key=key,
        default_headers={"X-Title": "JAISE2026 PDM evidence fusion"},
    )
    completed = set(progress["completed_request_keys"])
    for row in contract["requests"]:
        prediction_id = int(row["prediction_id"])
        slug = row["condition_slug"]
        request_key = _condition_key(prediction_id, slug)
        condition_dir = run_dir / "cases" / f"prediction_{prediction_id}" / slug
        condition_dir.mkdir(parents=True, exist_ok=True)
        package_source = ROOT / row["package_path"]
        package_target = condition_dir / "evidence_package.json"
        if not package_target.exists():
            shutil.copy2(package_source, package_target)
        if _sha256(package_target) != row["package_file_sha256"]:
            raise ValueError(f"package file hash drift for {request_key}")
        package = json.loads(package_target.read_text(encoding="utf-8"))
        if package["package_hash"] != row["package_hash"]:
            raise ValueError(f"package hash drift for {request_key}")
        private_target = condition_dir.parent / "private_evaluation.json"
        if not private_target.exists():
            shutil.copy2(Path(selection_path).resolve().parent / private_by_id[prediction_id], private_target)
        if _sha256(private_target) != row["private_evaluation_sha256"]:
            raise ValueError(f"private evaluation hash drift for {request_key}")
        request = (
            build_synthesis_request(package)
            if generation_version == "v4"
            else build_generation_request(package, version=generation_version)
        )
        request_path = condition_dir / "request.json"
        if request_path.exists():
            if json.loads(request_path.read_text(encoding="utf-8")) != request:
                raise ValueError(f"serialized request drift for {request_key}")
        else:
            _write_json(request_path, request)
        persistence_path = condition_dir / "persistence.json"
        if request_key in completed and persistence_path.exists():
            continue
        parsed_path = condition_dir / "parsed_response.json"
        if parsed_path.exists():
            parsed = json.loads(parsed_path.read_text(encoding="utf-8"))
        else:
            request_state_path = condition_dir / "request_state.json"
            if request_state_path.exists():
                progress["status"] = "blocked_unknown_request_outcome"
                progress["blocked_request_key"] = request_key
                progress["updated_at_utc"] = datetime.now(timezone.utc).isoformat()
                _write_json(run_dir / "progress.json", progress)
                raise RuntimeError(
                    f"request outcome is uncertain for {request_key}; automatic retry is forbidden"
                )
            _write_json(request_state_path, {
                "status": "started",
                "request_key": request_key,
                "started_at_utc": datetime.now(timezone.utc).isoformat(),
                "request_sha256": _sha256(request_path),
            })
            if generation_version == "v4":
                audit = execute_optional_synthesis(
                    client,
                    package,
                    lambda parsed, evidence, enforce_concepts=False: validate_optional_synthesis(
                        parsed, evidence, base_validation_fn=validate_grounded_response
                    ),
                )
                _write_json(condition_dir / "response_audit.json", audit)
                for index, attempt in enumerate(audit["attempts"], start=1):
                    if attempt.get("raw_response") is not None:
                        (condition_dir / f"raw_response_{index}.txt").write_text(
                            attempt["raw_response"], encoding="utf-8"
                        )
                parsed = build_hybrid_narrative(package, audit.get("accepted_claim"))
                _write_json(parsed_path, parsed)
            else:
                audit = execute_frozen_request(
                    client,
                    package,
                    version=generation_version,
                    validation_fn=(
                        validate_grounded_response
                        if generation_version == "v3" else None
                    ),
                )
                _save_response_artifacts(condition_dir, audit)
                parsed = audit.get("parsed")
            if generation_version != "v4" and audit["parse_failure"]:
                progress["status"] = "blocked_parse_failure"
                progress["blocked_request_key"] = request_key
                progress["updated_at_utc"] = datetime.now(timezone.utc).isoformat()
                _write_json(run_dir / "progress.json", progress)
                raise RuntimeError(f"manual parsing required for {request_key}")
            if generation_version != "v4" and audit.get("grounding_failure"):
                progress["status"] = "blocked_grounding_failure"
                progress["blocked_request_key"] = request_key
                progress["validation_issue_codes"] = audit.get("validation_issue_codes", [])
                progress["updated_at_utc"] = datetime.now(timezone.utc).isoformat()
                _write_json(run_dir / "progress.json", progress)
                raise RuntimeError(f"grounding validation failed for {request_key}")
        validation = (
            validate_grounded_response(parsed, package)
            if generation_version in {"v3", "v4"}
            else validate_response_evidence(parsed, package)
        )
        _write_json(condition_dir / "response_validation.json", validation)
        if generation_version in {"v3", "v4"}:
            require_valid_response(validation, request_key)
        request_state_path = condition_dir / "request_state.json"
        response_audit_path = condition_dir / "response_audit.json"
        if response_audit_path.exists():
            _write_json(request_state_path, {
                "status": "response_saved",
                "request_key": request_key,
                "completed_at_utc": datetime.now(timezone.utc).isoformat(),
                "request_sha256": _sha256(request_path),
                "response_audit_sha256": _sha256(response_audit_path),
                "parsed_response_sha256": _sha256(parsed_path),
            })
        narrative_id = persist_narrative(package, parsed, run_id, slug)
        _write_json(persistence_path, {
            "narrative_id": narrative_id,
            "prediction_id": prediction_id,
            "condition": row["condition"],
            "llm_model_version": f"{MODEL_ID}|{run_id}|{slug}",
        })
        if generation_version in {"v3", "v4", "v5", "v6", "v7"}:
            _write_json(condition_dir / "citation_manifest.json", {
                "schema_version": (
                    "fusion-citation-manifest-v4"
                    if generation_version == "v4"
                    else "fusion-citation-manifest-v5"
                    if generation_version == "v5"
                    else "fusion-citation-manifest-v6"
                    if generation_version == "v6"
                    else "fusion-citation-manifest-v7"
                    if generation_version == "v7"
                    else "fusion-citation-manifest-v3"
                ),
                "run_id": run_id,
                "condition": row["condition"],
                "prediction_id": prediction_id,
                "package_hash": package["package_hash"],
                "narrative_hash": canonical_json_hash(parsed),
                "narrative_id": narrative_id,
                "validation": validation,
                "claims": [
                    {
                        "claim_id": claim["claim_id"],
                        "section": claim["section"],
                        "evidence_ids": claim["evidence_ids"],
                    }
                    for claim in parsed["claims"]
                ],
            })
        completed.add(request_key)
        progress["completed_request_keys"] = sorted(completed)
        progress["narrative_ids"][request_key] = narrative_id
        progress["status"] = "running"
        progress["updated_at_utc"] = datetime.now(timezone.utc).isoformat()
        _write_json(run_dir / "progress.json", progress)
    if len(completed) != contract["expected_request_count"]:
        raise ValueError("run ended without all expected requests")
    progress["status"] = "complete"
    progress["completed_at_utc"] = datetime.now(timezone.utc).isoformat()
    progress["updated_at_utc"] = progress["completed_at_utc"]
    _write_json(run_dir / "progress.json", progress)
    return run_dir


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run-id", required=True)
    parser.add_argument("--selection", type=Path, default=SELECTION_PATH)
    parser.add_argument(
        "--generation-version",
        choices=("v2", "v3", "v4", "v5", "v6", "v7"),
        default="v7",
    )
    return parser


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    run_dir = run_experiment(args.run_id, args.selection, args.generation_version)
    print(json.dumps({"run_dir": str(run_dir), "status": "complete"}, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
