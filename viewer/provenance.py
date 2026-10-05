"""Resolve only saved generation artifacts matched to the stored narrative."""
import hashlib
import json
from pathlib import Path
import re

ROOT = Path(__file__).resolve().parents[1]


def generation_provenance(record, prediction_id, root=ROOT):
    unavailable = {"package_hash": None, "package_hash_note":
        "No verified generation package is available for this stored narrative.", "generation_artifact": None}
    if not record:
        return unavailable
    marker = record.get("llm_model_version", "") or ""
    parts = marker.split("|")
    if len(parts) != 3:
        return unavailable
    model, run, condition = parts
    if run == "sleep_case_v1" and condition == "pdm_enabled":
        directory = root / "outputs/narrative_experiments/sleep_edf_case"
        reference_name = "manifest.json"
    elif re.fullmatch(r"[A-Za-z0-9_-]+", run) and condition in {"pdm_enabled", "current_evidence"}:
        directory = root / "outputs/narrative_experiments/apnea_ecg/runs" / run / "cases" / f"prediction_{int(prediction_id)}" / condition
        reference_name = "persistence.json"
    else:
        return unavailable
    try:
        directory.resolve().relative_to(root.resolve())
        reference = json.loads((directory / reference_name).read_text(encoding="utf-8"))
        parsed = json.loads((directory / "parsed_response.json").read_text(encoding="utf-8"))
        package = json.loads((directory / "evidence_package.json").read_text(encoding="utf-8"))
        if (reference.get("narrative_id") != record["narrative_id"]
                or reference.get("prediction_id") != prediction_id
                or parsed != json.loads(record["narrative_text"])):
            return unavailable
        expected_marker = reference.get("llm_model_version") or f"{reference.get('model')}|sleep_case_v1|pdm_enabled"
        if expected_marker != marker:
            return unavailable
        identity = package.get("current_event") or package.get("current_evidence", {}).get("identity", {})
        if identity.get("prediction_id") != prediction_id:
            return unavailable
        stored_hash = package.pop("package_hash", None)
        encoded = json.dumps(package, sort_keys=True, separators=(",", ":"), ensure_ascii=False, allow_nan=False).encode("utf-8")
        digest = hashlib.sha256(encoded).hexdigest()
        if stored_hash != digest or (reference.get("package_hash") and reference["package_hash"] != digest):
            return unavailable
        return {"package_hash": digest,
            "package_hash_note": "Verified saved generation package matched to this narrative; displayed signals are read from the current database.",
            "generation_artifact": directory.relative_to(root).as_posix()}
    except (OSError, ValueError, TypeError, KeyError, AttributeError):
        return unavailable
