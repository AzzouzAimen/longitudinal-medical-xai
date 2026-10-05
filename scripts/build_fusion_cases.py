"""Freeze deterministic Apnea cases and export their paired evidence packages."""

from __future__ import annotations

import argparse
import hashlib
import json
from collections import Counter, defaultdict
from datetime import date, datetime
from decimal import Decimal
from pathlib import Path
from typing import Any

from pdm.db import connect
from pdm.evidence import build_paired_packages, build_paired_packages_v3, canonical_json_hash


ROOT = Path(__file__).resolve().parents[1]
DEFAULT_OUTPUT_ROOT = ROOT / "outputs" / "narrative_experiments" / "apnea_ecg"
MINIMUM_SEPARATION_SECONDS = 300
PATIENT_ALLOCATION = {
    "TP": {"APNEA_a01": 5, "APNEA_a02": 1},
    "FP": {"APNEA_c01": 2, "APNEA_c02": 2, "APNEA_c03": 2},
    "TN": {
        "APNEA_a02": 1,
        "APNEA_a03": 1,
        "APNEA_c01": 1,
        "APNEA_c02": 1,
    },
}
HELD_OUT_PATIENT_ALLOCATION = {
    "TP": {"APNEA_a01": 6},
    "FP": {"APNEA_c01": 1, "APNEA_c02": 2, "APNEA_c03": 3},
    "TN": {
        "APNEA_a02": 1,
        "APNEA_a03": 1,
        "APNEA_c01": 1,
        "APNEA_c02": 1,
    },
}
POST_DEVELOPMENT_PATIENT_ALLOCATION = {
    "TP": {"APNEA_a01": 6},
    "FP": {"APNEA_c02": 2, "APNEA_c03": 4},
    "TN": {
        "APNEA_a02": 1,
        "APNEA_a03": 1,
        "APNEA_c01": 1,
        "APNEA_c02": 1,
    },
}


def _json_default(value: Any) -> Any:
    if isinstance(value, (datetime, date)):
        return value.isoformat()
    if isinstance(value, Decimal):
        return float(value)
    raise TypeError(f"cannot serialize {type(value)!r}")


def _write_json(path: Path, payload: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(
        json.dumps(payload, indent=2, sort_keys=True, ensure_ascii=False, default=_json_default) + "\n",
        encoding="utf-8",
    )
    temporary.replace(path)


def _category(label: str, is_anomaly: bool) -> str:
    positive = label == "A"
    if positive and is_anomaly:
        return "TP"
    if positive:
        return "FN"
    return "FP" if is_anomaly else "TN"


def _sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def select_cases(
    candidates: list[dict], excluded_prediction_ids: set[int] | None = None,
    patient_allocation: dict[str, dict[str, int]] | None = None,
) -> list[dict]:
    """Apply the frozen patient quotas and deterministic diversity ranking."""

    excluded_prediction_ids = excluded_prediction_ids or set()
    patient_allocation = patient_allocation or PATIENT_ALLOCATION
    candidates = [
        row for row in candidates
        if int(row["prediction_id"]) not in excluded_prediction_ids
    ]
    prediction_ids = [int(row["prediction_id"]) for row in candidates]
    if len(prediction_ids) != len(set(prediction_ids)):
        raise ValueError("candidate pool contains duplicate prediction IDs")

    selected: list[dict] = []
    selected_by_recording: dict[str, list[datetime]] = defaultdict(list)
    for category, patient_quotas in patient_allocation.items():
        for patient_code, quota in patient_quotas.items():
            group = [
                dict(row) for row in candidates
                if row["category"] == category and row["patient_code"] == patient_code
            ]
            if len(group) < quota:
                raise ValueError(
                    f"frozen quota unsupported for {category}/{patient_code}: "
                    f"need {quota}, found {len(group)}"
                )
            history_ready = [row for row in group if int(row["history_count"]) >= 5]
            pool = history_ready if len(history_ready) >= quota else group
            ordered = sorted(
                pool,
                key=lambda row: (float(row["normalized_margin"]), int(row["prediction_id"])),
            )
            denominator = max(1, len(ordered) - 1)
            percentile = {
                int(row["prediction_id"]): index / denominator
                for index, row in enumerate(ordered)
            }
            available = {int(row["prediction_id"]): row for row in ordered}
            for slot in range(quota):
                target = (slot + 1) / (quota + 1)
                preferred_recurrence = "recurrent" if slot % 2 else "isolated"
                eligible = []
                for row in available.values():
                    timestamp = row["timestamp"]
                    if isinstance(timestamp, str):
                        timestamp = datetime.fromisoformat(timestamp)
                        row["timestamp"] = timestamp
                    prior_times = selected_by_recording[str(row["recording_id"])]
                    if any(
                        abs((timestamp - prior).total_seconds()) < MINIMUM_SEPARATION_SECONDS
                        for prior in prior_times
                    ):
                        continue
                    recurrence = "recurrent" if int(row["recent_anomaly_count"]) > 0 else "isolated"
                    rank = (
                        0 if recurrence == preferred_recurrence else 1,
                        abs(percentile[int(row["prediction_id"])] - target),
                        0 if int(row["history_count"]) >= 5 else 1,
                        int(row["prediction_id"]),
                    )
                    eligible.append((rank, recurrence, row))
                if not eligible:
                    raise ValueError(
                        f"300-second separation prevents frozen quota for "
                        f"{category}/{patient_code} slot {slot + 1}"
                    )
                _, recurrence, chosen = min(eligible, key=lambda item: item[0])
                chosen = dict(chosen)
                chosen["recurrence_class"] = recurrence
                chosen["selection_slot"] = slot + 1
                chosen["target_margin_percentile"] = target
                chosen["inclusion_reason"] = (
                    f"frozen {category}/{patient_code} slot {slot + 1}; "
                    f"{recurrence}; margin-diversity target {target:.3f}; "
                    f"{int(chosen['history_count'])} prior events"
                )
                selected.append(chosen)
                selected_by_recording[str(chosen["recording_id"])].append(chosen["timestamp"])
                del available[int(chosen["prediction_id"])]

    counts = Counter(row["category"] for row in selected)
    if counts != Counter({"TP": 6, "FP": 6, "TN": 4}):
        raise ValueError(f"selection category counts do not match 6/6/4: {dict(counts)}")
    return selected


def audit_apnea_pool() -> tuple[list[dict], dict]:
    """Read and audit the complete private Apnea OOF candidate universe."""

    artifact_root = ROOT / "outputs" / "detector_training" / "apnea"
    root_manifest = json.loads(
        (artifact_root / "loso_manifest.json").read_text(encoding="utf-8")
    )
    if root_manifest.get("artifact_status") != "validated":
        raise ValueError("Apnea LOSO root manifest is not validated")
    folds = {}
    for fold in root_manifest.get("folds", []):
        manifest_path = artifact_root / fold["fold_manifest"]
        actual_hash = _sha256_file(manifest_path)
        if actual_hash != fold.get("fold_manifest_sha256"):
            raise ValueError(f"fold manifest hash mismatch: {fold['test_patient']}")
        folds[str(fold["test_patient"])] = fold

    query = """
        SELECT ps.prediction_id, ps.window_id, ps.patient_id, p.patient_code,
               ps.recording_id, ps.timestamp, ps.normalized_margin,
               ps.is_anomaly, a.label, m.input_configuration,
               EXISTS (
                   SELECT 1 FROM clinical_visit cv
                   WHERE cv.patient_id = ps.patient_id AND cv.visit_start < ps.timestamp
               ) OR EXISTS (
                   SELECT 1 FROM diagnosis d
                   WHERE d.patient_id = ps.patient_id AND d.diagnosis_time < ps.timestamp
               ) AS clinical_context_available
        FROM prediction_series ps
        JOIN patient p ON p.patient_id = ps.patient_id
        JOIN recording r ON r.recording_id = ps.recording_id
        JOIN model m ON m.model_id = ps.model_id
        JOIN annotation a ON a.window_id = ps.window_id
          AND a.source = 'official_evaluation_only'
          AND a.annotation_type = 'apnea_official'
        WHERE r.modality = 'apnea'
        ORDER BY ps.patient_id, ps.timestamp, ps.prediction_id
    """
    columns = (
        "prediction_id", "window_id", "patient_id", "patient_code",
        "recording_id", "timestamp", "normalized_margin", "is_anomaly",
        "official_label", "model_configuration", "clinical_context_available",
    )
    with connect() as conn, conn.cursor() as cur:
        cur.execute(query)
        rows = [dict(zip(columns, values)) for values in cur.fetchall()]
    if not rows:
        raise ValueError("complete Apnea candidate pool is empty")

    prior_by_patient: dict[int, list[dict]] = defaultdict(list)
    for row in rows:
        config_value = row.pop("model_configuration")
        config = json.loads(config_value) if isinstance(config_value, str) else config_value
        if not isinstance(config, dict):
            raise ValueError(f"invalid model provenance for prediction {row['prediction_id']}")
        checkpoint_hash = config.get("model_checkpoint_sha256")
        if not checkpoint_hash:
            raise ValueError(f"missing checkpoint hash for prediction {row['prediction_id']}")
        test_patient = str(config.get("test_patient"))
        fold_artifact = folds.get(test_patient)
        if fold_artifact is None:
            raise ValueError(f"missing validated fold manifest for patient {test_patient}")
        if fold_artifact.get("model_checkpoint_sha256") != checkpoint_hash:
            raise ValueError(f"checkpoint hash mismatch for prediction {row['prediction_id']}")
        row["fold_id"] = config.get("fold_id")
        row["fold_manifest_sha256"] = fold_artifact["fold_manifest_sha256"]
        row["model_checkpoint_sha256"] = checkpoint_hash
        row["category"] = _category(row.pop("official_label"), bool(row["is_anomaly"]))
        previous = prior_by_patient[int(row["patient_id"])]
        row["history_count"] = len(previous)
        row["recent_anomaly_count"] = sum(bool(item["is_anomaly"]) for item in previous[-5:])
        row["clinical_context_available"] = bool(row["clinical_context_available"])
        previous.append(row)

    by_recording_category: dict[tuple[str, str], list[dict]] = defaultdict(list)
    for row in rows:
        by_recording_category[(str(row["recording_id"]), row["category"])].append(row)
    for group in by_recording_category.values():
        group.sort(key=lambda row: (row["timestamp"], row["prediction_id"]))
        for index, row in enumerate(group):
            neighbours = group[max(0, index - 1):index] + group[index + 1:index + 2]
            row["adjacent_same_category"] = any(
                abs((row["timestamp"] - other["timestamp"]).total_seconds())
                < MINIMUM_SEPARATION_SECONDS
                for other in neighbours
            )

    totals = Counter(row["category"] for row in rows)
    patient_counts: dict[str, dict[str, int]] = {}
    for patient_code in sorted({row["patient_code"] for row in rows}):
        counts = Counter(row["category"] for row in rows if row["patient_code"] == patient_code)
        patient_counts[patient_code] = {
            category: counts.get(category, 0) for category in ("TP", "FP", "TN", "FN")
        }
    audit = {
        "total_candidates": len(rows),
        "category_totals": {category: totals.get(category, 0) for category in ("TP", "FP", "TN", "FN")},
        "by_patient": patient_counts,
        "history_ready_count": sum(int(row["history_count"] >= 5) for row in rows),
        "clinical_context_count": sum(bool(row["clinical_context_available"]) for row in rows),
        "recurrent_count": sum(int(row["recent_anomaly_count"] > 0) for row in rows),
        "isolated_count": sum(int(row["recent_anomaly_count"] == 0) for row in rows),
        "adjacent_same_category_count": sum(bool(row["adjacent_same_category"]) for row in rows),
    }
    return rows, audit


def _selection_case(row: dict) -> dict:
    fields = (
        "prediction_id", "window_id", "patient_id", "patient_code",
        "recording_id", "timestamp", "category", "normalized_margin",
        "history_count", "recent_anomaly_count", "recurrence_class",
        "clinical_context_available", "adjacent_same_category", "fold_id",
        "fold_manifest_sha256", "model_checkpoint_sha256", "selection_slot",
        "target_margin_percentile", "inclusion_reason",
    )
    return {field: row[field] for field in fields}


def _selection_report(manifest: dict) -> str:
    audit = manifest["pool_audit"]
    lines = [
        "# Frozen Apnea Case Selection",
        "",
        f"Selection hash: `{manifest['selection_hash']}`",
        "",
        "## Complete OOF pool",
        "",
        "| Patient | TP | FP | TN | FN |",
        "|---|---:|---:|---:|---:|",
    ]
    for patient, counts in audit["by_patient"].items():
        lines.append(f"| {patient} | {counts['TP']} | {counts['FP']} | {counts['TN']} | {counts['FN']} |")
    lines.extend([
        "",
        "## Frozen cases",
        "",
        "| Prediction | Window | Patient | Category | Recurrence | Context | Reason |",
        "|---:|---:|---|---|---|---|---|",
    ])
    for case in manifest["selection_core"]["cases"]:
        lines.append(
            f"| {case['prediction_id']} | {case['window_id']} | {case['patient_code']} | "
            f"{case['category']} | {case['recurrence_class']} | "
            f"{str(case['clinical_context_available']).lower()} | {case['inclusion_reason']} |"
        )
    lines.extend([
        "",
        "Identities were frozen before dense materialization or narrative generation.",
        "Official labels and TP/FP/TN categories are private evaluation data.",
        "",
    ])
    return "\n".join(lines)


def freeze_selection(
    output_root: Path = DEFAULT_OUTPUT_ROOT,
    excluded_prediction_ids: set[int] | None = None,
) -> dict:
    candidates, audit = audit_apnea_pool()
    excluded_prediction_ids = excluded_prediction_ids or set()
    if not excluded_prediction_ids:
        allocation = PATIENT_ALLOCATION
    else:
        available_fp_c01 = sum(
            row["category"] == "FP"
            and row["patient_code"] == "APNEA_c01"
            and int(row["prediction_id"]) not in excluded_prediction_ids
            for row in candidates
        )
        allocation = (
            HELD_OUT_PATIENT_ALLOCATION
            if available_fp_c01 >= HELD_OUT_PATIENT_ALLOCATION["FP"].get("APNEA_c01", 0)
            else POST_DEVELOPMENT_PATIENT_ALLOCATION
        )
    selected = select_cases(candidates, excluded_prediction_ids, allocation)
    selection_core = {
        "category_target": {"TP": 6, "FP": 6, "TN": 4},
        "patient_allocation": allocation,
        "minimum_separation_seconds": MINIMUM_SEPARATION_SECONDS,
        "excluded_development_prediction_ids": sorted(excluded_prediction_ids),
        "cases": [_selection_case(row) for row in selected],
    }
    manifest = {
        "schema_version": "phase9-apnea-case-selection-v1",
        "pool_audit": audit,
        "selection_core": selection_core,
        "selection_hash": canonical_json_hash(selection_core),
    }
    path = output_root / "case_selection.json"
    if path.exists():
        existing = json.loads(path.read_text(encoding="utf-8"))
        if existing.get("selection_hash") != manifest["selection_hash"]:
            raise ValueError("existing frozen selection differs; refusing to overwrite identities")
        return existing
    _write_json(path, manifest)
    (output_root / "APNEA_CASE_SELECTION.md").write_text(
        _selection_report(manifest), encoding="utf-8"
    )
    return manifest


def _validate_frozen_manifest(manifest: dict) -> list[dict]:
    core = manifest.get("selection_core")
    if not isinstance(core, dict) or not isinstance(core.get("cases"), list):
        raise ValueError("selection manifest must contain selection_core.cases")
    if manifest.get("selection_hash") != canonical_json_hash(core):
        raise ValueError("selection hash does not match the immutable selection core")
    cases = core["cases"]
    prediction_ids = [int(case["prediction_id"]) for case in cases]
    if len(prediction_ids) != len(set(prediction_ids)):
        raise ValueError("selection manifest contains duplicate prediction IDs")
    return cases


def finalize_packages(selection_path: Path) -> dict:
    """Write paired public packages and physically separate private metadata."""

    selection_path = Path(selection_path).resolve()
    manifest = json.loads(selection_path.read_text(encoding="utf-8"))
    cases = _validate_frozen_manifest(manifest)
    output_root = selection_path.parent
    derived_cases = []
    for frozen in cases:
        prediction_id = int(frozen["prediction_id"])
        current, enabled, private = build_paired_packages(prediction_id)
        identity = current["current_evidence"]["identity"]
        for key in ("prediction_id", "window_id", "patient_id"):
            if str(identity[key]) != str(frozen[key]):
                raise ValueError(
                    f"frozen identity mismatch for prediction {prediction_id}: {key}"
                )
        if private.get("case_category") != frozen["category"]:
            raise ValueError(
                f"private category mismatch for prediction {prediction_id}: "
                f"{private.get('case_category')} != {frozen['category']}"
            )
        if not current["current_evidence"]["provenance"]["dense_evidence_materialized"]:
            raise ValueError(f"current dense evidence is missing for prediction {prediction_id}")
        incomplete_history = [
            int(item["prediction_id"])
            for item in enabled["pdm_context"]["detailed_history"]
            if not item["xai_summary"]["available"]
        ]
        if incomplete_history:
            raise ValueError(
                f"historical dense evidence is missing for predictions {incomplete_history}"
            )
        current_relative = Path("packages") / "current_evidence" / f"prediction_{prediction_id}.json"
        enabled_relative = Path("packages") / "pdm_enabled" / f"prediction_{prediction_id}.json"
        private_relative = Path("private_evaluation") / f"prediction_{prediction_id}.json"
        _write_json(output_root / current_relative, current)
        _write_json(output_root / enabled_relative, enabled)
        _write_json(output_root / private_relative, private)
        derived_cases.append(
            {
                "prediction_id": prediction_id,
                "current_package_path": current_relative.as_posix(),
                "pdm_package_path": enabled_relative.as_posix(),
                "private_evaluation_path": private_relative.as_posix(),
                "current_package_hash": current["package_hash"],
                "pdm_package_hash": enabled["package_hash"],
                "current_evidence_hash": current["current_evidence_hash"],
            }
        )
    manifest["derived_outputs"] = {
        "package_count": len(derived_cases) * 2,
        "private_evaluation_count": len(derived_cases),
        "cases": derived_cases,
    }
    manifest.pop("manifest_hash", None)
    manifest["manifest_hash"] = canonical_json_hash(manifest)
    _write_json(selection_path, manifest)
    return manifest


def finalize_packages_v3(selection_path: Path, output_root: Path) -> dict:
    """Export v3 packages beside the immutable v2 selection and packages."""

    selection_path = Path(selection_path).resolve()
    output_root = Path(output_root).resolve()
    if output_root == selection_path.parent:
        raise ValueError("v3 output root must differ from the frozen v2 package root")
    if (output_root / "case_selection.json").exists():
        raise FileExistsError(
            f"v3 package set is already frozen at {output_root}; refusing to overwrite"
        )
    manifest = json.loads(selection_path.read_text(encoding="utf-8"))
    cases = _validate_frozen_manifest(manifest)
    derived_cases = []
    for frozen in cases:
        prediction_id = int(frozen["prediction_id"])
        current, enabled, private = build_paired_packages_v3(prediction_id)
        identity = current["current_evidence"]["identity"]
        for key in ("prediction_id", "window_id", "patient_id"):
            if str(identity[key]) != str(frozen[key]):
                raise ValueError(
                    f"frozen identity mismatch for prediction {prediction_id}: {key}"
                )
        if private.get("case_category") != frozen["category"]:
            raise ValueError(f"private category mismatch for prediction {prediction_id}")
        current_relative = Path("packages") / "current_evidence" / f"prediction_{prediction_id}.json"
        enabled_relative = Path("packages") / "pdm_enabled" / f"prediction_{prediction_id}.json"
        private_relative = Path("private_evaluation") / f"prediction_{prediction_id}.json"
        _write_json(output_root / current_relative, current)
        _write_json(output_root / enabled_relative, enabled)
        _write_json(output_root / private_relative, private)
        derived_cases.append({
            "prediction_id": prediction_id,
            "current_package_path": current_relative.as_posix(),
            "pdm_package_path": enabled_relative.as_posix(),
            "private_evaluation_path": private_relative.as_posix(),
            "current_package_hash": current["package_hash"],
            "pdm_package_hash": enabled["package_hash"],
            "current_evidence_hash": current["current_evidence_hash"],
        })
    exported = {
        key: value for key, value in manifest.items()
        if key not in {"derived_outputs", "manifest_hash"}
    }
    exported["schema_version"] = "phase9-apnea-case-selection-v3-packages"
    try:
        source_label = selection_path.relative_to(ROOT).as_posix()
    except ValueError:
        source_label = selection_path.as_posix()
    exported["source_selection_path"] = str(source_label)
    exported["derived_outputs"] = {
        "evidence_package_version": 3,
        "package_count": len(derived_cases) * 2,
        "private_evaluation_count": len(derived_cases),
        "cases": derived_cases,
    }
    exported["manifest_hash"] = canonical_json_hash(exported)
    _write_json(output_root / "case_selection.json", exported)
    return exported


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    mode = parser.add_mutually_exclusive_group(required=True)
    mode.add_argument("--freeze-selection", action="store_true")
    mode.add_argument("--finalize-packages", action="store_true")
    mode.add_argument("--finalize-packages-v3", action="store_true")
    parser.add_argument("--output-root", type=Path, default=DEFAULT_OUTPUT_ROOT)
    parser.add_argument("--source-selection", type=Path, default=DEFAULT_OUTPUT_ROOT / "case_selection.json")
    parser.add_argument(
        "--exclude-selection", type=Path, action="append", default=[],
        help="selection manifest whose prediction identities must be excluded; may be repeated",
    )
    return parser


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    if args.freeze_selection:
        excluded = set()
        for exclusion_path in args.exclude_selection:
            prior = json.loads(exclusion_path.read_text(encoding="utf-8"))
            excluded.update(
                int(row["prediction_id"])
                for row in _validate_frozen_manifest(prior)
            )
            excluded.update(
                int(prediction_id)
                for prediction_id in prior.get("selection_core", {}).get(
                    "excluded_development_prediction_ids", []
                )
            )
        manifest = freeze_selection(args.output_root, excluded)
    elif args.finalize_packages:
        manifest = finalize_packages(args.output_root / "case_selection.json")
    else:
        manifest = finalize_packages_v3(args.source_selection, args.output_root)
    print(json.dumps(manifest, indent=2, sort_keys=True, default=_json_default))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
