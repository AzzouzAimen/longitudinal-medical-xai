"""Train and validate deterministic Apnea/Sleep fusion LOSO artifacts."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

from pdm.training import (
    DEFAULT_OUTPUT_ROOT,
    build_loso_manifest,
    train_fold,
    validate_loso_artifacts,
    write_out_of_fold_manifest,
)


def _datasets(value: str) -> list[str]:
    value = value.lower()
    if value == "all":
        return ["apnea", "sleep"]
    if value not in {"apnea", "sleep"}:
        raise argparse.ArgumentTypeError("dataset must be apnea, sleep, or all")
    return [value]


def train_dataset(dataset: str, seed: int, output_root: Path) -> dict:
    manifest = build_loso_manifest(dataset, seed=seed)
    dataset_root = output_root / dataset
    folds_root = dataset_root / "folds"
    # Existing validated outputs become provisional immediately when a rerun
    # starts, preventing downstream consumers from reading a mixed generation.
    write_out_of_fold_manifest(dataset, [], dataset_root)
    results = []
    for fold in manifest["folds"]:
        fold_dir = folds_root / fold["test_patient"]
        results.append(train_fold(dataset, fold, fold_dir))
    root_manifest = write_out_of_fold_manifest(dataset, results, dataset_root)
    return {
        "dataset": dataset,
        "fold_count": len(results),
        "patients": [result["test_patient"] for result in results],
        "manifest": str(root_manifest),
    }


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dataset", required=True, choices=["apnea", "sleep", "all"])
    parser.add_argument("--loso", action="store_true", help="train all deterministic patient-held-out folds")
    parser.add_argument("--validate-only", action="store_true", help="validate existing fold/model hashes and provenance")
    parser.add_argument("--seed", type=int, default=2026)
    parser.add_argument("--output-root", type=Path, default=DEFAULT_OUTPUT_ROOT)
    args = parser.parse_args(argv)
    if args.loso == args.validate_only:
        parser.error("specify exactly one of --loso or --validate-only")
    datasets = _datasets(args.dataset)
    try:
        if args.validate_only:
            summary = [validate_loso_artifacts(dataset, args.output_root / dataset, promote=True) for dataset in datasets]
        else:
            summary = [train_dataset(dataset, args.seed, args.output_root) for dataset in datasets]
    except Exception as exc:  # CLI must expose the exact blocker and return non-zero.
        print(f"ERROR: {type(exc).__name__}: {exc}", file=sys.stderr)
        return 1
    print(json.dumps(summary, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
