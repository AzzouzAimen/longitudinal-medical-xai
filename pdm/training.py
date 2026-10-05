"""Deterministic patient-held-out training and artifact integrity helpers.

This module deliberately keeps official annotations in evaluation-only files.
The model-facing arrays contain normalized signals and detector outputs only.
Apnea labels are used to calibrate its development threshold; Sleep uses only
validation-normal reconstruction errors. Official annotations are written to
private evaluation artifacts for each held-out fold.
"""

from __future__ import annotations

import hashlib
import json
import os
import platform
import time
from pathlib import Path
from typing import Any, Iterable

import numpy as np

from .model import AttentionAutoencoder, get_runtime_metadata, set_global_seed

ROOT = Path(__file__).resolve().parents[1]
DATA_ROOT = ROOT / "data"
DEFAULT_OUTPUT_ROOT = ROOT / "outputs" / "detector_training"

APNEA_PATIENTS = ("a01", "a02", "a03", "c01", "c02", "c03")
SLEEP_PATIENTS = ("SC400", "SC401", "SC402")
SLEEP_RECORDINGS = {
    "SC400": ("SC4001E0", "SC4002E0"),
    "SC401": ("SC4011E0", "SC4012E0"),
    "SC402": ("SC4021E0", "SC4022E0"),
}
SLEEP_INVALID_STAGES = frozenset({"?", "Movement time"})
SLEEP_WAKE_MARGIN_EPOCHS = 60  # 30 minutes at one 30-second epoch per window.

# Files that every fold must carry before it can be promoted for downstream
# evidence use.  ``fold_manifest.json`` and ``artifact_hashes.json`` are
# validated separately because each references the other by digest.
REQUIRED_FOLD_ARTIFACTS = frozenset(
    {
        "model.keras",
        "model_config.json",
        "normalization.json",
        "training_history.json",
        "thresholds.json",
        "runtime_metadata.json",
        "held_out_reconstruction.npy",
        "held_out_attention.npy",
        "held_out_residual.npy",
        "held_out_channel_errors.npy",
        "held_out_errors.npy",
        "held_out_decisions.json",
        "evaluation_annotations.json",
    }
)

_DATA_CACHE: dict[str, tuple[np.ndarray, list[dict[str, Any]]]] = {}


def _json_default(value: Any) -> Any:
    if isinstance(value, (np.integer,)):
        return int(value)
    if isinstance(value, (np.floating,)):
        return float(value)
    if isinstance(value, np.ndarray):
        return value.tolist()
    if isinstance(value, Path):
        return str(value)
    raise TypeError(f"cannot serialize {type(value)!r}")


def _write_json(path: Path, payload: Any) -> None:
    path.write_text(
        json.dumps(payload, indent=2, sort_keys=True, default=_json_default) + "\n",
        encoding="utf-8",
    )


def _sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def build_loso_manifest(dataset: str, seed: int = 2026) -> dict:
    """Build the deterministic nested LOSO split manifest.

    The validation patient is the next patient in the documented sorted order,
    wrapping around at the end. This keeps the test patient out of every
    training, normalization, early-stopping, and threshold-calibration scope.
    """

    dataset = str(dataset).lower()
    if dataset not in {"apnea", "sleep"}:
        raise ValueError("dataset must be 'apnea' or 'sleep'")
    patients = list(APNEA_PATIENTS if dataset == "apnea" else SLEEP_PATIENTS)
    folds = []
    for index, test_patient in enumerate(patients):
        validation_patient = patients[(index + 1) % len(patients)]
        train_patients = [p for p in patients if p not in {test_patient, validation_patient}]
        fold: dict[str, Any] = {
            "fold_id": f"{dataset}-{index + 1:02d}-{test_patient}",
            "index": index,
            "dataset": dataset,
            "seed": int(seed),
            "test_patient": test_patient,
            "validation_patient": validation_patient,
            "train_patients": train_patients,
            "validation_patients": [validation_patient],
            "test_recordings": list(SLEEP_RECORDINGS[test_patient] if dataset == "sleep" else (test_patient,)),
            "validation_recordings": list(SLEEP_RECORDINGS[validation_patient] if dataset == "sleep" else (validation_patient,)),
            "train_recordings": [
                recording
                for patient in train_patients
                for recording in (SLEEP_RECORDINGS[patient] if dataset == "sleep" else (patient,))
            ],
            "normalization_patients": list(train_patients),
            "fitting_patients": list(train_patients),
            "early_stopping_patients": [validation_patient],
            "threshold_calibration_patients": [validation_patient],
            "normalization_policy": (
                "official N windows from training patients only"
                if dataset == "apnea"
                else "stable stage 2 epochs without a derived stage transition from training patients only"
            ),
            "threshold_policy": (
                "validation-patient predictions maximize F1; equal-F1 ties choose the larger threshold"
                if dataset == "apnea"
                else "99th percentile of validation-normal stable stage 2 reconstruction errors"
            ),
            "preprocessing_policy": (
                "one-minute annotated windows"
                if dataset == "apnea"
                else "per-night sleep interval plus 30-minute wake margins; unknown and movement epochs excluded"
            ),
            "channel_threshold_policy": (
                "not applicable"
                if dataset == "apnea"
                else "99th percentile of validation-normal per-channel mean reconstruction error"
            ),
            "official_annotations": "evaluation_only; never model evidence or LLM input",
        }
        folds.append(fold)
    limitations = []
    if dataset == "sleep":
        limitations.append("Sleep LOSO has one training patient, one validation patient, and one test patient per fold.")
    return {
        "schema_version": "fusion-loso-v1",
        "dataset": dataset,
        "seed": int(seed),
        "split_strategy": "deterministic cyclic nested patient LOSO",
        "patients": patients,
        "fold_count": len(folds),
        "limitations": limitations,
        "folds": folds,
    }


def _apnea_features(signal: np.ndarray) -> np.ndarray:
    blocks = signal.reshape(100, 60)
    mean = blocks.mean(axis=1)
    std = blocks.std(axis=1)
    low = blocks.min(axis=1)
    high = blocks.max(axis=1)
    slope = (blocks[:, -1] - blocks[:, 0]) / 59.0
    energy = np.mean(blocks * blocks, axis=1)
    return np.stack([mean, std, low, high, high - low, slope, energy], axis=1).astype("float32")


def _sleep_stage_name(description: str) -> str:
    return str(description).replace("Sleep stage ", "").strip()


def select_sleep_epoch_indices(
    stages: Iterable[str], wake_margin_epochs: int = SLEEP_WAKE_MARGIN_EPOCHS
) -> list[int]:
    """Select the meaningful per-night Sleep-EDF interval.

    The retained interval starts 30 minutes before the first scored sleep
    epoch and ends 30 minutes after the last one. Unknown and movement-only
    epochs are never model inputs, even when they occur inside that interval.
    """

    values = [str(stage) for stage in stages]
    margin = int(wake_margin_epochs)
    if margin < 0:
        raise ValueError("wake_margin_epochs must be non-negative")
    sleep = [
        index
        for index, stage in enumerate(values)
        if stage != "W" and stage not in SLEEP_INVALID_STAGES
    ]
    if not sleep:
        return []
    start = max(0, sleep[0] - margin)
    stop = min(len(values), sleep[-1] + margin + 1)
    return [index for index in range(start, stop) if values[index] not in SLEEP_INVALID_STAGES]


def clear_dataset_cache() -> None:
    """Clear the in-process preprocessing cache."""

    _DATA_CACHE.clear()


def load_dataset(dataset: str) -> tuple[np.ndarray, list[dict[str, Any]]]:
    """Load raw, unnormalized windows and evaluation metadata for one dataset."""

    dataset = str(dataset).lower()
    if dataset in _DATA_CACHE:
        return _DATA_CACHE[dataset]
    if dataset == "apnea":
        import wfdb

        windows: list[np.ndarray] = []
        metadata: list[dict[str, Any]] = []
        for patient in APNEA_PATIENTS:
            base = DATA_ROOT / "apnea" / patient
            record = wfdb.rdrecord(str(base))
            annotation = wfdb.rdann(str(base), "apn")
            for index, (sample, label) in enumerate(zip(annotation.sample, annotation.symbol)):
                start = int(sample)
                signal = np.asarray(record.p_signal[start : start + 6000, 0], dtype="float32")
                if len(signal) != 6000:
                    continue
                windows.append(_apnea_features(signal))
                metadata.append({
                    "patient": patient,
                    "recording_id": patient,
                    "window_index": index,
                    "sample_start": start,
                    "sample_end": start + 6000,
                    "official_label": str(label),
                })
        if not windows:
            raise FileNotFoundError("no Apnea-ECG windows found under data/apnea")
        result = np.asarray(windows, dtype="float32"), metadata
    elif dataset == "sleep":
        import mne

        windows = []
        metadata = []
        for patient in SLEEP_PATIENTS:
            for recording_id in SLEEP_RECORDINGS[patient]:
                psg = DATA_ROOT / "sleep_edf" / f"{recording_id}-PSG.edf"
                candidates = sorted((DATA_ROOT / "sleep_edf").glob(f"{recording_id[:-1]}*-Hypnogram.edf"))
                hypnogram = candidates[0] if candidates else DATA_ROOT / "sleep_edf" / f"{recording_id[:-1]}C-Hypnogram.edf"
                if not psg.exists() or not hypnogram.exists():
                    raise FileNotFoundError(f"missing Sleep-EDF pair: {psg.name}, {hypnogram.name}")
                raw = mne.io.read_raw_edf(psg, preload=True, verbose=False)
                annotations = mne.read_annotations(hypnogram)
                wanted = ["EEG Fpz-Cz", "EEG Pz-Oz", "EOG horizontal"]
                missing = [name for name in wanted if name not in raw.ch_names]
                if missing:
                    raise ValueError(f"{recording_id} is missing modeled Sleep channels: {missing}")
                picks = [raw.ch_names.index(name) for name in wanted]
                sample_rate = float(raw.info["sfreq"])
                epoch_samples = int(round(sample_rate * 30.0))
                stages: list[str] = []
                for description, duration in zip(annotations.description, annotations.duration):
                    stages.extend([_sleep_stage_name(description)] * int(round(float(duration) / 30.0)))
                count = min(len(stages), raw.n_times // epoch_samples)
                if count == 0:
                    continue
                data = raw.get_data(picks=picks, start=0, stop=count * epoch_samples)
                data = data.reshape(len(picks), count, epoch_samples).transpose(1, 2, 0)
                data = data[:, ::10, :].astype("float32")
                selected_indices = select_sleep_epoch_indices(stages[:count])
                for index in selected_indices:
                    previous = stages[index - 1] if index else None
                    current = stages[index]
                    windows.append(data[index])
                    metadata.append({
                        "patient": patient,
                        "recording_id": recording_id,
                        "night": recording_id[5:7],
                        "window_index": index,
                        "sample_start": index * epoch_samples,
                        "sample_end": (index + 1) * epoch_samples,
                        "official_label": current,
                        "transition": bool(previous is not None and current != previous),
                    })
                del raw, data
        if not windows:
            raise FileNotFoundError("no Sleep-EDF windows found under data/sleep_edf")
        result = np.asarray(windows, dtype="float32"), metadata
    else:
        raise ValueError("dataset must be 'apnea' or 'sleep'")
    _DATA_CACHE[dataset] = result
    return result


def _patient_mask(metadata: list[dict[str, Any]], patients: Iterable[str]) -> np.ndarray:
    selected = set(patients)
    return np.asarray([item["patient"] in selected for item in metadata], dtype=bool)


def _normal_mask(dataset: str, metadata: list[dict[str, Any]]) -> np.ndarray:
    if dataset == "apnea":
        return np.asarray([item["official_label"] == "N" for item in metadata], dtype=bool)
    return np.asarray([item["official_label"] == "2" and not item.get("transition", False) for item in metadata], dtype=bool)


def fit_normalization(x: np.ndarray, normal_mask: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """Fit feature-wise normalization on the provided normal windows only."""

    x = np.asarray(x, dtype="float32")
    mask = np.asarray(normal_mask, dtype=bool)
    if x.ndim != 3 or len(mask) != len(x):
        raise ValueError("x and normal_mask must describe (windows, timesteps, channels)")
    if not np.any(mask):
        raise ValueError("normalization requires at least one training-normal window")
    mean = x[mask].mean(axis=(0, 1), keepdims=True).astype("float32")
    std = x[mask].std(axis=(0, 1), keepdims=True).astype("float32")
    std = np.where(np.isfinite(std) & (std > 1e-6), std, 1.0).astype("float32")
    return mean, std


def _as_positive_labels(labels: Iterable[Any]) -> np.ndarray:
    positive = {"a", "1", "anomaly", "true", "transition", "positive"}
    values = []
    for label in labels:
        if isinstance(label, (bool, np.bool_)):
            values.append(bool(label))
        elif isinstance(label, (int, float, np.integer, np.floating)):
            values.append(float(label) > 0)
        else:
            values.append(str(label).strip().lower() in positive)
    return np.asarray(values, dtype=bool)


def calibrate_threshold(validation_errors, validation_labels) -> float:
    """Choose the validation threshold maximizing F1, ties favoring larger values."""

    errors = np.asarray(validation_errors, dtype="float64").reshape(-1)
    labels = _as_positive_labels(validation_labels).reshape(-1)
    if len(errors) == 0 or len(errors) != len(labels):
        raise ValueError("validation errors and labels must be non-empty and have equal length")
    if not np.all(np.isfinite(errors)):
        raise ValueError("validation errors must be finite")
    candidates = sorted(float(value) for value in np.unique(errors))
    # With all-positive validation labels, an initial threshold below the
    # smallest observed error is the only candidate that predicts every
    # validation window positive.  ``nextafter`` keeps the candidate exact and
    # deterministic without introducing a data-dependent arbitrary epsilon.
    below_minimum = float(np.nextafter(candidates[0], -np.inf))
    if below_minimum > 0:
        candidates.insert(0, below_minimum)
    best_threshold = candidates[0]
    best_f1 = -1.0
    for threshold in candidates:
        predicted = errors > threshold
        true_positive = int(np.sum(predicted & labels))
        false_positive = int(np.sum(predicted & ~labels))
        false_negative = int(np.sum(~predicted & labels))
        denominator = 2 * true_positive + false_positive + false_negative
        f1 = (2 * true_positive / denominator) if denominator else 0.0
        if f1 > best_f1 or (f1 == best_f1 and threshold > best_threshold):
            best_f1 = f1
            best_threshold = threshold
    # Keep the selected positive value exactly, including tiny positive
    # below-minimum candidates. If the minimum error is zero, no positive
    # below-minimum candidate exists; use the smallest positive float instead
    # of returning an invalid non-positive normalization denominator.
    if best_threshold <= 0:
        best_threshold = float(np.nextafter(0.0, np.inf))
    return float(best_threshold)


def calibrate_normal_threshold(
    validation_errors, validation_normal_mask, percentile: float = 99
) -> float:
    """Calibrate an unsupervised threshold from validation-normal errors only."""

    errors = np.asarray(validation_errors, dtype="float64").reshape(-1)
    normal = np.asarray(validation_normal_mask, dtype=bool).reshape(-1)
    if len(errors) == 0 or len(errors) != len(normal):
        raise ValueError("validation errors and normal mask must be non-empty and have equal length")
    if not np.all(np.isfinite(errors)):
        raise ValueError("validation errors must be finite")
    if not np.any(normal):
        raise ValueError("threshold calibration requires validation-normal errors")
    percentile = float(percentile)
    if not 0 < percentile <= 100:
        raise ValueError("percentile must be in (0, 100]")
    threshold = float(np.percentile(errors[normal], percentile))
    if threshold <= 0:
        threshold = float(np.nextafter(0.0, np.inf))
    return threshold


def _fit_fold_model(
    train_x: np.ndarray,
    validation_normal_x: np.ndarray,
    seed: int,
    epochs: int = 20,
    batch_size: int = 64,
    patience: int = 3,
) -> tuple[AttentionAutoencoder, dict[str, list[float]]]:
    set_global_seed(seed)
    model = AttentionAutoencoder(epochs=epochs, batch_size=batch_size, early_stopping_patience=patience)
    model._build(tuple(train_x.shape[1:]))
    import tensorflow as tf

    monitor = "val_loss" if len(validation_normal_x) else "loss"
    callback = tf.keras.callbacks.EarlyStopping(monitor=monitor, patience=patience, restore_best_weights=True)
    kwargs: dict[str, Any] = {"epochs": epochs, "batch_size": batch_size, "shuffle": True, "callbacks": [callback], "verbose": 0}
    if len(validation_normal_x):
        kwargs["validation_data"] = (validation_normal_x, validation_normal_x)
    history = model.network.fit(train_x, train_x, **kwargs)
    model.history = history
    model.runtime_metadata = get_runtime_metadata(seed)
    model.network.get_layer("runtime_metadata").metadata = dict(model.runtime_metadata)
    return model, {key: [float(v) for v in values] for key, values in history.history.items()}


def _relative_artifact(path: Path, root: Path) -> str:
    return path.relative_to(root).as_posix()


def _artifact_hashes(root: Path) -> dict[str, str]:
    hashes = {}
    for path in sorted(root.rglob("*")):
        if path.is_file() and path.name not in {"artifact_hashes.json", "fold_manifest.json"}:
            hashes[_relative_artifact(path, root)] = _sha256_file(path)
    return hashes


def train_fold(dataset: str, fold: dict, output_dir: Path) -> dict:
    """Fit one fold and serialize model/evaluation artifacts."""

    dataset = str(dataset).lower()
    if dataset not in {"apnea", "sleep"}:
        raise ValueError("dataset must be 'apnea' or 'sleep'")
    supplied_test = fold.get("test_patient")
    if supplied_test in set(fold.get("train_patients", [])) | set(fold.get("validation_patients", [])):
        raise ValueError(f"patient leakage in supplied fold provenance: {supplied_test}")
    expected = build_loso_manifest(dataset, seed=int(fold.get("seed", 2026)))
    expected_fold = next((item for item in expected["folds"] if item["test_patient"] == supplied_test), None)
    if expected_fold is None:
        raise ValueError(f"unknown {dataset} test patient: {supplied_test}")
    fold = expected_fold
    fold_dir = Path(output_dir)
    fold_dir.mkdir(parents=True, exist_ok=True)
    # Invalidate any prior validated marker before replacing a fold. The root
    # manifest is also marked provisional by the CLI before the first fold, so
    # a partial rerun can never be consumed as final evidence.
    prior_manifest = fold_dir / "fold_manifest.json"
    if prior_manifest.exists():
        try:
            prior = json.loads(prior_manifest.read_text(encoding="utf-8"))
        except (OSError, json.JSONDecodeError):
            prior = None
        if isinstance(prior, dict):
            prior["artifact_status"] = "provisional"
            _write_json(prior_manifest, prior)
    started = time.perf_counter()
    x, metadata = load_dataset(dataset)
    train_patient_mask = _patient_mask(metadata, fold["train_patients"])
    validation_patient_mask = _patient_mask(metadata, fold["validation_patients"])
    test_patient_mask = _patient_mask(metadata, [fold["test_patient"]])
    normal = _normal_mask(dataset, metadata)
    train_normal_mask = train_patient_mask & normal
    validation_normal_mask = validation_patient_mask & normal
    if not np.any(train_normal_mask) or not np.any(validation_normal_mask):
        raise ValueError(f"{fold['fold_id']} lacks train/validation normal windows: train={int(train_normal_mask.sum())}, validation={int(validation_normal_mask.sum())}")
    mean, std = fit_normalization(x, train_normal_mask)
    normalized = ((x - mean) / std).astype("float32")
    train_x = normalized[train_normal_mask]
    validation_normal_x = normalized[validation_normal_mask]
    validation_all_x = normalized[validation_patient_mask]
    test_x = normalized[test_patient_mask]
    validation_labels = [
        metadata[index]["official_label"]
        for index in np.flatnonzero(validation_patient_mask)
    ]
    test_metadata = [metadata[index] for index in np.flatnonzero(test_patient_mask)]
    epochs = int(os.getenv("PDM_FUSION_EPOCHS", "20"))
    batch_size = int(os.getenv("PDM_FUSION_BATCH_SIZE", "64"))
    patience = int(os.getenv("PDM_FUSION_PATIENCE", "3"))
    model, history = _fit_fold_model(train_x, validation_normal_x, seed=int(fold["seed"]), epochs=epochs, batch_size=batch_size, patience=patience)
    validation_errors, _, _ = model.predict(validation_all_x)
    if dataset == "sleep":
        validation_normal_in_validation = normal[validation_patient_mask]
        threshold = calibrate_normal_threshold(
            validation_errors, validation_normal_in_validation, percentile=99
        )
    else:
        threshold = calibrate_threshold(validation_errors, validation_labels)
    model.threshold = threshold
    test_result = model.results(test_x)
    residual = test_x - test_result["reconstruction"]
    # Decisions and margins must be derived from the exact float32 errors that
    # are persisted. Computing them from a higher-precision transient value can
    # leave the JSON and NPY artifacts disagreeing close to the threshold.
    persisted_errors = np.asarray(test_result["reconstruction_error"], dtype="float32")
    persisted_margins = persisted_errors.astype("float64") - float(threshold)
    persisted_normalized_margins = persisted_margins / float(threshold)
    persisted_is_anomaly = persisted_errors.astype("float64") > float(threshold)
    persisted_decisions = np.where(persisted_is_anomaly, "anomaly", "normal").tolist()
    channel_errors = np.mean(np.square(residual), axis=1)
    channel_thresholds: list[float] = []
    if dataset == "sleep":
        _, validation_normal_reconstruction, _ = model.predict(validation_normal_x)
        validation_normal_channel_errors = np.mean(np.square(validation_normal_x - validation_normal_reconstruction), axis=1)
        channel_thresholds = [float(value) for value in np.percentile(validation_normal_channel_errors, 99, axis=0)]
    checkpoint = fold_dir / "model.keras"
    model.save(checkpoint)
    np.save(fold_dir / "held_out_reconstruction.npy", test_result["reconstruction"].astype("float32"))
    np.save(fold_dir / "held_out_attention.npy", test_result["attention"].astype("float32"))
    np.save(fold_dir / "held_out_residual.npy", residual.astype("float32"))
    np.save(fold_dir / "held_out_channel_errors.npy", channel_errors.astype("float32"))
    np.save(fold_dir / "held_out_errors.npy", persisted_errors)
    _write_json(fold_dir / "held_out_decisions.json", {
        "decision": persisted_decisions,
        "is_anomaly": [bool(value) for value in persisted_is_anomaly],
        "anomaly_margin": persisted_margins.tolist(),
        "normalized_margin": persisted_normalized_margins.tolist(),
    })
    _write_json(fold_dir / "evaluation_annotations.json", {"evaluation_only": True, "private_model_development_data": True, "test_patient": fold["test_patient"], "annotations": test_metadata})
    _write_json(fold_dir / "normalization.json", {"fit_scope": "training-normal windows only", "normalization_patients": fold["normalization_patients"], "mean": mean.reshape(-1).tolist(), "std": std.reshape(-1).tolist(), "train_normal_window_count": int(train_normal_mask.sum())})
    _write_json(fold_dir / "model_config.json", {"architecture": "BiLSTM temporal-attention autoencoder", "encoder_bilstm_units": [32, 16], "latent_dimension": 16, "decoder_lstm_units": [16, 32], "input_shape": list(train_x.shape[1:]), "loss": "clinical_pattern_loss", "loss_weights": {"pattern": 0.3, "trend": 0.4, "fidelity": 0.3}, "epochs": epochs, "batch_size": batch_size, "early_stopping_patience": patience, "seed": int(fold["seed"])})
    _write_json(fold_dir / "training_history.json", history)
    _write_json(fold_dir / "thresholds.json", {
        "overall": threshold,
        "overall_selection": (
            "validation F1 maximum"
            if dataset == "apnea"
            else "99th percentile of validation-normal stable stage 2 reconstruction errors"
        ),
        "overall_percentile": 99 if dataset == "sleep" else None,
        "overall_semantics": (
            "reconstruction outlier relative to stable stage 2 sleep; not a transition or clinical diagnosis"
            if dataset == "sleep"
            else "validation-label-calibrated reconstruction anomaly; not a probability"
        ),
        "tie_breaking": "larger threshold" if dataset == "apnea" else None,
        "validation_patient": fold["validation_patient"],
        "channel_thresholds": channel_thresholds,
        "channel_threshold_percentile": 99 if dataset == "sleep" else None,
        "channel_threshold_semantics": "mean per-channel reconstruction error, not probability",
    })
    runtime = dict(get_runtime_metadata(int(fold["seed"])))
    runtime.update({"platform": platform.platform(), "elapsed_seconds": time.perf_counter() - started, "train_windows": int(len(train_x)), "validation_normal_windows": int(len(validation_normal_x)), "validation_windows": int(len(validation_all_x)), "held_out_windows": int(len(test_x)), "resource_note": "CPU-safe prototype; exact bitwise reproducibility is not guaranteed across TensorFlow builds."})
    _write_json(fold_dir / "runtime_metadata.json", runtime)
    fold_manifest = dict(fold)
    fold_manifest.update({"schema_version": "fusion-fold-v1", "artifact_status": "provisional", "artifact_policy": "official annotations are evaluation-only and separate from model evidence", "model_checkpoint": "model.keras", "artifact_hashes_file": "artifact_hashes.json", "normalization_file": "normalization.json", "held_out_outputs": {"reconstruction": "held_out_reconstruction.npy", "attention": "held_out_attention.npy", "residual": "held_out_residual.npy", "channel_errors": "held_out_channel_errors.npy", "errors": "held_out_errors.npy", "decisions": "held_out_decisions.json"}, "evaluation_annotations_file": "evaluation_annotations.json"})
    _write_json(fold_dir / "fold_manifest.json", fold_manifest)
    hashes = _artifact_hashes(fold_dir)
    _write_json(fold_dir / "artifact_hashes.json", hashes)
    fold_manifest["artifact_hashes"] = hashes
    fold_manifest["model_checkpoint_sha256"] = hashes["model.keras"]
    fold_manifest["artifact_hashes_sha256"] = _sha256_file(fold_dir / "artifact_hashes.json")
    _write_json(fold_dir / "fold_manifest.json", fold_manifest)
    return {"dataset": dataset, "seed": int(fold["seed"]), "test_patient": fold["test_patient"], "fold_dir": str(fold_dir), "fold_manifest": str(fold_dir / "fold_manifest.json"), "model_checkpoint": str(checkpoint), "model_checkpoint_sha256": _sha256_file(checkpoint), "held_out_windows": int(len(test_x))}


def write_out_of_fold_manifest(dataset: str, fold_results: list[dict[str, Any]], output_dir: Path | None = None) -> Path:
    """Write the root manifest linking every held-out fold and its hashes."""

    dataset = str(dataset).lower()
    seed = int(fold_results[0].get("seed", 2026)) if fold_results else 2026
    expected = build_loso_manifest(dataset, seed=seed)
    root = Path(output_dir) if output_dir is not None else DEFAULT_OUTPUT_ROOT / dataset
    root.mkdir(parents=True, exist_ok=True)
    rows = []
    for result in sorted(fold_results, key=lambda item: item["test_patient"]):
        fold_manifest = Path(result["fold_manifest"])
        if not fold_manifest.exists():
            raise FileNotFoundError(f"missing fold manifest: {fold_manifest}")
        fold_dir = fold_manifest.parent
        model_path = fold_dir / "model.keras"
        rows.append({"test_patient": result["test_patient"], "fold_manifest": _relative_artifact(fold_manifest, root), "fold_manifest_sha256": _sha256_file(fold_manifest), "model_checkpoint": _relative_artifact(model_path, root), "model_checkpoint_sha256": _sha256_file(model_path), "artifact_hashes": _relative_artifact(fold_dir / "artifact_hashes.json", root)})
    manifest = {"schema_version": "fusion-loso-manifest-v1", "artifact_status": "provisional", "dataset": dataset, "seed": expected["seed"], "patients": expected["patients"], "fold_count": len(rows), "split_strategy": expected["split_strategy"], "limitations": expected["limitations"], "folds": rows, "provenance": "all fold/model hashes are SHA-256 and official annotations remain evaluation-only; provisional artifacts are not valid downstream evidence until --validate-only succeeds"}
    path = root / "loso_manifest.json"
    _write_json(path, manifest)
    return path


def _promote_validated(root: Path, root_manifest: dict[str, Any]) -> None:
    """Mark a fully validated root/fold set as safe for downstream use."""

    for row in root_manifest["folds"]:
        fold_path = root / row["fold_manifest"]
        fold = json.loads(fold_path.read_text(encoding="utf-8"))
        fold["artifact_status"] = "validated"
        _write_json(fold_path, fold)
        row["fold_manifest_sha256"] = _sha256_file(fold_path)
    root_manifest["artifact_status"] = "validated"
    root_manifest["validation_note"] = "All required fold/model/artifact hashes and cyclic provenance were verified."
    _write_json(root / "loso_manifest.json", root_manifest)


def validate_loso_artifacts(
    dataset: str,
    output_dir: Path | None = None,
    promote: bool = False,
) -> dict[str, Any]:
    """Validate split provenance and all fold/model hashes, raising on corruption.

    ``promote=True`` is used only by the validation CLI after every check has
    passed. Training writes ``provisional`` manifests so downstream consumers
    cannot treat an unvalidated or partial output as final evidence.
    """

    dataset = str(dataset).lower()
    root = Path(output_dir) if output_dir is not None else DEFAULT_OUTPUT_ROOT / dataset
    root_manifest_path = root / "loso_manifest.json"
    if not root_manifest_path.exists():
        raise FileNotFoundError(f"missing root LOSO manifest: {root_manifest_path}")
    root_manifest = json.loads(root_manifest_path.read_text(encoding="utf-8"))
    expected = build_loso_manifest(dataset, seed=int(root_manifest.get("seed", 2026)))
    if root_manifest.get("dataset") != dataset:
        raise ValueError("root manifest dataset mismatch")
    if set(root_manifest.get("patients", [])) != set(expected["patients"]):
        raise ValueError("root manifest patient coverage mismatch")
    rows = root_manifest.get("folds", [])
    if len(rows) != len(expected["patients"]):
        raise ValueError("root manifest does not contain one fold per patient")
    expected_by_patient = {fold["test_patient"]: fold for fold in expected["folds"]}
    artifact_root = root.resolve()

    def contained_path(reference: str, label: str) -> Path:
        """Resolve a manifest reference while forbidding root escape."""

        candidate = (root / reference).resolve()
        try:
            candidate.relative_to(artifact_root)
        except ValueError as exc:
            raise ValueError(
                f"{label} path escapes artifact root: {reference}"
            ) from exc
        return candidate

    fold_refs: list[str] = []
    model_refs: list[str] = []
    hash_refs: list[str] = []
    for row in rows:
        for key, seen, label in (
            ("fold_manifest", fold_refs, "fold"),
            ("model_checkpoint", model_refs, "model"),
            ("artifact_hashes", hash_refs, "artifact-hash"),
        ):
            reference = row.get(key)
            if not isinstance(reference, str) or not reference:
                raise ValueError(f"missing {label} path for {row.get('test_patient')}")
            resolved = contained_path(reference, label)
            if str(resolved) in seen:
                raise ValueError(f"unique {label} paths required")
            seen.append(str(resolved))
    held_out = []
    for row in rows:
        patient = row.get("test_patient")
        if patient in held_out:
            raise ValueError(f"patient appears in multiple held-out folds: {patient}")
        held_out.append(patient)
        fold_path = contained_path(row.get("fold_manifest", ""), "fold")
        if not fold_path.exists() or not row.get("fold_manifest_sha256"):
            raise ValueError(f"missing fold manifest/hash for {patient}")
        if _sha256_file(fold_path) != row["fold_manifest_sha256"]:
            raise ValueError(f"fold manifest hash mismatch for {patient}")
        fold = json.loads(fold_path.read_text(encoding="utf-8"))
        if fold.get("test_patient") != patient:
            raise ValueError(f"root row test_patient does not match referenced fold for {patient}")
        if patient not in expected["patients"]:
            raise ValueError(f"unexpected held-out patient: {patient}")
        expected_fold = expected_by_patient[patient]
        for field in (
            "validation_patient",
            "train_patients",
            "validation_patients",
            "test_recordings",
            "validation_recordings",
            "train_recordings",
            "normalization_patients",
            "fitting_patients",
            "early_stopping_patients",
            "threshold_calibration_patients",
        ):
            if fold.get(field) != expected_fold.get(field):
                raise ValueError(f"unexpected cyclic split in fold {patient}: {field}")
        if patient in fold.get("train_patients", []) or patient in fold.get("validation_patients", []):
            raise ValueError(f"patient leakage in fold provenance: {patient}")
        model_path = contained_path(row.get("model_checkpoint", ""), "model")
        if not row.get("model_checkpoint_sha256") or not model_path.exists():
            raise ValueError(f"missing model checkpoint/hash for {patient}")
        if _sha256_file(model_path) != row["model_checkpoint_sha256"]:
            raise ValueError(f"model checkpoint hash mismatch for {patient}")
        if fold.get("model_checkpoint") != model_path.relative_to(fold_path.parent).as_posix():
            raise ValueError(f"root row model path does not match referenced fold for {patient}")
        if fold.get("model_checkpoint_sha256") != row["model_checkpoint_sha256"]:
            raise ValueError(f"fold/model hash disagreement for {patient}")
        hashes_path = contained_path(row.get("artifact_hashes", ""), "artifact-hash")
        if not hashes_path.exists():
            raise ValueError(f"missing artifact hash file for {patient}")
        if fold.get("artifact_hashes_file") != hashes_path.relative_to(fold_path.parent).as_posix():
            raise ValueError(f"root row artifact-hash path does not match referenced fold for {patient}")
        if not fold.get("artifact_hashes_sha256"):
            raise ValueError(f"missing artifact_hashes sha256 for {patient}")
        if _sha256_file(hashes_path) != fold["artifact_hashes_sha256"]:
            raise ValueError(f"artifact_hashes sha256 mismatch for {patient}")
        hashes = json.loads(hashes_path.read_text(encoding="utf-8"))
        if fold.get("artifact_hashes") != hashes:
            raise ValueError(f"embedded artifact hash map mismatch for {patient}")
        missing_required = REQUIRED_FOLD_ARTIFACTS - set(hashes)
        if missing_required:
            missing = ", ".join(sorted(missing_required))
            raise ValueError(f"missing required artifact(s) for {patient}: {missing}")
        for relative_path, expected_hash in hashes.items():
            artifact = (hashes_path.parent / relative_path).resolve()
            try:
                artifact.relative_to(hashes_path.parent.resolve())
            except ValueError as exc:
                raise ValueError(
                    f"artifact path escapes fold root: {relative_path}"
                ) from exc
            if not artifact.exists() or _sha256_file(artifact) != expected_hash:
                raise ValueError(f"artifact hash mismatch for {patient}: {relative_path}")
    if set(held_out) != set(expected["patients"]):
        raise ValueError("held-out patient coverage is incomplete")
    if promote:
        _promote_validated(root, root_manifest)
    return {"dataset": dataset, "fold_count": len(rows), "patients": sorted(held_out), "root_manifest": str(root_manifest_path), "artifact_status": "validated" if promote else root_manifest.get("artifact_status", "provisional"), "valid": True}
