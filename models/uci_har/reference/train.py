"""UCI HAR reference training pipeline.

This script trains a compact temporal-attention autoencoder on locally supplied
UCI HAR inertial signals. Normal activity IDs are mandatory CLI inputs. The
threshold is selected from normal validation subjects; test labels are mapped
to normal/anomaly using explicit CLI ID sets.
"""

from __future__ import annotations

import argparse
import csv
import json
import math
import random
from pathlib import Path
from typing import Dict, Iterable, List, Optional, Sequence, Tuple

import numpy as np
import torch
from torch import nn
from torch.utils.data import DataLoader, TensorDataset


SEQUENCE_LENGTH = 128
FEATURE_NAMES = (
    "body_acc_x",
    "body_acc_y",
    "body_acc_z",
    "body_gyro_x",
    "body_gyro_y",
    "body_gyro_z",
    "total_acc_x",
    "total_acc_y",
    "total_acc_z",
)
FEATURE_COUNT = len(FEATURE_NAMES)


def parse_activity_ids(value: str) -> Tuple[int, ...]:
    """Parse a comma-separated set of positive UCI HAR activity IDs."""
    try:
        ids = tuple(int(part.strip()) for part in value.split(",") if part.strip())
    except ValueError as exc:
        raise argparse.ArgumentTypeError(
            "activity IDs must be comma-separated integers, for example 1,2"
        ) from exc
    if not ids:
        raise argparse.ArgumentTypeError("at least one activity ID is required")
    if any(activity_id < 1 for activity_id in ids):
        raise argparse.ArgumentTypeError("activity IDs must be positive integers")
    if len(set(ids)) != len(ids):
        raise argparse.ArgumentTypeError("activity IDs must not contain duplicates")
    return ids


def load_split(dataset_root: Path, split: str) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Load official UCI HAR inertial text files, subject IDs, and activity IDs."""
    if split not in ("train", "test"):
        raise ValueError("split must be 'train' or 'test'")

    signal_dir = dataset_root / split / "Inertial Signals"
    channel_arrays: List[np.ndarray] = []
    expected_rows: Optional[int] = None

    for feature_name in FEATURE_NAMES:
        path = signal_dir / f"{feature_name}_{split}.txt"
        if not path.is_file():
            raise FileNotFoundError(f"missing UCI HAR signal file: {path}")
        values = np.loadtxt(path, dtype=np.float32)
        values = np.atleast_2d(values)
        if values.ndim != 2 or values.shape[1] != SEQUENCE_LENGTH:
            raise ValueError(
                f"{path} must contain rows of {SEQUENCE_LENGTH} values; "
                f"found shape {values.shape}"
            )
        if expected_rows is None:
            expected_rows = values.shape[0]
        elif values.shape[0] != expected_rows:
            raise ValueError(f"signal row counts do not match in {signal_dir}")
        channel_arrays.append(values)

    if expected_rows is None or expected_rows == 0:
        raise ValueError(f"no signal windows found in {signal_dir}")

    signal = np.stack(channel_arrays, axis=-1).astype(np.float32, copy=False)
    label_path = dataset_root / split / f"y_{split}.txt"
    subject_path = dataset_root / split / f"subject_{split}.txt"
    if not label_path.is_file():
        raise FileNotFoundError(f"missing UCI HAR label file: {label_path}")
    if not subject_path.is_file():
        raise FileNotFoundError(f"missing UCI HAR subject file: {subject_path}")

    labels = np.asarray(np.loadtxt(label_path, dtype=np.int64)).reshape(-1)
    subjects = np.asarray(np.loadtxt(subject_path, dtype=np.int64)).reshape(-1)
    if len(labels) != len(signal) or len(subjects) != len(signal):
        raise ValueError(
            f"row count mismatch for {split}: signals={len(signal)}, "
            f"labels={len(labels)}, subjects={len(subjects)}"
        )
    if not np.isfinite(signal).all():
        raise ValueError(f"non-finite signal values found in {signal_dir}")
    return signal, labels, subjects


def split_training_subjects(
    subjects: np.ndarray, validation_fraction: float, seed: int
) -> Tuple[np.ndarray, np.ndarray]:
    """Return disjoint subject IDs for fit and validation partitions."""
    if not 0.0 < validation_fraction < 1.0:
        raise ValueError("validation_fraction must be between 0 and 1")
    unique_subjects = np.unique(subjects)
    if len(unique_subjects) < 2:
        raise ValueError("at least two training subjects are needed for validation")

    shuffled = unique_subjects.copy()
    np.random.default_rng(seed).shuffle(shuffled)
    n_validation = int(math.ceil(len(shuffled) * validation_fraction))
    n_validation = min(max(n_validation, 1), len(shuffled) - 1)
    validation_subjects = np.sort(shuffled[:n_validation])
    fit_subjects = np.sort(shuffled[n_validation:])
    if np.intersect1d(fit_subjects, validation_subjects).size:
        raise RuntimeError("subject split unexpectedly overlaps")
    return fit_subjects, validation_subjects


def fit_normalization(training_windows: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
    """Fit per-channel mean and scale using only fit-partition normal windows."""
    mean = training_windows.mean(axis=(0, 1), dtype=np.float64).astype(np.float32)
    scale = training_windows.std(axis=(0, 1), dtype=np.float64).astype(np.float32)
    scale = np.where(scale < 1e-8, 1.0, scale).astype(np.float32)
    return mean, scale


def normalize(signal: np.ndarray, mean: np.ndarray, scale: np.ndarray) -> np.ndarray:
    return ((signal - mean.reshape(1, 1, -1)) / scale.reshape(1, 1, -1)).astype(
        np.float32, copy=False
    )


class TemporalAttentionAutoencoder(nn.Module):
    """Small bidirectional LSTM autoencoder with learned temporal attention."""

    def __init__(self, feature_count: int = FEATURE_COUNT, hidden_size: int = 32):
        super().__init__()
        self.encoder = nn.LSTM(
            input_size=feature_count,
            hidden_size=hidden_size,
            batch_first=True,
            bidirectional=True,
        )
        self.attention_score = nn.Linear(hidden_size * 2, 1)
        self.context_projection = nn.Linear(hidden_size * 2, hidden_size * 2)
        self.decoder = nn.LSTM(
            input_size=hidden_size * 2,
            hidden_size=hidden_size,
            batch_first=True,
            bidirectional=True,
        )
        self.output_projection = nn.Linear(hidden_size * 2, feature_count)

    def forward(self, signal: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        encoded, _ = self.encoder(signal)
        weights = torch.softmax(self.attention_score(encoded).squeeze(-1), dim=1)
        context = torch.bmm(weights.unsqueeze(1), encoded).squeeze(1)
        latent = torch.tanh(self.context_projection(context))
        repeated = latent.unsqueeze(1).expand(-1, signal.shape[1], -1).contiguous()
        decoded, _ = self.decoder(repeated)
        reconstruction = self.output_projection(decoded)
        return reconstruction, weights


def make_loader(
    values: np.ndarray, batch_size: int, shuffle: bool, seed: int
) -> DataLoader:
    dataset = TensorDataset(torch.from_numpy(values))
    generator = torch.Generator().manual_seed(seed)
    return DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=shuffle,
        drop_last=False,
        num_workers=0,
        generator=generator if shuffle else None,
    )


def run_epoch(
    model: nn.Module,
    loader: DataLoader,
    device: torch.device,
    optimizer: Optional[torch.optim.Optimizer] = None,
) -> float:
    training = optimizer is not None
    model.train(training)
    total_squared_error = 0.0
    total_windows = 0

    with torch.set_grad_enabled(training):
        for (batch,) in loader:
            batch = batch.to(device)
            if training:
                optimizer.zero_grad(set_to_none=True)
            reconstruction, _ = model(batch)
            per_window_mse = (reconstruction - batch).square().mean(dim=(1, 2))
            loss = per_window_mse.mean()
            if training:
                loss.backward()
                optimizer.step()
            total_squared_error += float(per_window_mse.detach().sum().cpu())
            total_windows += int(batch.shape[0])

    if total_windows == 0:
        raise ValueError("cannot compute loss for an empty data split")
    return total_squared_error / total_windows


def score_windows(
    model: nn.Module, values: np.ndarray, batch_size: int, device: torch.device
) -> Tuple[np.ndarray, np.ndarray]:
    """Return per-window reconstruction MSE and temporal attention weights."""
    loader = make_loader(values, batch_size=batch_size, shuffle=False, seed=0)
    model.eval()
    scores: List[np.ndarray] = []
    attentions: List[np.ndarray] = []
    with torch.no_grad():
        for (batch,) in loader:
            batch = batch.to(device)
            reconstruction, weights = model(batch)
            error = (reconstruction - batch).square().mean(dim=(1, 2))
            scores.append(error.cpu().numpy())
            attentions.append(weights.cpu().numpy())
    return np.concatenate(scores), np.concatenate(attentions)


def validate_activity_mapping(
    normal_ids: Sequence[int], anomaly_ids: Sequence[int], test_labels: np.ndarray
) -> None:
    normal = set(normal_ids)
    anomaly = set(anomaly_ids)
    if not normal or not anomaly:
        raise ValueError("normal and anomaly activity ID sets must both be non-empty")
    overlap = normal & anomaly
    if overlap:
        raise ValueError(f"activity IDs cannot be both normal and anomaly: {sorted(overlap)}")
    observed = set(int(value) for value in np.unique(test_labels))
    unmapped = observed - normal - anomaly
    if unmapped:
        raise ValueError(
            f"test activities {sorted(unmapped)} are unmapped; include each in "
            "--normal-activities or --anomaly-activities"
        )


def binary_metrics(y_true: np.ndarray, y_pred: np.ndarray) -> Dict[str, float]:
    tp = int(np.sum((y_true == 1) & (y_pred == 1)))
    fp = int(np.sum((y_true == 0) & (y_pred == 1)))
    fn = int(np.sum((y_true == 1) & (y_pred == 0)))
    tn = int(np.sum((y_true == 0) & (y_pred == 0)))
    precision = tp / (tp + fp) if tp + fp else 0.0
    recall = tp / (tp + fn) if tp + fn else 0.0
    f1 = (
        2.0 * precision * recall / (precision + recall)
        if precision + recall
        else 0.0
    )
    total = tp + fp + fn + tn
    accuracy = (tp + tn) / total if total else 0.0
    return {
        "true_positive": tp,
        "false_positive": fp,
        "false_negative": fn,
        "true_negative": tn,
        "precision": precision,
        "recall": recall,
        "f1": f1,
        "accuracy": accuracy,
        "window_count": total,
    }


def resolve_device(device_name: str) -> torch.device:
    if device_name == "auto":
        if torch.cuda.is_available():
            return torch.device("cuda")
        mps_backend = getattr(torch.backends, "mps", None)
        if mps_backend is not None and mps_backend.is_available():
            return torch.device("mps")
        return torch.device("cpu")
    device = torch.device(device_name)
    if device.type == "cuda" and not torch.cuda.is_available():
        raise ValueError("CUDA was requested but is unavailable")
    if device.type == "mps":
        mps_backend = getattr(torch.backends, "mps", None)
        if mps_backend is None or not mps_backend.is_available():
            raise ValueError("MPS was requested but is unavailable")
    return device


def write_training_history(path: Path, history: Iterable[Tuple[int, float, float]]) -> None:
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.writer(handle)
        writer.writerow(("epoch", "training_mse", "validation_mse"))
        writer.writerows(history)


def train_and_evaluate(args: argparse.Namespace) -> Dict[str, object]:
    normal_ids = tuple(args.normal_activities)
    anomaly_ids = tuple(args.anomaly_activities)
    if set(normal_ids) & set(anomaly_ids):
        raise ValueError("normal and anomaly activity IDs must be disjoint")

    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(args.seed)
    device = resolve_device(args.device)

    dataset_root = Path(args.dataset_root).expanduser().resolve()
    train_signal, train_labels, train_subjects = load_split(dataset_root, "train")
    test_signal, test_labels, test_subjects = load_split(dataset_root, "test")
    shared_subjects = np.intersect1d(train_subjects, test_subjects)
    if shared_subjects.size:
        raise ValueError(
            f"training and test subjects overlap: {shared_subjects.tolist()}"
        )
    validate_activity_mapping(normal_ids, anomaly_ids, test_labels)

    fit_subjects, validation_subjects = split_training_subjects(
        train_subjects, args.validation_fraction, args.seed
    )
    fit_subject_mask = np.isin(train_subjects, fit_subjects)
    validation_subject_mask = np.isin(train_subjects, validation_subjects)
    fit_normal_mask = fit_subject_mask & np.isin(train_labels, normal_ids)
    validation_normal_mask = validation_subject_mask & np.isin(train_labels, normal_ids)
    if not fit_normal_mask.any():
        raise ValueError("the fit subjects contain no windows for the chosen normal activities")
    if not validation_normal_mask.any():
        raise ValueError(
            "the validation subjects contain no windows for the chosen normal activities; "
            "choose other normal IDs or a different seed/fraction"
        )
    if np.intersect1d(
        np.unique(train_subjects[fit_subject_mask]),
        np.unique(train_subjects[validation_subject_mask]),
    ).size:
        raise RuntimeError("training and validation subjects overlap")

    fit_signal = train_signal[fit_normal_mask]
    validation_signal = train_signal[validation_normal_mask]
    mean, scale = fit_normalization(fit_signal)
    fit_values = normalize(fit_signal, mean, scale)
    validation_values = normalize(validation_signal, mean, scale)
    test_values = normalize(test_signal, mean, scale)

    fit_loader = make_loader(
        fit_values, args.batch_size, shuffle=True, seed=args.seed
    )
    validation_loader = make_loader(
        validation_values, args.batch_size, shuffle=False, seed=args.seed
    )
    model = TemporalAttentionAutoencoder(hidden_size=args.hidden_size).to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=args.learning_rate)

    best_validation_mse = float("inf")
    best_epoch = 0
    best_state: Optional[Dict[str, torch.Tensor]] = None
    stale_epochs = 0
    history: List[Tuple[int, float, float]] = []
    for epoch in range(1, args.epochs + 1):
        training_mse = run_epoch(model, fit_loader, device, optimizer)
        validation_mse = run_epoch(model, validation_loader, device)
        history.append((epoch, training_mse, validation_mse))
        print(
            f"Epoch {epoch:03d} | train MSE {training_mse:.7f} | "
            f"validation MSE {validation_mse:.7f}"
        )
        if validation_mse < best_validation_mse:
            best_validation_mse = validation_mse
            best_epoch = epoch
            best_state = {
                name: tensor.detach().cpu().clone()
                for name, tensor in model.state_dict().items()
            }
            stale_epochs = 0
        else:
            stale_epochs += 1
            if stale_epochs >= args.patience:
                break

    if best_state is None:
        raise RuntimeError("training did not produce a validation checkpoint")
    model.load_state_dict(best_state)
    validation_errors, _ = score_windows(
        model, validation_values, args.batch_size, device
    )
    threshold = float(np.quantile(validation_errors, args.threshold_quantile))
    test_errors, test_attention = score_windows(
        model, test_values, args.batch_size, device
    )
    predictions = (test_errors > threshold).astype(np.int64)
    y_true = np.isin(test_labels, anomaly_ids).astype(np.int64)
    metrics = binary_metrics(y_true, predictions)

    output_dir = Path(args.output_dir).expanduser()
    output_dir.mkdir(parents=True, exist_ok=True)
    checkpoint = {
        "model_state_dict": best_state,
        "architecture": {
            "name": "TemporalAttentionAutoencoder",
            "feature_count": FEATURE_COUNT,
            "hidden_size": args.hidden_size,
            "sequence_length": SEQUENCE_LENGTH,
        },
        "normalization": {"mean": mean.tolist(), "scale": scale.tolist()},
        "threshold": threshold,
        "threshold_quantile": args.threshold_quantile,
        "normal_activities": list(normal_ids),
        "anomaly_activities": list(anomaly_ids),
        "fit_subjects": [int(value) for value in fit_subjects],
        "validation_subjects": [int(value) for value in validation_subjects],
        "best_epoch": best_epoch,
        "best_validation_mse": best_validation_mse,
    }
    torch.save(checkpoint, output_dir / "reference_model.pt")
    write_training_history(output_dir / "training_history.csv", history)

    with (output_dir / "test_window_errors.csv").open(
        "w", newline="", encoding="utf-8"
    ) as handle:
        writer = csv.writer(handle)
        writer.writerow(
            (
                "window_index",
                "subject_id",
                "activity_id",
                "true_anomaly",
                "reconstruction_mse",
                "predicted_anomaly",
            )
        )
        for index, (subject, label, truth, error, prediction) in enumerate(
            zip(test_subjects, test_labels, y_true, test_errors, predictions)
        ):
            writer.writerow(
                (
                    index,
                    int(subject),
                    int(label),
                    int(truth),
                    f"{float(error):.10g}",
                    int(prediction),
                )
            )
    np.savez_compressed(
        output_dir / "test_attention.npz",
        attention=test_attention,
        subject_id=test_subjects,
        activity_id=test_labels,
        reconstruction_mse=test_errors,
    )
    report: Dict[str, object] = {
        "dataset": "UCI HAR",
        "checkpoint": "reference_model.pt",
        "best_epoch": best_epoch,
        "best_validation_mse": best_validation_mse,
        "validation_subjects": [int(value) for value in validation_subjects],
        "threshold_quantile": args.threshold_quantile,
        "threshold": threshold,
        "normal_activities": list(normal_ids),
        "anomaly_activities": list(anomaly_ids),
        "test_metrics": metrics,
        "outputs": [
            "reference_model.pt",
            "training_history.csv",
            "test_window_errors.csv",
            "test_attention.npz",
        ],
    }
    with (output_dir / "metrics.json").open("w", encoding="utf-8") as handle:
        json.dump(report, handle, indent=2)
        handle.write("\n")
    print(f"Saved checkpoint and measured test outputs under {output_dir}")
    print(
        f"Test F1 {metrics['f1']:.4f} | precision {metrics['precision']:.4f} | "
        f"recall {metrics['recall']:.4f} | accuracy {metrics['accuracy']:.4f}"
    )
    return report


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="UCI HAR reference trainer."
    )
    parser.add_argument(
        "--dataset-root",
        required=True,
        help="local path to the extracted UCI HAR Dataset directory",
    )
    parser.add_argument(
        "--normal-activities",
        required=True,
        type=parse_activity_ids,
        help="comma-separated activity IDs used as normal, for example 1",
    )
    parser.add_argument(
        "--anomaly-activities",
        required=True,
        type=parse_activity_ids,
        help="comma-separated activity IDs mapped to anomalies during evaluation",
    )
    parser.add_argument("--output-dir", default="uci_har_reference_output")
    parser.add_argument("--validation-fraction", type=float, default=0.2)
    parser.add_argument("--threshold-quantile", type=float, default=0.95)
    parser.add_argument("--hidden-size", type=int, default=32)
    parser.add_argument("--epochs", type=int, default=100)
    parser.add_argument("--patience", type=int, default=15)
    parser.add_argument("--batch-size", type=int, default=128)
    parser.add_argument("--learning-rate", type=float, default=1e-3)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--device", choices=("auto", "cpu", "cuda", "mps"), default="auto")
    return parser


def main() -> None:
    parser = build_parser()
    args = parser.parse_args()
    if args.epochs < 1 or args.patience < 1 or args.batch_size < 1:
        parser.error("epochs, patience, and batch-size must be positive")
    if args.hidden_size < 1:
        parser.error("hidden-size must be positive")
    if not 0.0 < args.validation_fraction < 1.0:
        parser.error("validation-fraction must be between 0 and 1")
    if not 0.0 <= args.threshold_quantile <= 1.0:
        parser.error("threshold-quantile must be between 0 and 1")
    try:
        train_and_evaluate(args)
    except (FileNotFoundError, ValueError, RuntimeError) as exc:
        parser.error(str(exc))


if __name__ == "__main__":
    main()
