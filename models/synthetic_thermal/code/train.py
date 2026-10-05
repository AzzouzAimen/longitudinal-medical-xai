"""
train.py
 — Training Script
Topic M6 : Synthetic Thermal Time-Series
Implements Algorithm 3 (algorithm3_training.txt) exactly:
  • AdamW optimiser  (lr=0.001, weight_decay=0.0001)
  • ReduceLROnPlateau scheduler  (patience=5, factor=0.5)
  • Early stopping  (patience=50)
  • Xavier weight initialisation  (done inside model.py)
  • Saves best checkpoint to results/best_model.pt

Report outputs (auto-generated after training):
  • Figure 1 left  : Training + Validation loss curves (log scale)
  • Figure 1 right : Learning-rate schedule
  • Table IV CSV   : Ablation study results

Usage
-----
    # Single run with CPLoss (default)
    python train.py

    # Full ablation study (all 4 loss variants → Table IV)
    python train.py --ablation

    # Custom settings
    python train.py --epochs 500 --batch-size 64 --loss "MSE only"
"""

from __future__ import annotations

import argparse
import csv
import json
import logging
import os
import time
from pathlib import Path
from typing import Optional

import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from dotenv import load_dotenv

from model import TAAE
from loss  import CPLoss, get_loss, LOSS_VARIANTS

load_dotenv()

# ---------------------------------------------------------------------------
# Logging
# ---------------------------------------------------------------------------
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    datefmt="%H:%M:%S",
)
log = logging.getLogger("train")

# ---------------------------------------------------------------------------
# Output directory
# ---------------------------------------------------------------------------
OUT_DIR = Path("results")
OUT_DIR.mkdir(exist_ok=True)


# ===========================================================================
# Core training function  (Algorithm 3)
# ===========================================================================

def train_model(
    train_loader:  DataLoader,
    val_loader:    DataLoader,
    n_channels:    int   = 3,
    window_size:   int   = 60,
    max_epochs:    int   = 5,
    learning_rate: float = 0.001,
    weight_decay:  float = 0.0001,
    patience:      int   = 50,
    lr_patience:   int   = 5,
    lr_factor:     float = 0.5,
    loss_name:     str   = "CPLoss Full",
    checkpoint_path: Path = OUT_DIR / "best_model.pt",
    device:        Optional[torch.device] = None,
) -> dict:
    """
    Train TAAE with the chosen loss variant.

    Returns a history dict with keys:
        train_losses, val_losses, lr_history, best_epoch, stopped_epoch
    """
    if device is None:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    log.info(f"Device : {device}")

    # ---- Model & optimiser ------------------------------------------------
    model     = TAAE(n_channels=n_channels, window_size=window_size).to(device)
    criterion = get_loss(loss_name).to(device)
    optimizer = torch.optim.AdamW(
        model.parameters(), lr=learning_rate, weight_decay=weight_decay
    )
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, mode="min", factor=lr_factor, patience=lr_patience
    )

    log.info(
        f"Training | loss={loss_name} | epochs={max_epochs} "
        f"| patience={patience} | lr={learning_rate}"
    )

    # ---- History buffers ---------------------------------------------------
    train_losses: list[float] = []
    val_losses:   list[float] = []
    lr_history:   list[float] = []

    best_val_loss    = float("inf")
    patience_counter = 0
    best_epoch       = 0
    stopped_epoch    = max_epochs

    # ---- Training loop  (Algorithm 3) -------------------------------------
    for epoch in range(1, max_epochs + 1):

        # ==================== TRAINING PHASE ====================
        model.train()
        total_train = 0.0
        n_batches   = 0

        for batch in train_loader:
            # Unpack — loader may return (signal, label) or (signal, label, mask)
            if len(batch) == 3:
                signals, labels, masks = batch
                masks = masks.to(device)
            else:
                signals, labels = batch
                masks = None

            # Only train on healthy windows (label == 0) as per Algorithm 3
            healthy = (labels == 0)
            if healthy.sum() == 0:
                continue

            signals = signals[healthy].to(device)   # (B, C, T)
            if masks is not None:
                masks = masks[healthy]

            optimizer.zero_grad()
            x_hat, alpha = model(signals)
            loss = criterion(signals, x_hat, masks)
            loss.backward()
            nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            optimizer.step()

            total_train += loss.item()
            n_batches   += 1

        avg_train = total_train / max(n_batches, 1)

        # ==================== VALIDATION PHASE ====================
        model.eval()
        total_val = 0.0
        n_val_batches = 0

        with torch.no_grad():
            for batch in val_loader:
                if len(batch) == 3:
                    signals, labels, masks = batch
                    masks = masks.to(device)
                else:
                    signals, labels = batch
                    masks = None

                healthy = (labels == 0)
                if healthy.sum() == 0:
                    continue

                signals = signals[healthy].to(device)
                if masks is not None:
                    masks = masks[healthy]

                x_hat, _ = model(signals)
                loss = criterion(signals, x_hat, masks)
                total_val     += loss.item()
                n_val_batches += 1

        avg_val = total_val / max(n_val_batches, 1)

        # ==================== LR SCHEDULER ====================
        scheduler.step(avg_val)
        current_lr = optimizer.param_groups[0]["lr"]

        # ==================== RECORD HISTORY ====================
        train_losses.append(avg_train)
        val_losses.append(avg_val)
        lr_history.append(current_lr)

        # ==================== EARLY STOPPING ====================
        if avg_val < best_val_loss:
            best_val_loss    = avg_val
            patience_counter = 0
            best_epoch       = epoch
            torch.save(model.state_dict(), checkpoint_path)
        else:
            patience_counter += 1

        if epoch % 10 == 0:
            log.info(
                f"Epoch {epoch:4d} | "
                f"Train={avg_train:.6f} | Val={avg_val:.6f} | "
                f"LR={current_lr:.2e} | patience={patience_counter}/{patience}"
            )

        if patience_counter >= patience:
            log.info(f"Early stopping at epoch {epoch} (best epoch {best_epoch})")
            stopped_epoch = epoch
            break

    log.info(f"Best val loss : {best_val_loss:.6f} at epoch {best_epoch}")

    return {
        "loss_name":     loss_name,
        "train_losses":  train_losses,
        "val_losses":    val_losses,
        "lr_history":    lr_history,
        "best_epoch":    best_epoch,
        "stopped_epoch": stopped_epoch,
        "best_val_loss": best_val_loss,
    }


# ===========================================================================
# Quick evaluation helpers (for Table IV)
# ===========================================================================

def evaluate_model(
    model_path: Path,
    test_loader: DataLoader,
    n_channels:  int = 3,
    window_size: int = 60,
    threshold_pct: float = 85.0,
    device: Optional[torch.device] = None,
) -> dict:
    """
    Computes Test MAE, F1, and Individual Accuracy from a saved checkpoint.
    Returns a dict with keys: test_mae, f1, individual_acc.
    """
    if device is None:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    model = TAAE(n_channels=n_channels, window_size=window_size).to(device)
    model.load_state_dict(torch.load(model_path, map_location=device))
    model.eval()

    all_losses   = []
    all_labels   = []
    all_maes     = []

    with torch.no_grad():
        for batch in test_loader:
            signals, labels = batch[0], batch[1]
            signals = signals.to(device)
            x_hat, _ = model(signals)

            # Per-window reconstruction loss (MAE)
            mae = (signals - x_hat).abs().mean(dim=(1, 2))   # (B,)
            all_maes.extend(mae.cpu().tolist())
            all_losses.extend(mae.cpu().tolist())
            all_labels.extend(labels.tolist())

    losses = np.array(all_losses)
    labels = np.array(all_labels)

    # Threshold = 85th percentile of normal losses
    normal_losses = losses[labels == 0]
    tau = np.percentile(normal_losses, threshold_pct) if len(normal_losses) > 0 else losses.mean()

    preds = (losses > tau).astype(int)

    # F1
    tp = ((preds == 1) & (labels == 1)).sum()
    fp = ((preds == 1) & (labels == 0)).sum()
    fn = ((preds == 0) & (labels == 1)).sum()
    precision = tp / (tp + fp + 1e-8)
    recall    = tp / (tp + fn + 1e-8)
    f1 = 2 * precision * recall / (precision + recall + 1e-8)

    test_mae = float(np.mean(all_maes))

    # Individual accuracy (simplified: window-level here)
    individual_acc = float(100 * (preds == labels).mean())

    return {
        "test_mae":       round(test_mae * 1000, 4),   # ×10^-3 as per table
        "f1":             round(float(f1) * 100, 2),
        "individual_acc": round(individual_acc, 2),
    }


# ===========================================================================
# Figure 1 — Training curves + LR schedule
# ===========================================================================

def plot_figure1(history: dict, save_path: Path = OUT_DIR / "figure1.png"):
    """
    Figure 1 (left)  : Training + Validation loss (log scale) with early-stop line.
    Figure 1 (right) : Learning-rate schedule.
    """
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
    except ImportError:
        log.warning("matplotlib not installed — skipping Figure 1")
        return

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5))
    fig.suptitle("Figure 1 — Training Curves", fontsize=13)

    epochs     = range(1, len(history["train_losses"]) + 1)
    best_epoch = history["best_epoch"]

    # ---- Left : loss curves ------------------------------------------------
    ax1.semilogy(epochs, history["train_losses"], color="blue",   label="Train loss")
    ax1.semilogy(epochs, history["val_losses"],   color="orange", label="Val loss")
    ax1.axvline(x=best_epoch, color="red", linestyle="--",
                label=f"Early stop (ep {best_epoch})")
    ax1.set_xlabel("Epoch")
    ax1.set_ylabel("Loss (log scale)")
    ax1.set_title(f"Loss Curves — {history['loss_name']}")
    ax1.legend()
    ax1.grid(True, which="both", alpha=0.3)

    # ---- Right : LR schedule -----------------------------------------------
    ax2.plot(epochs, history["lr_history"], color="green", linewidth=1.5)
    ax2.set_xlabel("Epoch")
    ax2.set_ylabel("Learning Rate")
    ax2.set_title("Learning-Rate Schedule (ReduceLROnPlateau)")
    ax2.grid(True, alpha=0.3)

    plt.tight_layout()
    plt.savefig(save_path, dpi=150, bbox_inches="tight")
    plt.close()
    log.info(f"Figure 1 saved → {save_path}")


# ===========================================================================
# Table IV — Ablation study
# ===========================================================================

def run_ablation(
    train_loader: DataLoader,
    val_loader:   DataLoader,
    test_loader:  DataLoader,
    n_channels:   int = 3,
    window_size:  int = 60,
    max_epochs:   int = 5,
    device: Optional[torch.device] = None,
):
    """Run all 4 loss variants and save Table IV as CSV."""
    results = []

    for loss_name in LOSS_VARIANTS:
        log.info(f"\n{'='*60}")
        log.info(f"ABLATION : {loss_name}")
        log.info(f"{'='*60}")

        ckpt = OUT_DIR / f"best_model_{loss_name.replace(' ', '_').replace('+','p')}.pt"

        history = train_model(
            train_loader   = train_loader,
            val_loader     = val_loader,
            n_channels     = n_channels,
            window_size    = window_size,
            max_epochs     = max_epochs,
            loss_name      = loss_name,
            checkpoint_path= ckpt,
            device         = device,
        )

        # Plot Figure 1 per variant
        plot_figure1(
            history,
            save_path=OUT_DIR / f"figure1_{loss_name.replace(' ', '_').replace('+','p')}.png"
        )

        # Evaluate on test set
        eval_res = evaluate_model(
            model_path  = ckpt,
            test_loader = test_loader,
            n_channels  = n_channels,
            window_size = window_size,
            device      = device,
        )

        row = {
            "Loss Function":          loss_name,
            "Val. Loss":              round(history["best_val_loss"], 6),
            "Test MAE (×10⁻³)":      eval_res["test_mae"],
            "F1 (%)":                 eval_res["f1"],
            "Individual Acc. (%)":    eval_res["individual_acc"],
            "Best Epoch":             history["best_epoch"],
        }
        results.append(row)
        log.info(f"Result : {row}")

    # Save CSV
    csv_path = OUT_DIR / "table_IV_ablation.csv"
    with open(csv_path, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=results[0].keys())
        writer.writeheader()
        writer.writerows(results)

    log.info(f"\nTable IV saved → {csv_path}")

    # Pretty print
    print("\n" + "=" * 70)
    print("TABLE IV — Ablation Study : PatternLoss Components Impact")
    print("=" * 70)
    header = f"{'Loss Function':<20} {'Val.Loss':>10} {'MAE×10⁻³':>10} {'F1(%)':>8} {'IndAcc(%)':>10}"
    print(header)
    print("-" * 70)
    for r in results:
        print(
            f"{r['Loss Function']:<20} "
            f"{r['Val. Loss']:>10.6f} "
            f"{r['Test MAE (×10⁻³)']:>10.4f} "
            f"{r['F1 (%)']:>8.2f} "
            f"{r['Individual Acc. (%)']:>10.2f}"
        )
    print("=" * 70)

    return results


# ===========================================================================
# CLI entry point
# ===========================================================================

def main():
    parser = argparse.ArgumentParser(description=" – TAAE Training")
    parser.add_argument("--epochs",     type=int,   default=100)
    parser.add_argument("--batch-size", type=int,   default=256)
    parser.add_argument("--lr",         type=float, default=0.001)
    parser.add_argument("--patience",   type=int,   default=5)
    parser.add_argument("--loss",       type=str,   default="CPLoss Full",
                        choices=list(LOSS_VARIANTS),
                        help="Loss variant to use")
    parser.add_argument("--ablation",   action="store_true",
                        help="Run all 4 ablation variants (Table IV)")
    parser.add_argument("--dsn",        type=str,   default=None,
                        help="TimescaleDB DSN (defaults to .env)")
    parser.add_argument("--num-workers", type=int,  default=0)
    args = parser.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # ---- DataLoaders -------------------------------------------------------
    from thermal_tsdb_dataset import ThermalTSDBDataModule

    dm = ThermalTSDBDataModule(
        dsn         = args.dsn,
        batch_size  = args.batch_size,
        num_workers = args.num_workers,
    )
    
    # -- dm._patient_ids = dm._patient_ids[:4]
    train_loader, val_loader, test_loader = dm.get_loaders()

    n_channels  = 3
    window_size = 60

    # ---- Run ---------------------------------------------------------------
    if args.ablation:
        run_ablation(
            train_loader = train_loader,
            val_loader   = val_loader,
            test_loader  = test_loader,
            n_channels   = n_channels,
            window_size  = window_size,
            max_epochs   = args.epochs,
            device       = device,
        )
    else:
        history = train_model(
            train_loader   = train_loader,
            val_loader     = val_loader,
            n_channels     = n_channels,
            window_size    = window_size,
            max_epochs     = args.epochs,
            learning_rate  = args.lr,
            patience       = args.patience,
            loss_name      = args.loss,
            device         = device,
        )
        plot_figure1(history)
        log.info("Training complete. Best model saved to results/best_model.pt")


if __name__ == "__main__":
    main()
