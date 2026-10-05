# =============================================================================
# Training script (train.py)
# : LR scheduler + checkpoints + assemblage train.py final
# =============================================================================
# Usage :
#   python train.py
#   python train.py --epochs 1000 --patience 50 --lr 1e-3 --device cuda
# =============================================================================

from __future__ import annotations

import argparse
import csv
import time
from pathlib import Path
from typing import Tuple

import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset, random_split

# Assumes model.py and loss.py are in the same directory (Phase 1 final delivery)
try:
    from model import TAAE
except ImportError:
    print("Warning: model.py is unavailable; check the model module.")
    
try:
    from loss import build_cploss, CPLoss
except ImportError:
    print("Warning: loss.py is unavailable; check the loss module.")


# =============================================================================
# Training & Evaluation (from )
# =============================================================================

def train_one_epoch(
    model:     nn.Module,
    loader:    DataLoader,
    criterion: nn.Module,
    optimizer: torch.optim.Optimizer,
    device:    torch.device,
) -> float:
    """One full pass over the training set."""
    model.train()
    total_loss = 0.0
    n_batches  = 0

    for batch in loader:
        x = batch[0] if isinstance(batch, (list, tuple)) else batch
        x = x.to(device)

        # ── Forward ──────────────────────────────────────────────────────────
        x_hat, alpha, z = model(x)

        # ── CPLoss (scalar, reduction='mean') ────────────────────────────────
        loss = criterion(x, x_hat)

        # ── Backward ─────────────────────────────────────────────────────────
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        total_loss += loss.item()
        n_batches  += 1

    return total_loss / max(n_batches, 1)


@torch.no_grad()
def evaluate(
    model:     nn.Module,
    loader:    DataLoader,
    criterion: nn.Module,
    device:    torch.device,
) -> float:
    """Validation pass (no gradient)."""
    model.eval()
    total_loss = 0.0
    n_batches  = 0

    for batch in loader:
        x = batch[0] if isinstance(batch, (list, tuple)) else batch
        x = x.to(device)

        x_hat, alpha, z = model(x)
        loss = criterion(x, x_hat)

        total_loss += loss.item()
        n_batches  += 1

    return total_loss / max(n_batches, 1)


# =============================================================================
# Checkpoints & Early Stopping ()
# =============================================================================

class EarlyStopping:
    """
    Stops training when the validation loss has not improved for `patience`
    consecutive epochs. Saves the best model weights automatically (checkpoints).
    """

    def __init__(
        self,
        patience:  int   = 50,
        min_delta: float = 0.0,
        save_path: str   = "best_model.pt",
        verbose:   bool  = True,
    ):
        self.patience   = patience
        self.min_delta  = min_delta
        self.save_path  = save_path
        self.verbose    = verbose

        self.best_loss   : float = float("inf")
        self.counter     : int   = 0
        self.best_epoch  : int   = 0
        self.should_stop : bool  = False

    def step(self, val_loss: float, model: nn.Module, epoch: int) -> bool:
        improved = val_loss < self.best_loss - self.min_delta

        if improved:
            self.best_loss  = val_loss
            self.best_epoch = epoch
            self.counter    = 0
            # Sauvegarde modèle
            torch.save(model.state_dict(), self.save_path)
            if self.verbose:
                print(f"  ✔ Val loss improved → {val_loss:.6f} (saved → {self.save_path})")
        else:
            self.counter += 1
            if self.verbose:
                print(f"  ✗ No improvement for {self.counter}/{self.patience} epoch(s) (best: {self.best_loss:.6f} @ ep {self.best_epoch})")

        if self.counter >= self.patience:
            self.should_stop = True
            if self.verbose:
                print(f"\nEarly stopping at epoch {epoch}")

        return self.should_stop


# =============================================================================
# Main training function (Algorithm 3 Final Assembly)
# =============================================================================

def train(
    train_loader: DataLoader,
    val_loader:   DataLoader,
    model:        nn.Module,
    device:       torch.device,
    lr:           float = 0.001,
    weight_decay: float = 0.0001,
    max_epochs:   int   = 1000,
    patience:     int   = 50,
    save_path:    str   = "best_model.pt",
    log_path:     str   = "train_log.csv",
) -> dict:
    
    print("\n" + "=" * 60)
    print("Algorithm 3 — TAAE Training Loop (Final Assembly)")
    print("=" * 60)
    
    print("[Step 1] Building CPLoss (precomputing normalization stats)…")
    criterion = build_cploss(train_loader, model, device=str(device))
    criterion = criterion.to(device)

    print("[Step 2] Initialising optimiser and LR Scheduler…")
    optimizer = torch.optim.AdamW(
        model.parameters(), lr=lr, weight_decay=weight_decay
    )
    
    # LR Scheduler (): reduce LR by 0.5 if no improvement for 5 epochs
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, mode="min", factor=0.5, patience=5, min_lr=1e-6, verbose=True
    )

    # Checkpoints & Early Stopping ()
    early_stop = EarlyStopping(
        patience=patience, save_path=save_path, verbose=True
    )

    log_file = open(log_path, "w", newline="")
    writer   = csv.writer(log_file)
    writer.writerow(["epoch", "train_loss", "val_loss", "lr", "elapsed_s"])

    history: dict = {
        "train_loss": [],
        "val_loss":   [],
        "epochs_run": 0,
        "best_epoch": 0,
        "best_val_loss": float("inf"),
        "stopped_early": False,
    }

    t0 = time.time()

    print(f"[Step 3] Training for up to {max_epochs} epochs (patience={patience})…\n")

    for epoch in range(1, max_epochs + 1):
        ep_start = time.time()

        train_loss = train_one_epoch(model, train_loader, criterion, optimizer, device)
        val_loss   = evaluate(model, val_loader, criterion, device)

        elapsed = time.time() - ep_start
        current_lr = optimizer.param_groups[0]["lr"]

        # Log every 10 epochs (or every epoch if preferred, here we print every epoch but format nicely)
        print(f"Epoch {epoch:>4} | Train Loss: {train_loss:.6f} | Val Loss: {val_loss:.6f} | LR: {current_lr:.2e} | Time: {elapsed:.1f}s")

        # LR scheduler step
        scheduler.step(val_loss)

        history["train_loss"].append(train_loss)
        history["val_loss"].append(val_loss)
        history["epochs_run"] = epoch

        writer.writerow([epoch, f"{train_loss:.8f}", f"{val_loss:.8f}", f"{current_lr:.2e}", f"{elapsed:.2f}"])
        log_file.flush()

        # Checkpoint and Early Stopping step
        stop = early_stop.step(val_loss, model, epoch)
        if stop:
            history["stopped_early"] = True
            break

    log_file.close()

    # Load best weights
    if Path(save_path).exists():
        model.load_state_dict(torch.load(save_path, map_location=device))
        print(f"\n[Step 4] Best weights reloaded from '{save_path}'.")

    history["best_epoch"]    = early_stop.best_epoch
    history["best_val_loss"] = early_stop.best_loss

    total_time = time.time() - t0
    print(f"\n{'='*60}")
    print(f"Training complete in {total_time/60:.1f} min")
    print(f"  Best val loss : {history['best_val_loss']:.6f} @ epoch {history['best_epoch']}")
    print(f"  Log saved     : {log_path}")
    print(f"  Checkpoint    : {save_path}")
    print("=" * 60)

    return history


# =============================================================================
# CLI
# =============================================================================

def _make_synthetic_loaders(
    n_train: int = 8_000,
    n_val:   int = 2_000,
    seq_len: int = 128,
    features: int = 9,
    batch_size: int = 80,
) -> Tuple[DataLoader, DataLoader]:
    total = n_train + n_val
    X = torch.rand(total, seq_len, features)
    ds = TensorDataset(X)
    tr_ds, va_ds = random_split(
        ds, [n_train, n_val],
        generator=torch.Generator().manual_seed(42)
    )
    train_loader = DataLoader(tr_ds, batch_size=batch_size, shuffle=True)
    val_loader   = DataLoader(va_ds, batch_size=batch_size, shuffle=False)
    return train_loader, val_loader

def main():
    parser = argparse.ArgumentParser(description="TAAE Training Script — Algorithm 3 ( Final Assembly)")
    parser.add_argument("--epochs",       type=int,   default=1000,   help="Maximum number of training epochs (default 1000).")
    parser.add_argument("--patience",     type=int,   default=50,     help="Early-stopping patience (default 50).")
    parser.add_argument("--lr",           type=float, default=0.001,  help="Initial learning rate (default 0.001).")
    parser.add_argument("--weight_decay", type=float, default=0.0001, help="AdamW weight decay (default 0.0001).")
    parser.add_argument("--batch_size",   type=int,   default=80,     help="Batch size (default 80).")
    parser.add_argument("--device",       type=str,   default="auto", help="'cpu' | 'cuda' | 'mps' | 'auto'.")
    parser.add_argument("--save_path",    type=str,   default="best_model.pt", help="Checkpoint file path.")
    parser.add_argument("--log_path",     type=str,   default="train_log.csv", help="CSV log file path.")
    parser.add_argument("--dataset_root", type=str,   default="UCI_HAR", help="Path to UCI_HAR folder (optional).")
    args = parser.parse_args()

    if args.device == "auto":
        if torch.cuda.is_available(): device = torch.device("cuda")
        elif torch.backends.mps.is_available(): device = torch.device("mps")
        else: device = torch.device("cpu")
    else:
        device = torch.device(args.device)

    print(f"Device : {device}")

    # Initialize model
    try:
        model = TAAE(
            input_features=9,
            hidden1=64,
            hidden2=32,
            hidden3=32,
            hidden4=64,
            d_latent=16,
            d_attn=32,
            dropout=0.2,
            seq_len=128,
        ).to(device)
    except NameError:
        print("Model TAAE is not imported. Using a dummy model for testing...")
        model = nn.Sequential(nn.Linear(1,1)).to(device) # dummy

    # Data loaders
    try:
        import sys
        # Allow importing from  if har_dataloader is there
        sys.path.append(str(Path(__file__).parent / ""))
        from har_dataloader import make_har_loaders
        train_loader, val_loader = make_har_loaders(
            dataset_root = args.dataset_root,
            batch_size   = args.batch_size,
            pin_memory   = (device.type == "cuda"),
        )
    except (ImportError, FileNotFoundError):
        print("Could not load real data (har_dataloader or UCI_HAR missing). Using synthetic loaders.")
        train_loader, val_loader = _make_synthetic_loaders(batch_size=args.batch_size)

    # Train
    train(
        train_loader = train_loader,
        val_loader   = val_loader,
        model        = model,
        device       = device,
        lr           = args.lr,
        weight_decay = args.weight_decay,
        max_epochs   = args.epochs,
        patience     = args.patience,
        save_path    = args.save_path,
        log_path     = args.log_path,
    )

if __name__ == "__main__":
    main()
