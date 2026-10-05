"""
train_example.py

 – Exemple d'entraînement complet
Topic M6 : Synthetic Thermal Time-Series
Démontre comment utiliser ThermalTSDBDataModule avec un modèle d'attention
(Transformer simplifié) pour la détection d'anomalies thermiques.

Usage
-----
  python train_example.py --epochs 10 --batch-size 64 --num-workers 4

Le script :
  1. Initialise le DataModule (connexion TimescaleDB)
  2. Crée les DataLoaders train / val / test
  3. Définit un AnomalyTransformer simple
  4. Lance la boucle d'entraînement avec mesure de latence DB
  5. Exporte un rapport JSON (metrics + timing)
"""

from __future__ import annotations

import argparse
import json
import logging
import os
import time
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
from torch.optim import AdamW
from torch.optim.lr_scheduler import CosineAnnealingLR
from dotenv import load_dotenv

from thermal_tsdb_dataset import ThermalTSDBDataModule, _build_dsn_from_env

load_dotenv()

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    datefmt="%H:%M:%S",
)
log = logging.getLogger("train")

# ---------------------------------------------------------------------------
# Modèle : AnomalyTransformer (attention-based, léger)
# ---------------------------------------------------------------------------

class AnomalyTransformer(nn.Module):
    """
    Transformateur léger pour classification d'anomalies sur séries temporelles.

    Architecture
    ------------
    Input  : (B, C, T)   →  C canaux, T pas de temps
    Embed  : projection linéaire (C, T) → (T, d_model)
    Encode : N couches TransformerEncoder (self-attention multi-tête)
    Pool   : mean pooling sur la dimension temporelle
    Head   : MLP → 2 classes (normal / anomalie)

    Les cartes d'attention (attn_weights) sont stockées pour l'analyse
    d'explicabilité ( T8).
    """

    def __init__(
        self,
        n_channels: int   = 3,
        window_size: int  = 60,
        d_model: int      = 64,
        nhead: int        = 4,
        num_layers: int   = 2,
        dropout: float    = 0.1,
        n_classes: int    = 2,
    ):
        super().__init__()
        self.n_channels  = n_channels
        self.window_size = window_size
        self.d_model     = d_model

        # Projection des canaux vers d_model
        self.input_proj = nn.Linear(n_channels, d_model)

        # Encodage positionnel
        self.pos_enc = nn.Embedding(window_size, d_model)

        # Transformer Encoder
        enc_layer = nn.TransformerEncoderLayer(
            d_model         = d_model,
            nhead           = nhead,
            dim_feedforward = d_model * 4,
            dropout         = dropout,
            batch_first     = True,   # (B, T, d_model)
        )
        self.encoder = nn.TransformerEncoder(enc_layer, num_layers=num_layers)

        # Tête de classification
        self.classifier = nn.Sequential(
            nn.Linear(d_model, d_model // 2),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(d_model // 2, n_classes),
        )

        # Poids d'attention pour l'explicabilité (remplis lors du forward)
        self.last_attention_weights: torch.Tensor | None = None

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        x : (B, C, T)
        returns : (B, n_classes)  logits
        """
        B, C, T = x.shape

        # (B, C, T) → (B, T, C) → (B, T, d_model)
        x = x.permute(0, 2, 1)           # (B, T, C)
        x = self.input_proj(x)           # (B, T, d_model)

        # Encodage positionnel
        positions = torch.arange(T, device=x.device)
        x = x + self.pos_enc(positions)  # (B, T, d_model)

        # Encodeur Transformer
        x = self.encoder(x)              # (B, T, d_model)

        # Pooling temporel (mean)
        x = x.mean(dim=1)               # (B, d_model)

        return self.classifier(x)        # (B, n_classes)

    def get_attention_maps(self, x: torch.Tensor) -> torch.Tensor:
        """
        Calcule les cartes d'attention pour une fenêtre.
        Utilisé en  T8 (Explainability).

        Retourne : (n_layers, n_heads, T, T)
        """
        self.eval()
        attention_maps = []

        B, C, T = x.shape
        x = x.permute(0, 2, 1)
        x = self.input_proj(x)
        positions = torch.arange(T, device=x.device)
        x = x + self.pos_enc(positions)

        for layer in self.encoder.layers:
            # Accès aux poids d'attention de la couche
            with torch.no_grad():
                attn_output, attn_weights = layer.self_attn(
                    x, x, x, need_weights=True, average_attn_weights=False
                )
            attention_maps.append(attn_weights.detach())  # (B, n_heads, T, T)
            x = layer(x)

        # Stack : (n_layers, B, n_heads, T, T) → squeeze B si B=1
        return torch.stack(attention_maps, dim=0)


# ---------------------------------------------------------------------------
# Métriques
# ---------------------------------------------------------------------------

def compute_metrics(logits: torch.Tensor, labels: torch.Tensor) -> dict:
    """Calcule accuracy, precision, recall, F1 pour la classe anomalie."""
    preds = logits.argmax(dim=1)
    tp = ((preds == 1) & (labels == 1)).sum().float()
    fp = ((preds == 1) & (labels == 0)).sum().float()
    fn = ((preds == 0) & (labels == 1)).sum().float()
    tn = ((preds == 0) & (labels == 0)).sum().float()

    acc       = (tp + tn) / (tp + tn + fp + fn + 1e-8)
    precision = tp / (tp + fp + 1e-8)
    recall    = tp / (tp + fn + 1e-8)
    f1        = 2 * precision * recall / (precision + recall + 1e-8)

    return {
        "accuracy":  acc.item(),
        "precision": precision.item(),
        "recall":    recall.item(),
        "f1":        f1.item(),
        "tp": int(tp), "fp": int(fp), "fn": int(fn), "tn": int(tn),
    }


# ---------------------------------------------------------------------------
# Boucle d'entraînement
# ---------------------------------------------------------------------------

def train_one_epoch(
    model:     nn.Module,
    loader:    torch.utils.data.DataLoader,
    optimizer: torch.optim.Optimizer,
    criterion: nn.Module,
    device:    torch.device,
) -> dict:
    """Entraîne le modèle sur une époque. Retourne les métriques + timing."""
    model.train()
    total_loss = 0.0
    all_logits = []
    all_labels = []
    data_fetch_time = 0.0
    compute_time    = 0.0

    t_start = time.perf_counter()
    t_fetch = time.perf_counter()

    for signals, labels in loader:
        # Mesure du temps de lecture DB
        data_fetch_time += time.perf_counter() - t_fetch

        # Transfert GPU
        t_compute = time.perf_counter()
        signals = signals.to(device)    # (B, C, T)
        labels  = labels.to(device)

        optimizer.zero_grad()
        logits = model(signals)         # (B, 2)
        loss   = criterion(logits, labels)
        loss.backward()
        optimizer.step()

        total_loss += loss.item()
        all_logits.append(logits.detach().cpu())
        all_labels.append(labels.cpu())

        compute_time += time.perf_counter() - t_compute
        t_fetch = time.perf_counter()   # Reset chrono fetch

    epoch_time = time.perf_counter() - t_start
    all_logits = torch.cat(all_logits)
    all_labels = torch.cat(all_labels)
    metrics    = compute_metrics(all_logits, all_labels)
    metrics.update({
        "loss":              total_loss / len(loader),
        "epoch_s":           round(epoch_time, 3),
        "data_fetch_s":      round(data_fetch_time, 3),
        "compute_s":         round(compute_time, 3),
        "db_overhead_pct":   round(100 * data_fetch_time / epoch_time, 1),
        "samples_per_s":     round(len(all_labels) / epoch_time, 0),
    })
    return metrics


@torch.no_grad()
def evaluate(
    model:    nn.Module,
    loader:   torch.utils.data.DataLoader,
    criterion: nn.Module,
    device:   torch.device,
) -> dict:
    """Évalue le modèle sur val ou test."""
    model.eval()
    total_loss = 0.0
    all_logits = []
    all_labels = []

    for signals, labels in loader:
        signals = signals.to(device)
        labels  = labels.to(device)
        logits  = model(signals)
        loss    = criterion(logits, labels)
        total_loss += loss.item()
        all_logits.append(logits.cpu())
        all_labels.append(labels.cpu())

    all_logits = torch.cat(all_logits)
    all_labels = torch.cat(all_labels)
    metrics    = compute_metrics(all_logits, all_labels)
    metrics["loss"] = total_loss / len(loader)
    return metrics


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def parse_args():
    p = argparse.ArgumentParser(description=" – Training Loop TimescaleDB")
    p.add_argument("--dsn",         default=None)
    p.add_argument("--epochs",      type=int,   default=10)
    p.add_argument("--batch-size",  type=int,   default=32)
    p.add_argument("--num-workers", type=int,   default=0)
    p.add_argument("--lr",          type=float, default=1e-3)
    p.add_argument("--d-model",     type=int,   default=64)
    p.add_argument("--n-layers",    type=int,   default=2)
    p.add_argument("--out-dir",     default="results")
    return p.parse_args()


def main():
    args   = parse_args()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    log.info(f"Device : {device}")

    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    # ── 1. DataModule ────────────────────────────────────────────────
    dsn = args.dsn or _build_dsn_from_env()
    dm  = ThermalTSDBDataModule(
        dsn         = dsn,
        batch_size  = args.batch_size,
        num_workers = args.num_workers,
    )
    train_loader, val_loader, test_loader = dm.get_loaders()

    # Inférer les dimensions depuis un batch
    sample_signals, _ = next(iter(train_loader))
    _, n_channels, window_size = sample_signals.shape
    log.info(f"Shape signal : (B={args.batch_size}, C={n_channels}, T={window_size})")

    # ── 2. Modèle ────────────────────────────────────────────────────
    model = AnomalyTransformer(
        n_channels  = n_channels,
        window_size = window_size,
        d_model     = args.d_model,
        num_layers  = args.n_layers,
    ).to(device)

    n_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    log.info(f"Paramètres entraînables : {n_params:,}")

    # Classe pondérée pour corriger le déséquilibre (85% normal, 15% anomalie)
    class_weights = torch.tensor([0.15, 0.85], dtype=torch.float32).to(device)
    criterion = nn.CrossEntropyLoss(weight=class_weights)
    optimizer = AdamW(model.parameters(), lr=args.lr, weight_decay=1e-4)
    scheduler = CosineAnnealingLR(optimizer, T_max=args.epochs)

    # ── 3. Boucle d'entraînement ─────────────────────────────────────
    history = []
    best_val_f1 = 0.0
    best_model_path = out_dir / "best_model.pt"

    log.info("=" * 60)
    log.info("DÉBUT DE L'ENTRAÎNEMENT")
    log.info("=" * 60)

    for epoch in range(1, args.epochs + 1):
        train_metrics = train_one_epoch(
            model, train_loader, optimizer, criterion, device
        )
        val_metrics = evaluate(model, val_loader, criterion, device)
        scheduler.step()

        record = {
            "epoch": epoch,
            "train": train_metrics,
            "val":   val_metrics,
        }
        history.append(record)

        log.info(
            f"Epoch {epoch:02d}/{args.epochs} | "
            f"Train loss={train_metrics['loss']:.4f} F1={train_metrics['f1']:.3f} "
            f"[{train_metrics['epoch_s']:.1f}s, "
            f"DB={train_metrics['db_overhead_pct']}%] | "
            f"Val F1={val_metrics['f1']:.3f}"
        )

        if val_metrics["f1"] > best_val_f1:
            best_val_f1 = val_metrics["f1"]
            torch.save(model.state_dict(), best_model_path)
            log.info(f"  ✓ Meilleur modèle sauvegardé (val F1={best_val_f1:.3f})")

    # ── 4. Évaluation finale sur le test set ─────────────────────────
    model.load_state_dict(torch.load(best_model_path, map_location=device))
    test_metrics = evaluate(model, test_loader, criterion, device)
    log.info("=" * 60)
    log.info("ÉVALUATION FINALE (Test Set)")
    log.info(f"  Loss      : {test_metrics['loss']:.4f}")
    log.info(f"  Accuracy  : {test_metrics['accuracy']:.3f}")
    log.info(f"  Precision : {test_metrics['precision']:.3f}")
    log.info(f"  Recall    : {test_metrics['recall']:.3f}")
    log.info(f"  F1        : {test_metrics['f1']:.3f}")
    log.info(f"  TP={test_metrics['tp']} FP={test_metrics['fp']} "
             f"FN={test_metrics['fn']} TN={test_metrics['tn']}")
    log.info("=" * 60)

    # ── 5. Export du rapport ─────────────────────────────────────────
    report = {
        "config": {
            "epochs":      args.epochs,
            "batch_size":  args.batch_size,
            "num_workers": args.num_workers,
            "lr":          args.lr,
            "d_model":     args.d_model,
            "n_layers":    args.n_layers,
            "n_params":    n_params,
            "device":      str(device),
        },
        "test_metrics": test_metrics,
        "history":      history,
    }
    report_path = out_dir / "training_report.json"
    with open(report_path, "w") as f:
        json.dump(report, f, indent=2)
    log.info(f"Rapport sauvegardé → {report_path}")

    # Résumé timing DB overhead
    avg_db_pct = np.mean([e["train"]["db_overhead_pct"] for e in history])
    avg_samples_s = np.mean([e["train"]["samples_per_s"] for e in history])
    log.info(f"\nRésumé timing DataLoader DB :")
    log.info(f"  Overhead DB moyen par époque : {avg_db_pct:.1f}%")
    log.info(f"  Débit moyen                  : {avg_samples_s:.0f} samples/s")


if __name__ == "__main__":
    main()