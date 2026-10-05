"""
reconstruction_metrics.py
 – Task 2 : Métriques de reconstruction + Figures 2, 3, 5
Topic M6 : Synthetic Thermal Time-Series
Ce fichier produit :
  ✅ Table II  — results/table_II_reconstruction.csv
  ✅ Figure 2  — results/figure2_anomaly_scores.png
  ✅ Figure 3  — results/figure3_reconstruction.png
  ✅ Figure 5  — results/figure5_loss_distributions.png
  ✅ CSV bonus — results/subject_anomaly_pct.csv (utile pour Task 3)

Usage
-----
  python reconstruction_metrics.py --npy-dir ../Data-Wrangling/etl_output/npy
"""

from __future__ import annotations

import argparse
import logging
import re
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np
import pandas as pd
import torch
from torch.utils.data import DataLoader

from model import TAAE
from thermal_npy_dataset import DEFAULT_CHANNELS, ThermalNPYDataset

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    datefmt="%H:%M:%S",
)
log = logging.getLogger("reconstruction_metrics")

OUT_DIR     = Path("results")
WINDOW_SIZE = 60
N_CHANNELS  = 3
SEED        = 42


# ============================================================================
# SECTION 1 — CHARGEMENT
# ============================================================================

def discover_patient_ids(npy_dir: Path) -> List[int]:
    pattern = re.compile(r"patient_(\d+)_windows\.npy$")
    ids = []
    for path in sorted(npy_dir.glob("patient_*_windows.npy")):
        m = pattern.match(path.name)
        if m:
            ids.append(int(m.group(1)))
    if not ids:
        raise RuntimeError(f"Aucun fichier patient_XX_windows.npy dans {npy_dir}")
    return ids


def split_patient_ids(
    patient_ids: List[int],
    train_ratio: float = 0.70,
    val_ratio:   float = 0.15,
    seed: int = SEED,
) -> Tuple[List[int], List[int], List[int]]:
    """Reproduit EXACTEMENT le même split que evaluate.py et train.py."""
    ids = np.array(patient_ids, dtype=int)
    rng = np.random.default_rng(seed)
    rng.shuffle(ids)
    n_train = max(1, int(len(ids) * train_ratio))
    n_val   = max(1, int(len(ids) * val_ratio))
    train_ids = ids[:n_train].tolist()
    val_ids   = ids[n_train : n_train + n_val].tolist()
    test_ids  = ids[n_train + n_val:].tolist()
    log.info(f"Split → train={train_ids} | val={val_ids} | test={test_ids}")
    return train_ids, val_ids, test_ids


def load_model(checkpoint_path: Path, device: torch.device) -> TAAE:
    model = TAAE(n_channels=N_CHANNELS, window_size=WINDOW_SIZE).to(device)
    ckpt  = torch.load(checkpoint_path, map_location=device)
    if isinstance(ckpt, dict) and "model_state_dict" in ckpt:
        state = ckpt["model_state_dict"]
    elif isinstance(ckpt, dict) and "state_dict" in ckpt:
        state = ckpt["state_dict"]
    else:
        state = ckpt
    if any(str(k).startswith("module.") for k in state.keys()):
        state = {k[len("module."):]: v for k, v in state.items()}
    model.load_state_dict(state)
    model.eval()
    log.info(f"Modele charge depuis {checkpoint_path}")
    return model


def make_loader(npy_dir: Path, patient_ids: List[int], batch_size: int) -> DataLoader:
    ds = ThermalNPYDataset(npy_dir=npy_dir, patient_ids=patient_ids)
    return DataLoader(ds, batch_size=batch_size, shuffle=False, num_workers=0)


# ============================================================================
# SECTION 2 — INFERENCE
# ============================================================================

@torch.no_grad()
def run_inference(model: TAAE, loader: DataLoader, device: torch.device) -> Dict[str, np.ndarray]:
    """
    Passe toutes les fenetres dans le modele.
    Retourne : originals, reconstructions, labels, window_losses (MAE), attention.
    """
    all_orig, all_recon, all_labels, all_losses, all_attn = [], [], [], [], []
    for signals, labels in loader:
        signals = signals.to(device)
        x_hat, alpha = model(signals)
        mae = (signals - x_hat).abs().mean(dim=(1, 2))
        all_orig.append(signals.cpu().numpy())
        all_recon.append(x_hat.cpu().numpy())
        all_labels.append(labels.numpy())
        all_losses.append(mae.cpu().numpy())
        all_attn.append(alpha.cpu().numpy())
    return {
        "originals":       np.concatenate(all_orig,   axis=0),
        "reconstructions": np.concatenate(all_recon,  axis=0),
        "labels":          np.concatenate(all_labels, axis=0),
        "window_losses":   np.concatenate(all_losses, axis=0),
        "attention":       np.concatenate(all_attn,   axis=0),
    }


# ============================================================================
# SECTION 3 — TABLE II
# ============================================================================

def compute_reconstruction_metrics(originals: np.ndarray, reconstructions: np.ndarray) -> Dict[str, float]:
    N = originals.shape[0]
    x_flat    = originals.reshape(N, -1).astype(np.float64)
    xhat_flat = reconstructions.reshape(N, -1).astype(np.float64)
    err       = x_flat - xhat_flat

    mae  = float(np.abs(err).mean())
    rmse = float(np.sqrt((err ** 2).mean()))

    n_vals = float(x_flat.size)
    sum_x  = x_flat.sum();    sum_y  = xhat_flat.sum()
    sum_x2 = (x_flat**2).sum(); sum_y2 = (xhat_flat**2).sum()
    sum_xy = (x_flat * xhat_flat).sum()

    pearson_num = sum_xy - (sum_x * sum_y / n_vals)
    den_x = sum_x2 - (sum_x**2 / n_vals)
    den_y = sum_y2 - (sum_y**2 / n_vals)
    den   = np.sqrt(max(den_x, 0.0) * max(den_y, 0.0))
    pearson = float(pearson_num / den) if den > 0 else float("nan")

    cos_den = np.sqrt(sum_x2) * np.sqrt(sum_y2)
    cosine  = float(sum_xy / cos_den) if cos_den > 0 else float("nan")

    return {
        "MAE_1e3":   round(mae   * 1e3, 6),
        "RMSE_1e3":  round(rmse  * 1e3, 6),
        "Pearson":   round(pearson, 6),
        "CosineSim": round(cosine,  6),
    }


def build_and_print_table_II(train_data, val_data, test_data,
                              existing_csv: Optional[Path] = None):
    if existing_csv and existing_csv.exists():
        log.info(f"Table II existante detectee : {existing_csv} — reutilisee.")
        df = pd.read_csv(existing_csv)
        print("\n" + "="*75)
        print("TABLE II — Reconstruction Quality (evaluate.py de Task 1)")
        print("="*75)
        print(df.to_string(index=False))
        print("="*75 + "\n")
        return df

    rows = []
    for name, data, label_filter in [
        ("Training",       train_data, None),
        ("Validation",     val_data,   None),
        ("Test-Healthy",   test_data,  0),
        ("Test-Anomalous", test_data,  1),
    ]:
        lbls = data["labels"]
        mask = np.ones(len(lbls), dtype=bool) if label_filter is None else (lbls == label_filter)
        if mask.sum() == 0:
            continue
        m = compute_reconstruction_metrics(data["originals"][mask], data["reconstructions"][mask])
        rows.append({
            "Split": name, "Windows": int(mask.sum()),
            "MAE": m["MAE_1e3"]/1e3, "RMSE": m["RMSE_1e3"]/1e3,
            "Pearson Correlation": m["Pearson"], "Cosine Similarity": m["CosineSim"],
        })
        log.info(f"  {name:<22} | MAE={m['MAE_1e3']:.4f}e-3 | RMSE={m['RMSE_1e3']:.4f}e-3 | "
                 f"r={m['Pearson']:.4f} | cos={m['CosineSim']:.4f}")

    df = pd.DataFrame(rows)
    out = OUT_DIR / "table_II_reconstruction.csv"
    df.to_csv(out, index=False)
    log.info(f"Table II sauvegardee -> {out}")
    print("\n" + "="*75)
    print("TABLE II — Reconstruction Quality")
    print("="*75)
    print(df.to_string(index=False))
    print("="*75 + "\n")
    return df


# ============================================================================
# SECTION 4 — FIGURE 2
# ============================================================================

def compute_subject_anomaly_pct(data, window_threshold, npy_dir, patient_ids):
    losses = data["window_losses"]
    labels = data["labels"]

    pid_per_window = []
    for pid in patient_ids:
        meta_path = npy_dir / f"patient_{pid:02d}_windows_meta.csv"
        if meta_path.exists():
            meta = pd.read_csv(meta_path)
            pid_per_window.extend([pid] * len(meta))

    pid_arr = np.array(pid_per_window[:len(losses)])
    rows = []
    for pid in patient_ids:
        mask = pid_arr == pid
        if mask.sum() == 0:
            continue
        preds = (losses[mask] > window_threshold).astype(int)
        rows.append({
            "patient_id":        pid,
            "anomaly_pct":       round(100.0 * float(preds.mean()), 2),
            "true_pathological": int(labels[mask].sum() > 0),
        })
    return pd.DataFrame(rows)


def plot_figure2(subject_df, subject_threshold,
                 save_path=OUT_DIR / "figure2_anomaly_scores.png"):
    try:
        import matplotlib; matplotlib.use("Agg")
        import matplotlib.pyplot as plt, matplotlib.patches as mpatches
    except ImportError:
        log.warning("matplotlib manquant — Figure 2 ignoree."); return

    healthy = subject_df[subject_df["true_pathological"] == 0]
    patho   = subject_df[subject_df["true_pathological"] == 1]
    rng = np.random.default_rng(SEED)

    fig, ax = plt.subplots(figsize=(9, 6))
    fig.suptitle("Figure 2 — Distribution des scores d'anomalie individuels",
                 fontsize=13, fontweight="bold")

    ax.scatter(rng.uniform(0.80, 1.20, len(healthy)), healthy["anomaly_pct"].values,
               color="steelblue", s=90, zorder=5, edgecolors="navy", linewidths=0.5,
               label=f"Patients sains (n={len(healthy)})")
    ax.scatter(rng.uniform(1.80, 2.20, len(patho)), patho["anomaly_pct"].values,
               color="crimson", s=90, zorder=5, edgecolors="darkred", linewidths=0.5,
               label=f"Patients pathologiques (n={len(patho)})")

    ax.axhline(y=subject_threshold, color="black", linestyle="--", linewidth=1.8,
               label=f"Seuil optimal = {subject_threshold:.1f}%")

    max_h = float(healthy["anomaly_pct"].max()) if len(healthy) > 0 else 0
    min_p = float(patho["anomaly_pct"].min())   if len(patho)   > 0 else 100
    if min_p > max_h:
        ax.axhspan(max_h, min_p, color="limegreen", alpha=0.10, label="Separation parfaite")

    ax.set_xticks([1, 2])
    ax.set_xticklabels(["Patients Sains", "Patients Pathologiques"], fontsize=12)
    ax.set_ylabel("Pourcentage de fenetres anomaliques (%)", fontsize=11)
    ax.set_xlim(0.4, 2.6); ax.set_ylim(-5, 105)
    ax.legend(fontsize=9); ax.grid(True, axis="y", alpha=0.3)
    plt.tight_layout()
    fig.savefig(save_path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    log.info(f"Figure 2 sauvegardee -> {save_path}")


# ============================================================================
# SECTION 5 — FIGURE 3
# ============================================================================

def plot_figure3(test_data, save_path=OUT_DIR / "figure3_reconstruction.png",
                 channel_idx=0, channel_name="Temperature normalisee — sein gauche"):
    try:
        import matplotlib; matplotlib.use("Agg")
        import matplotlib.pyplot as plt, matplotlib.patches as mpatches
    except ImportError:
        log.warning("matplotlib manquant — Figure 3 ignoree."); return

    labels = test_data["labels"]; losses = test_data["window_losses"]
    healthy_mask = labels == 0;   anomalous_mask = labels == 1

    if healthy_mask.sum() == 0 or anomalous_mask.sum() == 0:
        log.warning("Pas de fenetres saines ET anomaliques en test — Figure 3 ignoree.")
        return

    hi = int(np.argmin(np.where(healthy_mask,   losses,  np.inf)))
    ai = int(np.argmax(np.where(anomalous_mask, losses, -np.inf)))
    ts = np.arange(WINDOW_SIZE)

    fig, (ax_top, ax_bot) = plt.subplots(2, 1, figsize=(13, 8))
    fig.suptitle("Figure 3 — Exemples de Reconstruction", fontsize=14, fontweight="bold")

    # ── Panel haut : sain ────────────────────────────────────────────────────
    orig_h  = test_data["originals"][hi, channel_idx, :]
    recon_h = test_data["reconstructions"][hi, channel_idx, :]
    attn_h  = test_data["attention"][hi]

    ax_top.plot(ts, orig_h,  color="steelblue",  linewidth=2.2, label="Signal original")
    ax_top.plot(ts, recon_h, color="darkorange",  linewidth=1.5, linestyle="--",
                label="Signal reconstruit")
    ax_top.set_title(f"Signal SAIN  (perte MAE = {losses[hi]*1e3:.3f} x10^-3)", fontsize=11)
    ax_top.set_ylabel(channel_name, fontsize=10)
    ax_top.set_xlabel("Timestep (secondes)", fontsize=10)
    ax_top.legend(fontsize=9); ax_top.grid(True, alpha=0.3)
    ax2 = ax_top.twinx()
    ax2.fill_between(ts, attn_h, alpha=0.12, color="grey")
    ax2.set_ylabel("Attention α", fontsize=8, color="grey")
    ax2.set_ylim(0, max(attn_h.max() * 4, 1e-8))
    ax2.tick_params(axis="y", labelcolor="grey", labelsize=7)

    # ── Panel bas : anomalique ───────────────────────────────────────────────
    orig_a  = test_data["originals"][ai, channel_idx, :]
    recon_a = test_data["reconstructions"][ai, channel_idx, :]
    attn_a  = test_data["attention"][ai]
    err_a   = np.abs(orig_a - recon_a)
    div_thr = err_a.mean() + err_a.std()
    div_mask = err_a > div_thr

    ax_bot.plot(ts, orig_a,  color="steelblue",  linewidth=2.2, label="Signal original")
    ax_bot.plot(ts, recon_a, color="darkorange",  linewidth=1.5, linestyle="--",
                label="Signal reconstruit")

    in_r = False; r_start = 0
    for t in range(WINDOW_SIZE):
        if div_mask[t] and not in_r:
            r_start = t; in_r = True
        elif not div_mask[t] and in_r:
            ax_bot.axvspan(r_start, t, color="red", alpha=0.18); in_r = False
    if in_r:
        ax_bot.axvspan(r_start, WINDOW_SIZE, color="red", alpha=0.18)

    ax3 = ax_bot.twinx()
    ax3.fill_between(ts, attn_a, alpha=0.15, color="grey")
    ax3.set_ylabel("Attention α", fontsize=8, color="grey")
    ax3.set_ylim(0, max(attn_a.max() * 4, 1e-8))
    ax3.tick_params(axis="y", labelcolor="grey", labelsize=7)

    div_patch = mpatches.Patch(color="red", alpha=0.3, label="Zone de divergence")
    handles, lbls = ax_bot.get_legend_handles_labels()
    ax_bot.legend(handles=handles + [div_patch], fontsize=9)
    ax_bot.set_title(f"Signal ANOMALIQUE  (perte MAE = {losses[ai]*1e3:.3f} x10^-3)", fontsize=11)
    ax_bot.set_ylabel(channel_name, fontsize=10)
    ax_bot.set_xlabel("Timestep (secondes)", fontsize=10)
    ax_bot.grid(True, alpha=0.3)

    plt.tight_layout()
    fig.savefig(save_path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    log.info(f"Figure 3 sauvegardee -> {save_path}")


# ============================================================================
# SECTION 6 — FIGURE 5
# ============================================================================

def plot_figure5(train_data, val_data, test_data,
                 save_path=OUT_DIR / "figure5_loss_distributions.png"):
    try:
        import matplotlib; matplotlib.use("Agg")
        import matplotlib.pyplot as plt
    except ImportError:
        log.warning("matplotlib manquant — Figure 5 ignoree."); return

    test_labels = test_data["labels"]
    groups = [
        ("Train",               train_data["window_losses"],                    "steelblue"),
        ("Validation",          val_data["window_losses"],                      "royalblue"),
        ("Test\n(Sain)",        test_data["window_losses"][test_labels == 0],   "seagreen"),
        ("Test\n(Anomalique)",  test_data["window_losses"][test_labels == 1],   "crimson"),
    ]
    groups = [(n, d, c) for n, d, c in groups if len(d) > 0]

    data_plot   = [np.clip(g[1], 1e-10, None) for g in groups]
    labels_plot = [g[0] for g in groups]
    colors_plot = [g[2] for g in groups]

    fig, ax = plt.subplots(figsize=(10, 6))
    fig.suptitle("Figure 5 — Distribution des pertes de reconstruction",
                 fontsize=13, fontweight="bold")

    bp = ax.boxplot(data_plot, labels=labels_plot, patch_artist=True, notch=False,
                    showfliers=True,
                    flierprops=dict(marker="o", markersize=3, alpha=0.35, linestyle="none"),
                    medianprops=dict(color="black", linewidth=2))
    for patch, color in zip(bp["boxes"], colors_plot):
        patch.set_facecolor(color); patch.set_alpha(0.55)

    ax.set_yscale("log")
    ax.set_ylabel("Perte de reconstruction — MAE (log)", fontsize=11)
    ax.set_xlabel("Groupe", fontsize=11)
    ax.grid(True, axis="y", which="both", alpha=0.3)

    for i, arr in enumerate(data_plot, start=1):
        med = float(np.median(arr))
        ax.text(i, med * 1.6, f"med={med*1e3:.2f}e-3",
                ha="center", va="bottom", fontsize=8, color="black")

    plt.tight_layout()
    fig.savefig(save_path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    log.info(f"Figure 5 sauvegardee -> {save_path}")


# ============================================================================
# SECTION 7 — LECTURE DU THRESHOLD DEPUIS TABLE I
# ============================================================================

def read_threshold_from_table_I(table_i_path: Path) -> Tuple[float, float]:
    if not table_i_path.exists():
        log.warning(f"table_I_metrics.csv introuvable — seuils par defaut.")
        return 0.0, 5.0
    df = pd.read_csv(table_i_path)
    if df.empty:
        return 0.0, 5.0
    row = df.iloc[0]
    win   = float(row.get("Window Loss Threshold", 0.0))
    subj  = float(row.get("Optimal Subject Threshold (%)", 5.0))
    log.info(f"Seuils lus -> window={win:.6f} | subject={subj:.1f}%")
    return win, subj


# ============================================================================
# SECTION 8 — MAIN
# ============================================================================

def parse_args():
    p = argparse.ArgumentParser(description=" Task 2 — Figures 2, 3, 5 + Table II")
    p.add_argument("--npy-dir",    required=True,
                   help="Dossier .npy de  (patient_XX_windows.npy)")
    p.add_argument("--checkpoint", default="results/best_model.pt")
    p.add_argument("--batch-size", type=int, default=256)
    p.add_argument("--train-ratio", type=float, default=0.70)
    p.add_argument("--val-ratio",   type=float, default=0.15)
    p.add_argument("--seed",        type=int,   default=SEED)
    return p.parse_args()


def main():
    args   = parse_args()
    OUT_DIR.mkdir(parents=True, exist_ok=True)

    npy_dir   = Path(args.npy_dir)
    ckpt_path = Path(args.checkpoint)
    device    = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    log.info(f"Device : {device}")

    patient_ids = discover_patient_ids(npy_dir)
    log.info(f"{len(patient_ids)} patients : {patient_ids}")

    train_ids, val_ids, test_ids = split_patient_ids(
        patient_ids, args.train_ratio, args.val_ratio, args.seed
    )

    model = load_model(ckpt_path, device)

    log.info("=== Inference Train ===")
    train_data = run_inference(model, make_loader(npy_dir, train_ids, args.batch_size), device)
    log.info("=== Inference Val ===")
    val_data   = run_inference(model, make_loader(npy_dir, val_ids,   args.batch_size), device)
    log.info("=== Inference Test ===")
    test_data  = run_inference(model, make_loader(npy_dir, test_ids,  args.batch_size), device)

    # TABLE II
    log.info("=== Table II ===")
    build_and_print_table_II(train_data, val_data, test_data,
                             existing_csv=OUT_DIR / "table_II_reconstruction.csv")

    # SEUILS
    window_threshold, subject_threshold = read_threshold_from_table_I(
        OUT_DIR / "table_I_metrics.csv"
    )
    if window_threshold == 0.0:
        healthy_train = train_data["window_losses"][train_data["labels"] == 0]
        window_threshold = float(np.percentile(healthy_train, 85))
        log.info(f"window_threshold calcule = {window_threshold:.6f}")

    # FIGURE 2
    log.info("=== Figure 2 ===")
    subject_df = compute_subject_anomaly_pct(
        test_data, window_threshold, npy_dir, test_ids
    )
    if not subject_df.empty:
        log.info(f"\n{subject_df.to_string(index=False)}\n")
        subject_df.to_csv(OUT_DIR / "subject_anomaly_pct.csv", index=False)
        plot_figure2(subject_df, subject_threshold)

    # FIGURE 3
    log.info("=== Figure 3 ===")
    plot_figure3(test_data)

    # FIGURE 5
    log.info("=== Figure 5 ===")
    plot_figure5(train_data, val_data, test_data)

    print("\n" + "="*60)
    print("TASK 2 TERMINE. Fichiers produits :")
    for p in [
        OUT_DIR / "table_II_reconstruction.csv",
        OUT_DIR / "figure2_anomaly_scores.png",
        OUT_DIR / "figure3_reconstruction.png",
        OUT_DIR / "figure5_loss_distributions.png",
        OUT_DIR / "subject_anomaly_pct.csv",
    ]:
        print(f"  {'OK' if p.exists() else 'MANQUANT'}  {p}")
    print("="*60 + "\n")


if __name__ == "__main__":
    main()