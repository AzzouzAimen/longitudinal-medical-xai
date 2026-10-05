# =============================================================================
# UCI HAR model evaluation
# =============================================================================
# Script d'évaluation
#
# Ce script intègre :
#   -   : calcul du seuil τ (85e percentile sur les fenêtres normales)
#   -   : métriques signal (F1, Précision, Rappel) + Individual Accuracy
#   -  : assemblage final, rapport console + export JSON/CSV
#
# Usage :
#   python eval.py --dataset_root UCI_HAR --checkpoint best_model.pt --device cpu
#   python eval.py --dataset_root UCI_HAR --checkpoint best_model.pt \
#                  --percentile 85 --device cuda --output_dir results/
#
# Pipeline complet (Algorithm 4) :
#   1. Charger le modèle TAAE depuis le checkpoint
#   2. Calculer τ sur les fenêtres normales du train ()
#   3. Reconstruire les fenêtres du test → erreurs MSE par fenêtre
#   4. Prédire anomalie si erreur > τ
#   5. Calculer les métriques signal : F1, Précision, Rappel, Accuracy ()
#   6. Calculer l'Individual Accuracy par sujet ()
#   7. Afficher le rapport final + sauvegarder les résultats
# =============================================================================

from __future__ import annotations

import argparse
import json
import time
from collections import defaultdict
from pathlib import Path
from typing import Dict, Tuple

import numpy as np
import torch
from torch.utils.data import DataLoader, TensorDataset

try:
    from model import TAAE
except ImportError as exc:
    raise ImportError(
        "Impossible d'importer TAAE. Assurez-vous que model.py est disponible."
    ) from exc

# =============================================================================
# Constantes
# =============================================================================

CHANNEL_FILES = [
    "body_acc_x.npy",
    "body_acc_y.npy",
    "body_acc_z.npy",
    "body_gyro_x.npy",
    "body_gyro_y.npy",
    "body_gyro_z.npy",
    "total_acc_x.npy",
    "total_acc_y.npy",
    "total_acc_z.npy",
]

NORMAL_LABEL = 1  # WALKING = classe normale dans UCI-HAR

ACTIVITY_NAMES = {
    1: "WALKING (Normal)",
    2: "WALKING_UPSTAIRS",
    3: "WALKING_DOWNSTAIRS",
    4: "SITTING",
    5: "STANDING",
    6: "LAYING",
}


# =============================================================================
# Chargement des données (nettoyées en .npy)
# =============================================================================

def load_split(
    dataset_root: str,
    split: str,
    require_subjects: bool = True,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray | None]:
    """
    Charge un split (train ou test) depuis les fichiers .npy pré-traités.

    Args:
        dataset_root    : Racine du dataset (contient train/ et test/).
        split           : 'train' ou 'test'.
        require_subjects: Si True, charge subjects.npy ; lève une erreur si absent.

    Returns:
        X        : (N, 128, 9)  float32
        labels   : (N,)         int64
        subjects : (N,)         int64  ou None si require_subjects=False et absent
    """
    root = Path(dataset_root)
    split_dir = root / split

    if not split_dir.exists():
        raise FileNotFoundError(f"Dossier introuvable : {split_dir}")

    arrays = []
    for fname in CHANNEL_FILES:
        path = split_dir / fname
        if not path.exists():
            raise FileNotFoundError(f"Fichier manquant : {path}")
        arrays.append(np.load(str(path)).astype(np.float32))

    X = np.stack(arrays, axis=-1)                         # (N, 128, 9)
    labels = np.load(str(split_dir / "labels.npy")).astype(np.int64).ravel()

    subjects = None
    subjects_path = split_dir / "subjects.npy"
    if subjects_path.exists():
        subjects = np.load(str(subjects_path)).astype(np.int64).ravel()
    elif require_subjects:
        raise FileNotFoundError(f"Fichier manquant : {subjects_path}")

    if len(X) != len(labels):
        raise ValueError(
            f"Mismatch : X={len(X)} fenêtres, labels={len(labels)} entrées"
        )
    if subjects is not None and len(X) != len(subjects):
        raise ValueError(
            f"Mismatch : X={len(X)} fenêtres, subjects={len(subjects)} entrées"
        )

    return X, labels, subjects


# =============================================================================
# Chargement du modèle
# =============================================================================

def load_model(checkpoint_path: str, device: torch.device) -> torch.nn.Module:
    """
    Instancie le TAAE et charge les poids depuis le checkpoint.

    Args:
        checkpoint_path : Chemin vers le fichier .pt sauvegardé par train.py.
        device          : Dispositif cible (cpu / cuda / mps).

    Returns:
        model : TAAE en mode eval(), prêt pour l'inférence.
    """
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

    checkpoint = Path(checkpoint_path)
    if not checkpoint.exists():
        raise FileNotFoundError(f"Checkpoint introuvable : {checkpoint}")

    state_dict = torch.load(str(checkpoint), map_location=device)
    model.load_state_dict(state_dict)
    model.eval()
    return model


# =============================================================================
#  — Calcul du seuil τ
# =============================================================================

def compute_reconstruction_errors(
    model: torch.nn.Module,
    X: np.ndarray,
    device: torch.device,
    batch_size: int = 64,
) -> np.ndarray:
    """
    Calcule l'erreur de reconstruction MSE pour chaque fenêtre.

    Formule : e_i = (1 / (L * F)) * Σ_{t,f} (x[t,f] - x_hat[t,f])^2

    Args:
        model      : TAAE en eval mode.
        X          : (N, 128, 9) float32.
        device     : Dispositif de calcul.
        batch_size : Taille des mini-batches pour l'inférence.

    Returns:
        errors : (N,) float32 — erreur MSE par fenêtre.
    """
    dataset = TensorDataset(torch.from_numpy(X))
    loader = DataLoader(dataset, batch_size=batch_size, shuffle=False)

    all_errors = []
    with torch.no_grad():
        for batch in loader:
            x = batch[0].to(device)
            x_hat, *_ = model(x)
            # Moyenne sur les dimensions temporelle et feature → scalaire par fenêtre
            batch_errors = torch.mean((x - x_hat) ** 2, dim=(1, 2))
            all_errors.append(batch_errors.cpu().numpy())

    return np.concatenate(all_errors, axis=0)


def compute_threshold(
    errors: np.ndarray,
    labels: np.ndarray,
    percentile: int = 85,
) -> float:
    """
    Calcule le seuil d'anomalie τ comme le `percentile`-ième centile des
    erreurs de reconstruction sur les fenêtres normales (, Algo 4).

    Args:
        errors     : (N,) erreurs MSE.
        labels     : (N,) étiquettes (NORMAL_LABEL = 1 pour normal).
        percentile : Centile utilisé (défaut = 85).

    Returns:
        tau : float — seuil d'anomalie.
    """
    normal_mask = labels == NORMAL_LABEL
    normal_errors = errors[normal_mask]

    if normal_errors.size == 0:
        raise ValueError(
            "Aucune fenêtre normale trouvée dans le split train. "
            "Vérifiez NORMAL_LABEL et la structure des données."
        )

    tau = float(np.percentile(normal_errors, percentile))
    return tau


# =============================================================================
#  — Métriques signal et Individual Accuracy
# =============================================================================

def binarize_labels(labels: np.ndarray) -> np.ndarray:
    """
    Convertit les étiquettes UCI-HAR en binaire :
        0 → normal (WALKING, label=1)
        1 → anomalie (toutes les autres classes)
    """
    return (labels != NORMAL_LABEL).astype(int)


def compute_signal_metrics(
    y_true: np.ndarray,
    y_pred: np.ndarray,
) -> Dict[str, float]:
    """
    Calcule les métriques de classification au niveau fenêtre.

    Returns:
        dict avec TP, FP, FN, TN, Précision, Rappel, F1, Accuracy.
    """
    tp = int(np.sum((y_true == 1) & (y_pred == 1)))
    fp = int(np.sum((y_true == 0) & (y_pred == 1)))
    fn = int(np.sum((y_true == 1) & (y_pred == 0)))
    tn = int(np.sum((y_true == 0) & (y_pred == 0)))

    precision = tp / (tp + fp) if (tp + fp) > 0 else 0.0
    recall    = tp / (tp + fn) if (tp + fn) > 0 else 0.0
    f1        = (
        2 * precision * recall / (precision + recall)
        if (precision + recall) > 0
        else 0.0
    )
    accuracy  = (tp + tn) / max(1, len(y_true))

    return {
        "tp":        tp,
        "fp":        fp,
        "fn":        fn,
        "tn":        tn,
        "precision": round(precision, 6),
        "recall":    round(recall,    6),
        "f1":        round(f1,        6),
        "accuracy":  round(accuracy,  6),
    }


def compute_individual_accuracy(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    subject_ids: np.ndarray,
) -> Dict[str, float]:
    """
    Calcule l'Individual Accuracy (per-subject classification).

    Pour chaque sujet, calcule le % de fenêtres classées comme anomalie.
    Un seuil optimal est recherché pour maximiser la précision de
    classification sujet-par-sujet (normal vs. anomalous subject).

    Args:
        y_true      : (N,) étiquettes binaires fenêtre.
        y_pred      : (N,) prédictions binaires fenêtre.
        subject_ids : (N,) identifiants de sujets.

    Returns:
        dict avec subject_count, threshold, individual_accuracy.
    """
    subject_stats = defaultdict(
        lambda: {"anomalous_windows": 0, "total_windows": 0, "has_anomaly": False}
    )

    for sid, yt, yp in zip(subject_ids, y_true, y_pred):
        s = subject_stats[int(sid)]
        s["total_windows"]     += 1
        s["anomalous_windows"] += int(yp == 1)
        if yt == 1:
            s["has_anomaly"] = True

    percentages = []
    true_labels  = []
    for s in subject_stats.values():
        pct = 100.0 * s["anomalous_windows"] / s["total_windows"]
        percentages.append(pct)
        true_labels.append(int(s["has_anomaly"]))

    if not percentages:
        return {"subject_count": 0, "threshold_pct": 0.0, "individual_accuracy": 0.0}

    # Recherche du seuil optimal (percentage anomalous windows)
    best_threshold = 0.0
    best_accuracy  = -1.0
    for candidate in sorted(set(percentages)):
        correct = sum(
            1 for pct, tl in zip(percentages, true_labels)
            if (int(pct > candidate) == tl)
        )
        acc = correct / len(percentages)
        if acc > best_accuracy:
            best_accuracy  = acc
            best_threshold = candidate

    return {
        "subject_count":       len(percentages),
        "threshold_pct":       round(best_threshold, 4),
        "individual_accuracy": round(best_accuracy,  6),
    }


def compute_per_activity_metrics(
    errors: np.ndarray,
    labels: np.ndarray,
    tau: float,
) -> Dict[int, Dict[str, float]]:
    """
    Calcule l'erreur de reconstruction moyenne et le taux de détection
    pour chaque classe d'activité (analyse qualitative).

    Returns:
        dict { label_int : { 'mean_error', 'detection_rate', 'count' } }
    """
    results = {}
    for activity_id in np.unique(labels):
        mask      = labels == activity_id
        act_errors = errors[mask]
        results[int(activity_id)] = {
            "count":          int(mask.sum()),
            "mean_error":     round(float(np.mean(act_errors)), 8),
            "std_error":      round(float(np.std(act_errors)),  8),
            "detection_rate": round(float(np.mean(act_errors > tau)), 6),
        }
    return results


# =============================================================================
#  — Rapport console + export
# =============================================================================

def print_report(
    tau: float,
    percentile: int,
    signal_metrics: Dict,
    individual_metrics: Dict,
    per_activity: Dict,
    elapsed: float,
) -> None:
    """Affiche le rapport d'évaluation complet sur la console."""

    sep = "=" * 65

    print(f"\n{sep}")
    print("  RAPPORT D'ÉVALUATION — TAAE |  | Topic M5")
    print(f"  Human Activity Anomaly Detection — Dataset UCI-HAR")
    print(sep)

    # ── Seuil ─────────────────────────────────────────────────────────────────
    print(f"\n{'─'*65}")
    print(f"  SEUIL D'ANOMALIE τ  ( — {percentile}e centile)")
    print(f"{'─'*65}")
    print(f"  τ = {tau:.8f}")
    print(f"  Toute fenêtre avec erreur MSE > τ est classée ANOMALIE")

    # ── Métriques signal ──────────────────────────────────────────────────────
    sm = signal_metrics
    print(f"\n{'─'*65}")
    print(f"  MÉTRIQUES SIGNAL ( — niveau fenêtre)")
    print(f"{'─'*65}")
    print(f"  TP={sm['tp']:>6}  FP={sm['fp']:>6}  FN={sm['fn']:>6}  TN={sm['tn']:>6}")
    print(f"  Précision  : {sm['precision']:.4f}")
    print(f"  Rappel     : {sm['recall']:.4f}")
    print(f"  F1-Score   : {sm['f1']:.4f}")
    print(f"  Accuracy   : {sm['accuracy']:.4f}")

    # ── Individual Accuracy ───────────────────────────────────────────────────
    im = individual_metrics
    print(f"\n{'─'*65}")
    print(f"  INDIVIDUAL ACCURACY ( — niveau sujet)")
    print(f"{'─'*65}")
    print(f"  Sujets évalués        : {im['subject_count']}")
    print(f"  Seuil optimal (% anomalie / sujet) : {im['threshold_pct']:.2f}%")
    print(f"  Individual Accuracy   : {im['individual_accuracy']:.4f}")

    # ── Par activité ──────────────────────────────────────────────────────────
    print(f"\n{'─'*65}")
    print(f"  ANALYSE PAR ACTIVITÉ")
    print(f"{'─'*65}")
    header = f"  {'Activité':<30} {'N':>6} {'Err. moy.':>12} {'Taux détect.':>14}"
    print(header)
    print(f"  {'-'*60}")
    for act_id, stats in sorted(per_activity.items()):
        name = ACTIVITY_NAMES.get(act_id, f"Classe {act_id}")
        marker = "← NORMAL" if act_id == NORMAL_LABEL else ""
        print(
            f"  {name:<30} {stats['count']:>6} "
            f"{stats['mean_error']:>12.6f} "
            f"{stats['detection_rate']:>13.4f}  {marker}"
        )

    # ── Pied de page ──────────────────────────────────────────────────────────
    print(f"\n{'─'*65}")
    print(f"  Temps d'évaluation : {elapsed:.2f}s")
    print(sep + "\n")


def save_results(
    output_dir: str,
    tau: float,
    signal_metrics: Dict,
    individual_metrics: Dict,
    per_activity: Dict,
    test_errors: np.ndarray,
    y_true: np.ndarray,
    y_pred: np.ndarray,
) -> None:
    """
    Sauvegarde les résultats dans output_dir/ :
        - eval_results.json  : métriques + seuil
        - window_errors.csv  : erreur + vrai label + prédiction par fenêtre
    """
    out = Path(output_dir)
    out.mkdir(parents=True, exist_ok=True)

    # JSON — métriques globales
    summary = {
        "threshold":          tau,
        "signal_metrics":     signal_metrics,
        "individual_metrics": individual_metrics,
        "per_activity":       {
            ACTIVITY_NAMES.get(k, str(k)): v
            for k, v in per_activity.items()
        },
    }
    json_path = out / "eval_results.json"
    with open(json_path, "w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2, ensure_ascii=False)
    print(f"  ✔ Résultats sauvegardés → {json_path}")

    # CSV — erreurs par fenêtre
    csv_path = out / "window_errors.csv"
    with open(csv_path, "w", encoding="utf-8") as f:
        f.write("window_id,mse_error,true_label,pred_label\n")
        for i, (err, yt, yp) in enumerate(zip(test_errors, y_true, y_pred)):
            f.write(f"{i},{err:.8f},{yt},{yp}\n")
    print(f"  ✔ Erreurs fenêtres sauvegardées → {csv_path}")


# =============================================================================
# Pipeline principal — Algorithm 4 (, assemblage final)
# =============================================================================

def evaluate(
    dataset_root:  str,
    checkpoint:    str,
    device_name:   str  = "cpu",
    percentile:    int  = 85,
    batch_size:    int  = 64,
    output_dir:    str  = "eval_results",
    save:          bool = True,
) -> Dict:
    """
    Pipeline complet d'évaluation (Algorithm 4).

    Retourne un dictionnaire récapitulatif des résultats.
    """
    t_start = time.time()
    device  = torch.device(device_name)

    print(f"\n{'='*65}")
    print("  Évaluation TAAE — Algorithm 4 ()")
    print(f"{'='*65}")
    print(f"  Dataset    : {dataset_root}")
    print(f"  Checkpoint : {checkpoint}")
    print(f"  Device     : {device}")
    print(f"  Percentile : {percentile}")

    # ── Étape 1 : Charger le modèle ──────────────────────────────────────────
    print("\n[Étape 1] Chargement du modèle TAAE…")
    model = load_model(checkpoint, device)
    n_params = sum(p.numel() for p in model.parameters())
    print(f"  Paramètres totaux : {n_params:,}")

    # ── Étape 2 : Données train (pour le seuil τ) ────────────────────────────
    print("\n[Étape 2] Chargement des données train (calcul de τ)…")
    X_train, labels_train, _ = load_split(
        dataset_root, "train", require_subjects=False
    )
    print(f"  Train : {len(X_train)} fenêtres "
          f"(dont {int(np.sum(labels_train == NORMAL_LABEL))} normales)")

    # ── Étape 3 : Erreurs de reconstruction sur le train ────────────────────
    print("\n[Étape 3] Calcul des erreurs de reconstruction (train)…")
    train_errors = compute_reconstruction_errors(model, X_train, device, batch_size)

    # ── Étape 4 : Seuil τ () ────────────────────────────────────────
    print(f"\n[Étape 4 — ] Calcul du seuil τ ({percentile}e centile)…")
    tau = compute_threshold(train_errors, labels_train, percentile=percentile)
    print(f"  τ = {tau:.8f}")

    # ── Étape 5 : Données test ───────────────────────────────────────────────
    print("\n[Étape 5] Chargement des données test…")
    X_test, labels_test, subjects_test = load_split(
        dataset_root, "test", require_subjects=True
    )
    print(f"  Test  : {len(X_test)} fenêtres "
          f"(dont {int(np.sum(labels_test == NORMAL_LABEL))} normales)")

    # ── Étape 6 : Erreurs test + prédictions ─────────────────────────────────
    print("\n[Étape 6] Calcul des erreurs de reconstruction (test)…")
    test_errors = compute_reconstruction_errors(model, X_test, device, batch_size)

    y_true = binarize_labels(labels_test)
    y_pred = (test_errors > tau).astype(int)

    # ── Étape 7 : Métriques signal () ────────────────────────────────
    print("\n[Étape 7 — ] Calcul des métriques signal…")
    signal_metrics = compute_signal_metrics(y_true, y_pred)

    # ── Étape 8 : Individual Accuracy () ─────────────────────────────
    print("\n[Étape 8 — ] Calcul de l'Individual Accuracy…")
    individual_metrics = compute_individual_accuracy(y_true, y_pred, subjects_test)

    # ── Étape 9 : Analyse par activité ───────────────────────────────────────
    per_activity = compute_per_activity_metrics(test_errors, labels_test, tau)

    elapsed = time.time() - t_start

    # ── Étape 10 : Rapport () ───────────────────────────────────────
    print_report(tau, percentile, signal_metrics, individual_metrics,
                 per_activity, elapsed)

    # ── Étape 11 : Sauvegarde ─────────────────────────────────────────────────
    if save:
        print("[Sauvegarde des résultats]")
        save_results(output_dir, tau, signal_metrics, individual_metrics,
                     per_activity, test_errors, y_true, y_pred)

    return {
        "tau":                tau,
        "signal_metrics":     signal_metrics,
        "individual_metrics": individual_metrics,
        "per_activity":       per_activity,
        "elapsed":            elapsed,
    }


# =============================================================================
# CLI
# =============================================================================

def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="eval.py — Évaluation TAAE (, )",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument(
        "--dataset_root", type=str, required=True,
        help="Chemin vers le dossier UCI_HAR (contient train/ et test/).",
    )
    parser.add_argument(
        "--checkpoint", type=str, default="best_model.pt",
        help="Chemin vers le checkpoint .pt sauvegardé par train.py.",
    )
    parser.add_argument(
        "--device", type=str, default="cpu",
        choices=["cpu", "cuda", "mps", "auto"],
        help="Dispositif de calcul.",
    )
    parser.add_argument(
        "--percentile", type=int, default=85,
        help="Centile pour le calcul du seuil τ sur les fenêtres normales.",
    )
    parser.add_argument(
        "--batch_size", type=int, default=64,
        help="Taille des mini-batches pour l'inférence.",
    )
    parser.add_argument(
        "--output_dir", type=str, default="eval_results",
        help="Dossier de sortie pour les résultats JSON/CSV.",
    )
    parser.add_argument(
        "--no_save", action="store_true",
        help="Ne pas sauvegarder les résultats sur disque.",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()

    if args.device == "auto":
        if torch.cuda.is_available():
            device = "cuda"
        elif torch.backends.mps.is_available():
            device = "mps"
        else:
            device = "cpu"
    else:
        device = args.device

    evaluate(
        dataset_root = args.dataset_root,
        checkpoint   = args.checkpoint,
        device_name  = device,
        percentile   = args.percentile,
        batch_size   = args.batch_size,
        output_dir   = args.output_dir,
        save         = not args.no_save,
    )


if __name__ == "__main__":
    main()
