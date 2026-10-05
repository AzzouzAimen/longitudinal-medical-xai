"""
Thermal noise robustness analysis.
Injects double-sigma noise into test set, measures F1 degradation.
Output: results/benchmarks/
"""

import os
import argparse, csv, json, logging, sys
from pathlib import Path
import numpy as np
import torch
from sklearn.metrics import f1_score, precision_score, recall_score

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from dotenv import load_dotenv
load_dotenv(Path(__file__).resolve().parent.parent.parent / ".env")
logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
log = logging.getLogger("robustness")

OUTPUT_DIR  = Path(__file__).resolve().parent.parent / "results" / "benchmarks"
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
DEVICE      = torch.device("cuda" if torch.cuda.is_available() else "cpu")
CHANNEL_IDX = [2, 3, 4]   # left_norm, right_norm, asymmetry
BATCH_SIZE  = 128

def load_windows(npy_dir, patient_ids):
    windows, labels = [], []
    for pid in patient_ids:
        npy  = npy_dir / f"{pid}_windows.npy"
        meta = npy_dir / f"{pid}_windows_meta.csv"
        if not npy.exists():
            continue
        arr = np.load(npy)[:, CHANNEL_IDX, :].astype(np.float32)
        if meta.exists():
            import csv as _csv
            with open(meta) as f:
                lbs = [int(r["label"]) for r in _csv.DictReader(f)]
            lbs = np.array(lbs[:len(arr)])
        else:
            lbs = np.zeros(len(arr), dtype=int)
        windows.append(arr); labels.append(lbs)
    return np.concatenate(windows), np.concatenate(labels)

def load_model(model_path):
    from model import TAAE
    model = TAAE(n_channels=3, window_size=60)
    ckpt  = torch.load(model_path, map_location=DEVICE)
    model.load_state_dict(ckpt.get("model_state_dict", ckpt))
    return model.to(DEVICE).eval()

def get_losses(model, windows):
    losses = []
    with torch.no_grad():
        for i in range(0, len(windows), BATCH_SIZE):
            x     = torch.tensor(windows[i:i+BATCH_SIZE]).to(DEVICE)
            x_hat, _ = model(x)
            losses.extend(((x - x_hat)**2).mean(dim=(1,2)).cpu().numpy())
    return np.array(losses)

def metrics(losses, labels, tau):
    preds = (losses > tau).astype(int)
    return {
        "f1":        round(float(f1_score(labels,        preds, zero_division=0)), 4),
        "precision": round(float(precision_score(labels, preds, zero_division=0)), 4),
        "recall":    round(float(recall_score(labels,    preds, zero_division=0)), 4),
    }

def main():
    p = argparse.ArgumentParser()
    p.add_argument("--model",   default="results/best_model.pt")
    p.add_argument("--npy-dir", default=os.getenv("NPY_DIR"))
    p.add_argument("--patients", type=int, default=None)
    p.add_argument("--seed",    type=int, default=42)
    args = p.parse_args()
    np.random.seed(args.seed); torch.manual_seed(args.seed)

    npy_dir = Path(args.npy_dir or os.getenv("NPY_DIR"))
    pids    = sorted(p.stem.replace("_windows","") for p in npy_dir.glob("patient_*_windows.npy"))
    if args.patients: pids = pids[:args.patients]

    log.info(f"Loading {len(pids)} patients...")
    windows, labels = load_windows(npy_dir, pids)

    # Split: last 20% = test, rest healthy = threshold calibration
    n        = len(windows)
    test_idx = np.arange(int(n * 0.8), n)
    train_h  = np.where(labels[:int(n * 0.8)] == 0)[0]

    test_w, test_l = windows[test_idx], labels[test_idx]
    train_hw       = windows[train_h]

    log.info(f"Test: {len(test_w)} windows ({test_l.sum()} anomalous)")

    model = load_model(Path(args.model))

    # Threshold τ = 85th percentile of healthy train losses
    train_losses = get_losses(model, train_hw)
    tau = float(np.percentile(train_losses, 85))
    log.info(f"τ = {tau:.6f}")

    # sigma per channel from healthy training windows
    sigma = train_hw.std(axis=(0, 2))   # (C,)
    log.info(f"σ_original = {sigma}")

    # Clean test
    clean_losses  = get_losses(model, test_w)
    clean_m       = metrics(clean_losses, test_l, tau)

    # Noisy test (2σ)
    noise         = np.random.normal(0, 2*sigma[None,:,None], test_w.shape).astype(np.float32)
    noisy_losses  = get_losses(model, test_w + noise)
    noisy_m       = metrics(noisy_losses, test_l, tau)

    deg = (clean_m["f1"] - noisy_m["f1"]) / max(clean_m["f1"], 1e-9) * 100

    print(f"\n{'Metric':<12} {'Clean':>8} {'Noisy(2σ)':>10} {'Δ':>8}")
    print("-" * 42)
    for k in ["f1", "precision", "recall"]:
        print(f"{k.upper():<12} {clean_m[k]:>8.4f} {noisy_m[k]:>10.4f} {noisy_m[k]-clean_m[k]:>+8.4f}")
    print(f"\nF1 degradation: {deg:.2f}%")

    csv_path = OUTPUT_DIR / "robustness_results.csv"
    with open(csv_path, "w", newline="") as f:
        w = csv.DictWriter(f, ["condition","f1","precision","recall"])
        w.writeheader()
        w.writerow({"condition":"clean",       **clean_m})
        w.writerow({"condition":"noisy_2sigma", **noisy_m})

    json_path = OUTPUT_DIR / "robustness_results.json"
    with open(json_path, "w") as f:
        json.dump({"tau": tau, "sigma_original": sigma.tolist(),
                   "clean": clean_m, "noisy_2sigma": noisy_m,
                   "f1_degradation_pct": round(deg, 2)}, f, indent=2)

    log.info(f"Saved → {csv_path}, {json_path}")

if __name__ == "__main__":
    main()
