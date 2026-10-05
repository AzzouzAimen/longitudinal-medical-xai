from dataclasses import dataclass
import json
from pathlib import Path
import numpy as np
import torch
from medattnaid.data import fetch_windows, split_healthy_windows
from medattnaid.metrics import detection_metrics, reconstruction_metrics
from medattnaid.model import TemporalAttentionAutoencoder


# Parametres utilises pour charger le modele et produire les resultats d'evaluation.
@dataclass
class EvaluationConfig:
    model_path: Path = Path("results/model/best_model.pt")
    subject: str = "chb01"
    channel: str = "ALL"
    window_length: int = 100
    stride: int = 100
    limit: int = 240000
    batch_size: int = 128
    output_path: Path = Path("results/evaluation_results.json")
    threshold_percentile: int = 85


# Evalue le modele par erreur de reconstruction et applique un seuil d'anomalie.
def evaluate(args: EvaluationConfig):
    batch = fetch_windows(args.subject, args.channel, args.window_length, args.stride, args.limit)
    train_idx, val_idx, test_idx = split_healthy_windows(batch)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = TemporalAttentionAutoencoder(input_dim=batch.windows.shape[-1]).to(device)
    model.load_state_dict(torch.load(args.model_path, map_location=device))
    model.eval()

    # On reconstruit toutes les fenetres pour obtenir une erreur MSE par fenetre.
    x = torch.tensor(batch.windows, device=device)
    losses = []
    recon = []
    attn = []
    with torch.no_grad():
        for start in range(0, len(x), args.batch_size):
            x_hat, alpha = model(x[start:start + args.batch_size])
            mse = torch.mean((x[start:start + args.batch_size] - x_hat) ** 2, dim=(1, 2))
            losses.extend(mse.cpu().numpy().tolist())
            recon.append(x_hat.cpu().numpy())
            attn.append(alpha.cpu().numpy())
    recon = np.concatenate(recon)
    attn = np.concatenate(attn)
    reconstruction_errors = np.asarray(losses)

    # Le seuil suit le sujet: percentile 85 des erreurs sur les fenetres saines d'entrainement.
    tau = float(np.percentile(reconstruction_errors[train_idx], args.threshold_percentile))
    y_true = batch.labels[test_idx]
    scores = reconstruction_errors[test_idx]
    y_pred = (scores > tau).astype(int)

    # Les resultats gardent les metriques globales et les details par fenetre pour l'analyse.
    results = {
        "threshold": tau,
        "detection": detection_metrics(y_true, y_pred, scores),
        "reconstruction": {
            "validation": reconstruction_metrics(batch.windows[val_idx], recon[val_idx]),
            "test": reconstruction_metrics(batch.windows[test_idx], recon[test_idx]),
        },
        "window_results": [
            {
                "window_id": int(i),
                "subject_id": str(batch.subject_ids[i]),
                "recording_id": str(batch.recording_ids[i]),
                "channel": str(batch.channels[i]),
                "start_time": str(batch.start_times[i]),
                "label": int(batch.labels[i]),
                "reconstruction_error": float(reconstruction_errors[i]),
                "is_anomaly": bool(reconstruction_errors[i] > tau),
                "attention_weights": attn[i].astype(float).round(8).tolist(),
            }
            for i in test_idx
        ],
    }
    args.output_path.parent.mkdir(parents=True, exist_ok=True)
    args.output_path.write_text(json.dumps(results, indent=2))


if __name__ == "__main__":
    evaluate(EvaluationConfig())
