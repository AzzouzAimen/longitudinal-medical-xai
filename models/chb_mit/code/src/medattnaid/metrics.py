import numpy as np
from scipy.stats import pearsonr
from sklearn.metrics import precision_recall_fscore_support, roc_auc_score, accuracy_score


def reconstruction_metrics(x, x_hat):
    x = np.asarray(x).reshape(len(x), -1)
    x_hat = np.asarray(x_hat).reshape(len(x_hat), -1)
    diff = x - x_hat
    mae = float(np.mean(np.abs(diff)))
    rmse = float(np.sqrt(np.mean(diff ** 2)))
    rho_values = []
    cos_values = []
    for a, b in zip(x, x_hat):
        if np.std(a) > 0 and np.std(b) > 0:
            rho_values.append(pearsonr(a, b)[0])
        denom = np.linalg.norm(a) * np.linalg.norm(b)
        if denom > 0:
            cos_values.append(float(np.dot(a, b) / denom))
    return {
        "mae": mae,
        "rmse": rmse,
        "pearson": float(np.mean(rho_values)) if rho_values else 0.0,
        "cosine": float(np.mean(cos_values)) if cos_values else 0.0,
    }


def detection_metrics(y_true, y_pred, scores=None):
    precision, recall, f1, _ = precision_recall_fscore_support(
        y_true, y_pred, average="binary", zero_division=0
    )
    result = {
        "precision": float(precision),
        "recall": float(recall),
        "f1": float(f1),
        "accuracy": float(accuracy_score(y_true, y_pred)),
    }
    if scores is not None and len(set(y_true)) == 2:
        result["auc_roc"] = float(roc_auc_score(y_true, scores))
    else:
        result["auc_roc"] = 0.0
    return result


def wilson_interval(correct, total, z=1.96):
    if total == 0:
        return 0.0, 0.0
    phat = correct / total
    denom = 1 + z ** 2 / total
    centre = phat + z ** 2 / (2 * total)
    spread = z * np.sqrt((phat * (1 - phat) + z ** 2 / (4 * total)) / total)
    return (centre - spread) / denom, (centre + spread) / denom
