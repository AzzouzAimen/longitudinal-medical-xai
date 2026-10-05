import torch
from torch import nn


# Fonction de perte clinique qui combine fidelite, forme du signal et tendance locale.
class ClinicalPatternLoss(nn.Module):
    def __init__(self, stats=None, lambda_fidelity=0.3, lambda_pattern=0.3, lambda_trend=0.4):
        super().__init__()
        self.lambda_fidelity = lambda_fidelity
        self.lambda_pattern = lambda_pattern
        self.lambda_trend = lambda_trend
        self.stats = stats or {
            "pattern_min": 0.0404,
            "pattern_max": 1.5859,
            "trend_min": 0.000018451,
            "trend_max": 0.00083290,
            "fidelity_min": 0.00000013287,
            "fidelity_max": 0.00015482,
        }

    # Calcule la perte totale normalisee avec les poids demandes dans le sujet.
    def forward(self, x, x_hat):
        fidelity = torch.mean((x - x_hat) ** 2)
        pattern = self.pattern_loss(x, x_hat)
        trend = self.trend_loss(x, x_hat)
        fidelity = self.normalize(fidelity, "fidelity")
        pattern = self.normalize(pattern, "pattern")
        trend = self.normalize(trend, "trend")
        return self.lambda_fidelity * fidelity + self.lambda_pattern * pattern + self.lambda_trend * trend

    # Ramene chaque composante sur une echelle comparable avant la somme ponderee.
    def normalize(self, value, key):
        low = self.stats[f"{key}_min"]
        high = self.stats[f"{key}_max"]
        return torch.clamp((value - low) / max(high - low, 1e-12), 0.0, 10.0)

    @staticmethod
    # Penalise les differences de direction entre les variations du signal original et reconstruit.
    def pattern_loss(x, x_hat):
        grad_x = torch.diff(x, dim=1)
        grad_hat = torch.diff(x_hat, dim=1)
        mismatch = torch.abs(torch.sign(grad_x) - torch.sign(grad_hat))
        return torch.mean(mismatch)

    @staticmethod
    # Compare les pentes locales pour conserver les tendances lentes du signal EEG.
    def trend_loss(x, x_hat, window_size=20, step_size=10):
        device = x.device
        t = torch.arange(window_size, device=device, dtype=x.dtype)
        t = t - torch.mean(t)
        denominator = torch.sum(t ** 2)
        losses = []
        for start in range(0, x.size(1) - window_size + 1, step_size):
            w = x[:, start:start + window_size, :]
            r = x_hat[:, start:start + window_size, :]
            w_centered = w - torch.mean(w, dim=1, keepdim=True)
            r_centered = r - torch.mean(r, dim=1, keepdim=True)
            slope_w = torch.sum(w_centered * t.view(1, -1, 1), dim=1) / denominator
            slope_r = torch.sum(r_centered * t.view(1, -1, 1), dim=1) / denominator
            losses.append(torch.mean(torch.abs(slope_w - slope_r)))
        return torch.stack(losses).mean()


# Variante d'ablation qui garde MSE et la contrainte de forme.
class MSEPatternLoss(ClinicalPatternLoss):
    # Additionne l'erreur de reconstruction et l'erreur de direction.
    def forward(self, x, x_hat):
        return torch.mean((x - x_hat) ** 2) + self.pattern_loss(x, x_hat)


# Variante d'ablation qui garde MSE et la contrainte de tendance.
class MSETrendLoss(ClinicalPatternLoss):
    # Additionne l'erreur de reconstruction et l'erreur de pente locale.
    def forward(self, x, x_hat):
        return torch.mean((x - x_hat) ** 2) + self.trend_loss(x, x_hat)
