import torch
import torch.nn as nn

mse = nn.MSELoss()

# ---------- fidelity ----------
def fidelity_loss(x, x_hat):
    return torch.mean((x - x_hat) ** 2)

# ---------- pattern ----------
def pattern_loss(x, x_hat):
    dx = torch.diff(x, dim=1)
    dx_hat = torch.diff(x_hat, dim=1)
    return torch.mean(torch.abs(dx - dx_hat))

# ---------- trend ----------
def trend_loss(x, x_hat):
    cx = torch.cumsum(x, dim=1)
    cx_hat = torch.cumsum(x_hat, dim=1)
    return torch.mean(torch.abs(cx - cx_hat))


# ---------- FULL CP LOSS ----------
def cp_loss(x, x_hat):
    lf = fidelity_loss(x, x_hat)
    lp = pattern_loss(x, x_hat)
    lt = trend_loss(x, x_hat)

    # normalization (simplified version for prototype)
    return 0.3 * lf + 0.3 * lp + 0.4 * lt