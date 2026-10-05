"""
loss.py
 — Loss Functions
Topic M6 : Synthetic Thermal Time-Series
Implements Algorithm 2 (algorithm2_cploss.txt) exactly, plus the 3 ablation
variants required by Table IV.

Classes
-------
CPLoss          Full Clinical Pattern Loss (Equation 12)
                L_CP = λ_fidelity * L_fid_norm
                     + λ_pattern  * L_pat_norm
                     + λ_trend    * L_trend_norm

MSEOnlyLoss     Baseline: plain MSE, no normalisation
MSEPatternLoss  MSE + Pattern (equal weights 0.5 / 0.5)
MSETrendLoss    MSE + Trend   (equal weights 0.5 / 0.5)

All losses accept an optional `mask` tensor (B, T) that is True where a
sample is interpolated — those positions are down-weighted by 0.5.

Usage
-----
    from loss import CPLoss
    criterion = CPLoss()
    loss = criterion(x, x_hat, mask=interp_mask)
"""

from __future__ import annotations
import torch
import torch.nn as nn


# ---------------------------------------------------------------------------
# Helper : apply interpolation mask
# ---------------------------------------------------------------------------
def _apply_mask(error: torch.Tensor, mask: torch.Tensor | None) -> torch.Tensor:
    """
    error : (B, C, T)   point-wise squared (or absolute) error
    mask  : (B, T) bool — True where sample is interpolated
    Returns weighted error with same shape.
    """
    if mask is None:
        return error
    # Expand mask to (B, 1, T) so it broadcasts over channels
    w = torch.ones_like(mask, dtype=error.dtype)
    w[mask] = 0.5                          # down-weight interpolated points
    return error * w.unsqueeze(1)          # (B, C, T)


# ---------------------------------------------------------------------------
# 1. Fidelity Loss  (Equation 15)
# ---------------------------------------------------------------------------
def _fidelity_loss(
    x: torch.Tensor,
    x_hat: torch.Tensor,
    mask: torch.Tensor | None = None,
) -> torch.Tensor:
    """
    L_fidelity = mean over (T, C) of (x - x_hat)^2
    Shape: x, x_hat = (B, C, T)
    Returns scalar.
    """
    sq_err = (x - x_hat) ** 2             # (B, C, T)
    sq_err = _apply_mask(sq_err, mask)
    return sq_err.mean()


# ---------------------------------------------------------------------------
# 2. Pattern Loss  (Equation 13)
# ---------------------------------------------------------------------------
def _pattern_loss(
    x: torch.Tensor,
    x_hat: torch.Tensor,
    mask: torch.Tensor | None = None,
) -> torch.Tensor:
    """
    Counts direction mismatches between consecutive timesteps.
    L_pattern = mismatch_count / (T-1)

    x, x_hat : (B, C, T)
    Returns scalar.
    """
    # Temporal gradients (B, C, T-1)
    grad_x     = x[..., 1:] - x[..., :-1]
    grad_x_hat = x_hat[..., 1:] - x_hat[..., :-1]

    # Signs  (-1, 0, +1)
    sign_x     = torch.sign(grad_x)
    sign_x_hat = torch.sign(grad_x_hat)

    # Mismatch: |sign_x - sign_x_hat| / 2  gives 0 or 1
    mismatch = (sign_x - sign_x_hat).abs()   # values in {0, 1, 2}

    # Optional mask: apply to T-1 dimension
    if mask is not None:
        # mask is (B, T); take all but last timestep → (B, T-1)
        w = torch.ones(mask.shape[0], mask.shape[1] - 1,
                       dtype=mismatch.dtype, device=mismatch.device)
        w[mask[:, :-1]] = 0.5
        mismatch = mismatch * w.unsqueeze(1)

    T = x.shape[-1]
    return mismatch.mean() / 2.0           # normalise to [0,1]


# ---------------------------------------------------------------------------
# 3. Trend Loss  (Equation 14)
# ---------------------------------------------------------------------------
def _trend_loss(
    x: torch.Tensor,
    x_hat: torch.Tensor,
    trend_window: int = 20,
    step_size:    int = 10,
) -> torch.Tensor:
    """
    Computes linear slopes in sliding windows and penalises slope differences.

    x, x_hat : (B, C, T)
    Returns scalar.
    """
    B, C, T = x.shape
    W = trend_window

    # Time indices (shared across batches/channels)
    t_idx   = torch.arange(W, dtype=x.dtype, device=x.device)  # (W,)
    t_mean  = t_idx.mean()                                      # scalar
    t_dev   = t_idx - t_mean                                    # (W,)
    denom   = (t_dev ** 2).sum()                                # scalar

    slopes_x     = []
    slopes_x_hat = []

    starts = range(0, T - W + 1, step_size)
    for start in starts:
        seg_x     = x[..., start: start + W]        # (B, C, W)
        seg_x_hat = x_hat[..., start: start + W]    # (B, C, W)

        # Slope  m = sum((t - t_mean)(y - y_mean)) / sum((t - t_mean)^2)
        y_mean_x     = seg_x.mean(dim=-1, keepdim=True)         # (B, C, 1)
        y_mean_x_hat = seg_x_hat.mean(dim=-1, keepdim=True)

        num_x     = ((t_dev * (seg_x     - y_mean_x)).sum(dim=-1))     # (B,C)
        num_x_hat = ((t_dev * (seg_x_hat - y_mean_x_hat)).sum(dim=-1))

        slopes_x.append(num_x / denom)
        slopes_x_hat.append(num_x_hat / denom)

    if not slopes_x:
        return torch.tensor(0.0, device=x.device)

    slopes_x     = torch.stack(slopes_x, dim=-1)      # (B, C, n_windows)
    slopes_x_hat = torch.stack(slopes_x_hat, dim=-1)

    return (slopes_x - slopes_x_hat).abs().mean()


# ---------------------------------------------------------------------------
# 4. Full CPLoss  (Algorithm 2 – Equation 12)
# ---------------------------------------------------------------------------
class CPLoss(nn.Module):
    """
    Clinical Pattern Loss — full version.

    Parameters
    ----------
    lambda_fidelity : float  Weight for fidelity term.  Default 0.3
    lambda_pattern  : float  Weight for pattern  term.  Default 0.3
    lambda_trend    : float  Weight for trend    term.  Default 0.4
    pattern_min/max : float  Min-max from training set (Algorithm 2 defaults).
    trend_min/max   : float
    fidelity_min/max: float
    """

    def __init__(
        self,
        lambda_fidelity: float = 0.3,
        lambda_pattern:  float = 0.3,
        lambda_trend:    float = 0.4,
        # Normalisation constants from Algorithm 2
        pattern_min:     float = 0.0404,
        pattern_max:     float = 1.5859,
        trend_min:       float = 0.000018451,
        trend_max:       float = 0.00083290,
        fidelity_min:    float = 0.00000013287,
        fidelity_max:    float = 0.00015482,
    ):
        super().__init__()
        self.lf = lambda_fidelity
        self.lp = lambda_pattern
        self.lt = lambda_trend

        # Normalisation bounds
        self.pat_min  = pattern_min;   self.pat_max  = pattern_max
        self.tre_min  = trend_min;     self.tre_max  = trend_max
        self.fid_min  = fidelity_min;  self.fid_max  = fidelity_max

    # Utility: safe min-max normalisation (avoids divide-by-zero)
    @staticmethod
    def _norm(val: torch.Tensor, vmin: float, vmax: float) -> torch.Tensor:
        rng = vmax - vmin
        if rng < 1e-12:
            return val * 0.0
        return (val - vmin) / rng

    def forward(
        self,
        x:    torch.Tensor,
        x_hat: torch.Tensor,
        mask:  torch.Tensor | None = None,
    ) -> torch.Tensor:
        """
        x, x_hat : (B, C, T)
        mask      : (B, T) bool  — True = interpolated point
        Returns scalar loss.
        """
        L_fid  = _fidelity_loss(x, x_hat, mask)
        L_pat  = _pattern_loss(x, x_hat, mask)
        L_tre  = _trend_loss(x, x_hat)

        L_fid_n = self._norm(L_fid, self.fid_min, self.fid_max)
        L_pat_n = self._norm(L_pat, self.pat_min,  self.pat_max)
        L_tre_n = self._norm(L_tre, self.tre_min,  self.tre_max)

        return self.lf * L_fid_n + self.lp * L_pat_n + self.lt * L_tre_n


# ---------------------------------------------------------------------------
# 5. Ablation variants (Table IV)
# ---------------------------------------------------------------------------

class MSEOnlyLoss(nn.Module):
    """Ablation variant 1 — plain MSE, no clinical components."""

    def forward(
        self,
        x:    torch.Tensor,
        x_hat: torch.Tensor,
        mask:  torch.Tensor | None = None,
    ) -> torch.Tensor:
        return _fidelity_loss(x, x_hat, mask)


class MSEPatternLoss(nn.Module):
    """
    Ablation variant 2 — MSE + Pattern Loss (equal weights 0.5 / 0.5).
    No normalisation applied (raw values combined).
    """

    def forward(
        self,
        x:    torch.Tensor,
        x_hat: torch.Tensor,
        mask:  torch.Tensor | None = None,
    ) -> torch.Tensor:
        return 0.5 * _fidelity_loss(x, x_hat, mask) \
             + 0.5 * _pattern_loss(x, x_hat, mask)


class MSETrendLoss(nn.Module):
    """
    Ablation variant 3 — MSE + Trend Loss (equal weights 0.5 / 0.5).
    No normalisation applied.
    """

    def forward(
        self,
        x:    torch.Tensor,
        x_hat: torch.Tensor,
        mask:  torch.Tensor | None = None,
    ) -> torch.Tensor:
        return 0.5 * _fidelity_loss(x, x_hat, mask) \
             + 0.5 * _trend_loss(x, x_hat)


# ---------------------------------------------------------------------------
# Factory — handy for train.py ablation loop
# ---------------------------------------------------------------------------
LOSS_VARIANTS = {
    "MSE only":    MSEOnlyLoss,
    "MSE+Pattern": MSEPatternLoss,
    "MSE+Trend":   MSETrendLoss,
    "CPLoss Full": CPLoss,
}


def get_loss(name: str) -> nn.Module:
    """Return a loss instance by ablation name."""
    if name not in LOSS_VARIANTS:
        raise ValueError(f"Unknown loss '{name}'. Choose from {list(LOSS_VARIANTS)}")
    return LOSS_VARIANTS[name]()


# ---------------------------------------------------------------------------
# Smoke test
# ---------------------------------------------------------------------------
if __name__ == "__main__":
    B, C, T = 4, 3, 60
    x     = torch.rand(B, C, T)
    x_hat = torch.rand(B, C, T)
    mask  = torch.zeros(B, T, dtype=torch.bool)
    mask[:, 10:15] = True   # pretend some points are interpolated

    for name, cls in LOSS_VARIANTS.items():
        loss_val = cls()(x, x_hat, mask)
        print(f"  {name:15s} → loss = {loss_val.item():.6f}")

    print("✓ loss.py smoke-test passed.")
