"""
UCI HAR reconstruction loss
===========================================================================
Clinical Pattern Loss (CPLoss) — Algorithm 2, final combination.

Implements:
    1. TrendLoss     — Algorithm 2, Equation 14 (long-range temporal trend)
    2. CPLoss        — Combines all three loss components with λ1=0.3, λ2=0.3, λ3=0.4
    3. compute_trend_stats()  — precomputes min/max for normalization
    4. build_cploss()         — factory: runs all three stats passes, returns CPLoss

Final formula (Algorithm 2, Combination step):
    L_CP = 0.3 * L_fidelity_norm + 0.3 * L_pattern_norm + 0.4 * L_trend_norm

Depends on:
    mse_loss.py     — FidelityLoss, compute_fidelity_stats
    pattern_loss.py — PatternLoss,  compute_pattern_stats

Usage (inside train.py, training script):
    from loss import build_cploss
    criterion = build_cploss(train_loader, model, device)
    loss = criterion(x_hat, x)
"""

import torch
import torch.nn as nn

try:
    from mse_loss import FidelityLoss, compute_fidelity_stats
    from pattern_loss import PatternLoss, compute_pattern_stats
except ImportError:
    from .mse_loss import FidelityLoss, compute_fidelity_stats
    from .pattern_loss import PatternLoss, compute_pattern_stats


# =============================================================================
# TrendLoss — Algorithm 2, Equation 14
# =============================================================================

class TrendLoss(nn.Module):
    """
    Trend Loss: captures long-range temporal dynamics (Algorithm 2, Equation 14).

    Divides each window into S non-overlapping segments, computes the mean of
    each segment (coarse moving average), then measures the MSE between the
    trend envelopes of the original and reconstructed signals.

        segments_x[s]     = mean(x[s*K : (s+1)*K, :])   for s = 0..S-1
        L_trend_per_window = (1/S) * Σ_s ||segments_x[s] - segments_x_hat[s]||²

    With the default S=8 over 128-sample windows (50 Hz), each segment spans
    16 samples = 0.32 s, giving an 8-point macro-shape representation at ~3 Hz.

    This complements:
      • FidelityLoss — exact point-wise reconstruction accuracy
      • PatternLoss  — step-by-step direction (first finite differences)
      • TrendLoss    — coarse trajectory, inter-segment envelope shape

    Reduction modes (same convention as other loss modules):
        'mean'     → scalar — average over the batch. For backpropagation.
        'sequence' → (B,)   — per-window trend loss. For CPLoss normalization,
                              anomaly thresholding (evaluation).
        'none'     → (B, S, F) — raw per-segment squared differences.

    Args:
        num_segments (int): Number of equal-length segments S. Must divide
            the sequence length evenly, or the tail is discarded.
            Default: 8  (16 samples/segment for L=128).
        reduction (str): 'mean' | 'sequence' | 'none'.
    """

    def __init__(self, num_segments: int = 8, reduction: str = 'mean'):
        super().__init__()
        if reduction not in ('mean', 'sequence', 'none'):
            raise ValueError("reduction must be one of ['mean', 'sequence', 'none']")
        self.num_segments = num_segments
        self.reduction = reduction

    def _segment_means(self, x: torch.Tensor) -> torch.Tensor:
        """
        Compute the mean of each non-overlapping segment.

        Args:
            x: (B, L, F)
        Returns:
            segments: (B, S, F)  where S = num_segments
        """
        B, L, F = x.shape
        S = self.num_segments
        seg_len = L // S
        # Discard tail samples if L is not divisible by S
        x_trimmed = x[:, : seg_len * S, :]          # (B, seg_len*S, F)
        x_reshaped = x_trimmed.view(B, S, seg_len, F)
        return x_reshaped.mean(dim=2)                # (B, S, F)

    def forward(self, x: torch.Tensor, x_hat: torch.Tensor) -> torch.Tensor:
        """
        Compute the Trend Loss.

        Args:
            x     (torch.Tensor): Original sequence.        Shape (B, L, F).
            x_hat (torch.Tensor): Reconstructed sequence.   Shape (B, L, F).

        Returns:
            torch.Tensor:
                reduction='mean'     → scalar
                reduction='sequence' → (B,)
                reduction='none'     → (B, S, F)
        """
        if x.shape != x_hat.shape:
            raise ValueError(
                f"Shape mismatch: x={x.shape}, x_hat={x_hat.shape}"
            )
        if x.dim() != 3:
            raise ValueError(
                f"Expected 3-D input (B, L, F), got {x.dim()}-D tensor of shape {x.shape}. "
                "Wrap a single sequence in a batch dimension: x.unsqueeze(0)."
            )

        trend_x     = self._segment_means(x)       # (B, S, F)
        trend_x_hat = self._segment_means(x_hat)   # (B, S, F)

        sq_diff = (trend_x - trend_x_hat) ** 2     # (B, S, F)

        if self.reduction == 'none':
            return sq_diff                          # (B, S, F)

        # Average over features → (B, S), then over segments → (B,)
        per_window = sq_diff.mean(dim=2).mean(dim=1)   # (B,)

        if self.reduction == 'sequence':
            return per_window   # (B,)

        return per_window.mean()   # scalar


# =============================================================================
# compute_trend_stats
# =============================================================================

@torch.no_grad()
def compute_trend_stats(
    dataloader,
    model,
    device: str = 'cpu',
    num_segments: int = 8,
) -> dict:
    """
    Precomputes min/max Trend Loss on the healthy training set.

    Required before constructing CPLoss so that L_trend can be normalised to [0, 1]:
        L_trend_norm = (L_trend - trend_min) / (trend_max - trend_min)

    Args:
        dataloader   : DataLoader over the healthy training windows.
                       Each batch must be a tensor or a (tensor, ...) tuple/list.
        model        : TAAE model — returns (x_hat, alpha, z).
        device       : 'cpu' | 'cuda' | 'mps'.
        num_segments : Number of segments S (must match the value used in CPLoss).

    Returns:
        dict with keys:
            'trend_min' (float)
            'trend_max' (float)
    """
    model.eval()
    loss_fn = TrendLoss(num_segments=num_segments, reduction='sequence')
    all_losses: list = []

    for batch in dataloader:
        x = batch[0] if isinstance(batch, (list, tuple)) else batch
        x = x.to(device)

        outputs = model(x)
        x_hat   = outputs[0] if isinstance(outputs, (list, tuple)) else outputs

        losses = loss_fn(x, x_hat)          # (B,)
        all_losses.extend(losses.cpu().tolist())

    if not all_losses:
        raise ValueError("Dataloader yielded no samples — cannot compute trend statistics.")

    return {
        'trend_min': float(min(all_losses)),
        'trend_max': float(max(all_losses)),
    }


# =============================================================================
# CPLoss — Algorithm 2, final combination
# =============================================================================

class CPLoss(nn.Module):
    """
    Clinical Pattern Loss (CPLoss) — Algorithm 2, combination step.

    Combines three independently-normalised loss components:

        L_CP = λ1 * L_fidelity_norm + λ2 * L_pattern_norm + λ3 * L_trend_norm
             = 0.3 * L_fidelity_norm + 0.3 * L_pattern_norm + 0.4 * L_trend_norm

    Each component is clamped to [0, 1] after normalisation (handles small
    out-of-range values at inference on anomalous windows gracefully).

    Construct via the factory:
        criterion = build_cploss(train_loader, model, device)

    Or manually when stats are already known:
        criterion = CPLoss(
            fidelity_min=..., fidelity_max=...,
            pattern_min=...,  pattern_max=...,
            trend_min=...,    trend_max=...,
        )

    Args:
        fidelity_min / fidelity_max : training-set bounds from compute_fidelity_stats()
        pattern_min  / pattern_max  : training-set bounds from compute_pattern_stats()
        trend_min    / trend_max    : training-set bounds from compute_trend_stats()
        lambda1 (float): weight for L_fidelity (default 0.3)
        lambda2 (float): weight for L_pattern  (default 0.3)
        lambda3 (float): weight for L_trend    (default 0.4)
        num_segments (int): segments S for TrendLoss (default 8)
        reduction (str): 'mean' (scalar, for backprop) | 'sequence' ((B,), for eval)
        eps (float): small constant to avoid division by zero in normalisation
    """

    def __init__(
        self,
        fidelity_min: float,
        fidelity_max: float,
        pattern_min:  float,
        pattern_max:  float,
        trend_min:    float,
        trend_max:    float,
        lambda1:      float = 0.3,
        lambda2:      float = 0.3,
        lambda3:      float = 0.4,
        num_segments: int   = 8,
        reduction:    str   = 'mean',
        eps:          float = 1e-8,
    ):
        super().__init__()

        if reduction not in ('mean', 'sequence'):
            raise ValueError("reduction must be 'mean' or 'sequence'")
        if abs(lambda1 + lambda2 + lambda3 - 1.0) > 1e-6:
            raise ValueError(
                f"Weights must sum to 1.0, got λ1={lambda1}, λ2={lambda2}, λ3={lambda3} "
                f"(sum={lambda1 + lambda2 + lambda3:.6f})"
            )

        self.lambda1   = lambda1
        self.lambda2   = lambda2
        self.lambda3   = lambda3
        self.reduction = reduction
        self.eps       = eps

        # Sub-modules — use sequence mode to normalise per-window before combining
        self.fidelity_fn = FidelityLoss(reduction='sequence')
        self.pattern_fn  = PatternLoss(reduction='sequence')
        self.trend_fn    = TrendLoss(num_segments=num_segments, reduction='sequence')

        # Register normalization bounds as non-trainable buffers so they move
        # with the module when .to(device) / .cuda() / .cpu() is called.
        self.register_buffer('fidelity_min', torch.tensor(fidelity_min, dtype=torch.float32))
        self.register_buffer('fidelity_max', torch.tensor(fidelity_max, dtype=torch.float32))
        self.register_buffer('pattern_min',  torch.tensor(pattern_min,  dtype=torch.float32))
        self.register_buffer('pattern_max',  torch.tensor(pattern_max,  dtype=torch.float32))
        self.register_buffer('trend_min',    torch.tensor(trend_min,    dtype=torch.float32))
        self.register_buffer('trend_max',    torch.tensor(trend_max,    dtype=torch.float32))

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    def _normalize(
        self,
        x:    torch.Tensor,
        xmin: torch.Tensor,
        xmax: torch.Tensor,
    ) -> torch.Tensor:
        """Min-max normalisation clamped to [0, 1]."""
        norm = (x - xmin) / (xmax - xmin + self.eps)
        return norm.clamp(0.0, 1.0)

    # ------------------------------------------------------------------
    # Forward
    # ------------------------------------------------------------------

    def forward(self, x: torch.Tensor, x_hat: torch.Tensor) -> torch.Tensor:
        """
        Compute the CPLoss for a batch.

        Args:
            x     (torch.Tensor): Original windows.       Shape (B, L, F).
            x_hat (torch.Tensor): Reconstructed windows.  Shape (B, L, F).

        Returns:
            torch.Tensor:
                reduction='mean'     → scalar   (use for training / backprop)
                reduction='sequence' → (B,)     (use for eval / anomaly scoring)
        """
        # ── Step 1: compute per-sequence component losses ───────────────────
        L_fidelity = self.fidelity_fn(x, x_hat)   # (B,)
        L_pattern  = self.pattern_fn(x, x_hat)    # (B,)
        L_trend    = self.trend_fn(x, x_hat)      # (B,)

        # ── Step 2: normalise each component to [0, 1] ──────────────────────
        L_fidelity_norm = self._normalize(L_fidelity, self.fidelity_min, self.fidelity_max)
        L_pattern_norm  = self._normalize(L_pattern,  self.pattern_min,  self.pattern_max)
        L_trend_norm    = self._normalize(L_trend,    self.trend_min,    self.trend_max)

        # ── Step 3: weighted combination ────────────────────────────────────
        L_CP = (
            self.lambda1 * L_fidelity_norm   # 0.3 × fidelity  ()
          + self.lambda2 * L_pattern_norm    # 0.3 × pattern   ()
          + self.lambda3 * L_trend_norm      # 0.4 × trend     ()
        )                                   # (B,) ∈ [0, 1]

        if self.reduction == 'sequence':
            return L_CP        # (B,) — per-window anomaly score

        return L_CP.mean()     # scalar — for backpropagation

    # ------------------------------------------------------------------
    # Utility
    # ------------------------------------------------------------------

    def get_stats(self) -> dict:
        """Returns the normalisation bounds currently stored in the module."""
        return {
            'fidelity_min': self.fidelity_min.item(),
            'fidelity_max': self.fidelity_max.item(),
            'pattern_min':  self.pattern_min.item(),
            'pattern_max':  self.pattern_max.item(),
            'trend_min':    self.trend_min.item(),
            'trend_max':    self.trend_max.item(),
        }

    def __repr__(self) -> str:
        stats = self.get_stats()
        return (
            f"CPLoss("
            f"λ1={self.lambda1}, λ2={self.lambda2}, λ3={self.lambda3} | "
            f"fidelity=[{stats['fidelity_min']:.4f}, {stats['fidelity_max']:.4f}] "
            f"pattern=[{stats['pattern_min']:.4f}, {stats['pattern_max']:.4f}] "
            f"trend=[{stats['trend_min']:.4f}, {stats['trend_max']:.4f}])"
        )


# =============================================================================
# build_cploss — factory function (entry point for training script)
# =============================================================================

def build_cploss(
    train_loader,
    model,
    device:       str   = 'cpu',
    lambda1:      float = 0.3,
    lambda2:      float = 0.3,
    lambda3:      float = 0.4,
    num_segments: int   = 8,
    reduction:    str   = 'mean',
    eps:          float = 1e-8,
) -> CPLoss:
    """
    Factory function: runs three stats-collection passes over the healthy
    training set, then constructs and returns a ready-to-use CPLoss.

    This is the recommended entry point for training script.

    Example (train.py):
        from loss import build_cploss
        criterion = build_cploss(train_loader, model, device=device)
        # ... inside training loop:
        x_hat, alpha, z = model(x)
        loss = criterion(x, x_hat)
        loss.backward()

    Args:
        train_loader  : DataLoader over the *healthy* training set only.
        model         : TAAE model. Must already be on `device`.
        device        : 'cpu' | 'cuda' | 'mps'.
        lambda1..3    : Component weights (must sum to 1.0).
        num_segments  : Segments S for TrendLoss (default 8).
        reduction     : 'mean' for training, 'sequence' for eval.
        eps           : Division guard in normalisation.

    Returns:
        CPLoss instance with all normalisation buffers set and on `device`.
    """
    print("[build_cploss] Computing fidelity stats...")
    fidelity_stats = compute_fidelity_stats(train_loader, model, device)

    print("[build_cploss] Computing pattern stats...")
    pattern_stats  = compute_pattern_stats(train_loader, model, device)

    print("[build_cploss] Computing trend stats...")
    trend_stats    = compute_trend_stats(train_loader, model, device, num_segments=num_segments)

    criterion = CPLoss(
        fidelity_min = fidelity_stats['fidelity_min'],
        fidelity_max = fidelity_stats['fidelity_max'],
        pattern_min  = pattern_stats['pattern_min'],
        pattern_max  = pattern_stats['pattern_max'],
        trend_min    = trend_stats['trend_min'],
        trend_max    = trend_stats['trend_max'],
        lambda1      = lambda1,
        lambda2      = lambda2,
        lambda3      = lambda3,
        num_segments = num_segments,
        reduction    = reduction,
        eps          = eps,
    ).to(device)

    print(f"[build_cploss] Done. {criterion}")
    return criterion
