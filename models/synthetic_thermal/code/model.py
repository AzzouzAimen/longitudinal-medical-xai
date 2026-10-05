"""
model.py
 — TAAE Model Architecture
Topic M6 : Synthetic Thermal Time-Series
Temporal Attention AutoEncoder (TAAE)
Implemented directly from Algorithm 1 (algorithm1_taae.txt)

Architecture:
    Input (B, C, T)
        ↓
    ENCODER  : 2 stacked Bidirectional LSTMs  (hidden 64 → 32)
        ↓
    ATTENTION: Additive attention → context vector
        ↓
    BOTTLENECK: Linear → LayerNorm  (latent dim = 16)
        ↓
    DECODER  : 2 stacked Bidirectional LSTMs  (hidden 32 → 64)
        ↓
    OUTPUT   : Linear + Sigmoid  → X_hat (B, C, T)

Usage:
    model = TAAE(n_channels=3, window_size=60)
    x_hat, alpha = model(x)          # x : (B, C, T)
"""

import torch
import torch.nn as nn


class TAAE(nn.Module):
    """
    Temporal Attention AutoEncoder.

    Parameters
    ----------
    n_channels  : int   Number of input channels (C). Default 3.
    window_size : int   Number of timesteps (T). Default 60.
    d_latent    : int   Latent bottleneck dimension. Default 16.
    d_attn      : int   Attention projection dimension. Default 32.
    dropout     : float Dropout rate after each BiLSTM block. Default 0.2.
    """

    def __init__(
        self,
        n_channels:  int   = 3,
        window_size: int   = 60,
        d_latent:    int   = 16,
        d_attn:      int   = 32,
        dropout:     float = 0.2,
    ):
        super().__init__()
        self.n_channels  = n_channels
        self.window_size = window_size
        self.d_latent    = d_latent

        # ----------------------------------------------------------------
        # ENCODER
        # BiLSTM-1 : input_size=C  → output_size=64*2=128  (bidirectional)
        # BiLSTM-2 : input_size=128 → output_size=32*2=64
        # ----------------------------------------------------------------
        self.enc_lstm1 = nn.LSTM(
            input_size=n_channels, hidden_size=64,
            num_layers=1, batch_first=True, bidirectional=True
        )
        self.enc_drop1 = nn.Dropout(dropout)

        self.enc_lstm2 = nn.LSTM(
            input_size=128, hidden_size=32,
            num_layers=1, batch_first=True, bidirectional=True
        )
        self.enc_drop2 = nn.Dropout(dropout)

        # ----------------------------------------------------------------
        # ATTENTION  (additive / Bahdanau style)
        # h2[t] ∈ R^64  →  a[t] ∈ R^32  →  e[t] ∈ R
        # ----------------------------------------------------------------
        self.attn_W = nn.Linear(64, d_attn, bias=True)   # W_a
        self.attn_v = nn.Linear(d_attn, 1, bias=False)   # v^T

        # ----------------------------------------------------------------
        # LATENT BOTTLENECK
        # c ∈ R^64  →  z ∈ R^16  →  LayerNorm
        # ----------------------------------------------------------------
        self.bottleneck   = nn.Linear(64, d_latent)
        self.layer_norm   = nn.LayerNorm(d_latent)

        # ----------------------------------------------------------------
        # DECODER PROJECTION
        # z ∈ R^16  →  z_proj ∈ R^64  (then repeated T times)
        # ----------------------------------------------------------------
        self.dec_proj = nn.Linear(d_latent, 64)

        # ----------------------------------------------------------------
        # DECODER
        # BiLSTM-3 : input_size=64  → output_size=32*2=64
        # BiLSTM-4 : input_size=64  → output_size=64*2=128
        # ----------------------------------------------------------------
        self.dec_lstm3 = nn.LSTM(
            input_size=64, hidden_size=32,
            num_layers=1, batch_first=True, bidirectional=True
        )
        self.dec_drop3 = nn.Dropout(dropout)

        self.dec_lstm4 = nn.LSTM(
            input_size=64, hidden_size=64,
            num_layers=1, batch_first=True, bidirectional=True
        )
        self.dec_drop4 = nn.Dropout(dropout)

        # ----------------------------------------------------------------
        # OUTPUT LAYER
        # h4[t] ∈ R^128  →  x_hat[t] ∈ R^C
        # ----------------------------------------------------------------
        self.output_layer = nn.Linear(128, n_channels)

        # Weight initialisation (Xavier / Glorot as per Algorithm 3)
        self._init_weights()

    # ------------------------------------------------------------------
    # Weight initialisation
    # ------------------------------------------------------------------
    def _init_weights(self):
        for name, param in self.named_parameters():
            if "weight_ih" in name or "weight_hh" in name:
                nn.init.xavier_uniform_(param.data)
            elif "bias" in name:
                nn.init.zeros_(param.data)
            elif param.dim() >= 2:
                nn.init.xavier_uniform_(param.data)

    # ------------------------------------------------------------------
    # Forward pass  (Algorithm 1)
    # ------------------------------------------------------------------
    def forward(
        self,
        x: torch.Tensor,
        interp_mask: torch.Tensor | None = None,
    ):
        """
        Parameters
        ----------
        x           : (B, C, T)  Input signal batch.
        interp_mask : (B, T) bool tensor — True where sample is interpolated.
                      Passed through but NOT used inside the model itself;
                      the loss function uses it to down-weight those points.

        Returns
        -------
        x_hat  : (B, C, T)  Reconstructed signal.
        alpha  : (B, T)     Attention weights per timestep.
        """
        B, C, T = x.shape

        # Transpose to (B, T, C) for LSTM  (batch_first=True)
        h = x.permute(0, 2, 1)   # (B, T, C)

        # ---- ENCODER ------------------------------------------------
        h1, _ = self.enc_lstm1(h)          # (B, T, 128)
        h1    = self.enc_drop1(h1)

        h2, _ = self.enc_lstm2(h1)         # (B, T, 64)
        h2    = self.enc_drop2(h2)

        # ---- ATTENTION ----------------------------------------------
        # a[t] = tanh(W_a * h2[t] + b_a)   → (B, T, d_attn)
        a = torch.tanh(self.attn_W(h2))    # (B, T, 32)
        # e[t] = v^T * a[t]                → (B, T, 1) → (B, T)
        e = self.attn_v(a).squeeze(-1)     # (B, T)
        # alpha = softmax(e)
        alpha = torch.softmax(e, dim=-1)   # (B, T)

        # context vector c = sum(alpha[t] * h2[t])  → (B, 64)
        c = torch.bmm(alpha.unsqueeze(1), h2).squeeze(1)  # (B, 64)

        # ---- BOTTLENECK ---------------------------------------------
        z = self.bottleneck(c)             # (B, 16)
        z = self.layer_norm(z)             # (B, 16)

        # ---- DECODER ------------------------------------------------
        # Project and repeat T times
        z_proj = torch.relu(self.dec_proj(z))          # (B, 64)
        z_proj = z_proj.unsqueeze(1).repeat(1, T, 1)   # (B, T, 64)

        h3, _ = self.dec_lstm3(z_proj)     # (B, T, 64)
        h3    = self.dec_drop3(h3)

        h4, _ = self.dec_lstm4(h3)         # (B, T, 128)
        h4    = self.dec_drop4(h4)

        # ---- OUTPUT -------------------------------------------------
        out   = torch.sigmoid(self.output_layer(h4))  # (B, T, C)
        x_hat = out.permute(0, 2, 1)                  # (B, C, T)

        return x_hat, alpha


# ---------------------------------------------------------------------------
# Quick smoke-test
# ---------------------------------------------------------------------------
if __name__ == "__main__":
    B, C, T = 4, 3, 60
    model = TAAE(n_channels=C, window_size=T)

    total_params = sum(p.numel() for p in model.parameters())
    print(f"TAAE | params: {total_params:,}")

    x     = torch.randn(B, C, T)
    x_hat, alpha = model(x)

    print(f"Input  shape : {x.shape}")
    print(f"Output shape : {x_hat.shape}")
    print(f"Alpha  shape : {alpha.shape}")
    assert x_hat.shape == x.shape,      "x_hat shape mismatch"
    assert alpha.shape == (B, T),       "alpha shape mismatch"
    assert torch.allclose(alpha.sum(dim=-1), torch.ones(B), atol=1e-5), \
        "attention weights must sum to 1"

    print("✓ model.py smoke-test passed.")
