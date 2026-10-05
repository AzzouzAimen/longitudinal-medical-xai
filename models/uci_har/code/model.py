# =============================================================================
# UCI HAR temporal attention autoencoder
# =============================================================================
# Assemble l'encodeur () et le décodeur () en un seul modèle
# TAAE complet : Temporal Attention AutoEncoder
#
# Modification vs version originale :
#   Le vecteur de contexte c (batch, 64) produit par l'encodeur est maintenant
#   transmis au décodeur et concaténé à z avant la projection.
#   Cela permet au décodeur d'exploiter l'information temporelle complète
#   capturée par l'attention, et pas seulement le bottleneck z.
#
# Architecture complète :
#   X (batch, L, f)
#   ──────────────────────── ENCODER ─────────────────────────
#   → BiLSTM1 (h=64) → Dropout(0.2)
#   → BiLSTM2 (h=32) → Dropout(0.2)
#   → Temporal Attention → c (batch, 64), alpha (batch, L)
#   → Linear + LayerNorm → z (batch, 16)
#   ──────────────────────── DECODER ─────────────────────────
#   → concat(z, c)   → (batch, 80)
#   → Linear + ReLU  → z_proj (batch, 64)
#   → Repeat(L)      → (batch, L, 64)
#   → BiLSTM3 (h=32) → Dropout(0.2)
#   → BiLSTM4 (h=64) → Dropout(0.2)
#   → Linear + Sigmoid → X_hat (batch, L, f)
#
# Outputs :
#   X_hat : (batch, L, f)   — signal reconstruit
#   alpha : (batch, L)      — poids d'attention temporelle
#   z     : (batch, 16)     — représentation latente
# =============================================================================

import torch
import torch.nn as nn

from encoder import TAEEncoder
from decoder import TAEDecoder


class TAAE(nn.Module):
    """
    Temporal Attention AutoEncoder (TAAE) — modèle complet.

    Combine TAEEncoder () et TAEDecoder ().
    Le vecteur de contexte c est transmis du décodeur à l'encodeur
    pour une reconstruction plus fidèle.

    Usage :
        model = TAAE()
        x_hat, alpha, z = model(x)
        loss = criterion(x_hat, x)
    """

    def __init__(
        self,
        input_features: int = 9,
        hidden1: int        = 64,
        hidden2: int        = 32,
        hidden3: int        = 32,
        hidden4: int        = 64,
        d_latent: int       = 16,
        d_attn: int         = 32,
        dropout: float      = 0.2,
        seq_len: int        = 128,
    ):
        """
        Args:
            input_features : nombre de canaux (f = 9 pour HAR)
            hidden1        : hidden size BiLSTM couche 1 (= 64)
            hidden2        : hidden size BiLSTM couche 2 (= 32)
            hidden3        : hidden size BiLSTM couche 3 (= 32)
            hidden4        : hidden size BiLSTM couche 4 (= 64)
            d_latent       : dimension du vecteur latent (= 16)
            d_attn         : dimension de l'espace d'attention (= 32)
            dropout        : taux de dropout (= 0.2)
            seq_len        : longueur de la séquence L (= 128)
        """
        super().__init__()

        self.encoder = TAEEncoder(
            input_features=input_features,
            hidden1=hidden1,
            hidden2=hidden2,
            d_latent=d_latent,
            d_attn=d_attn,
            dropout=dropout,
        )

        self.decoder = TAEDecoder(
            d_latent=d_latent,
            hidden3=hidden3,
            hidden4=hidden4,
            output_features=input_features,
            dropout=dropout,
            seq_len=seq_len,
            d_context=hidden2 * 2,   # 64 = 2 * hidden2 — dimension de c
        )

    def forward(self, x: torch.Tensor):
        """
        Forward pass complet du TAAE.

        Args:
            x : (batch, L, f) — fenêtre de signal d'entrée

        Returns:
            x_hat : (batch, L, f) — signal reconstruit
            alpha : (batch, L)    — poids d'attention temporelle
            z     : (batch, 16)   — vecteur latent
        """
        # ── Encodeur ──────────────────────────────────────────────────────────
        # z     : (batch, 16)
        # alpha : (batch, L)
        # c     : (batch, 64)
        z, alpha, c = self.encoder(x)

        # ── Décodeur ──────────────────────────────────────────────────────────
        # c est maintenant transmis au décodeur
        x_hat = self.decoder(z, c)

        return x_hat, alpha, z

    def encode(self, x: torch.Tensor):
        """
        Encodage seul (utile à l'inférence / visualisation de l'espace latent).

        Returns:
            z     : (batch, 16)
            alpha : (batch, L)
            c     : (batch, 64)
        """
        z, alpha, c = self.encoder(x)
        return z, alpha, c

    def reconstruct(self, x: torch.Tensor):
        """
        Reconstruction complète + calcul de la loss de reconstruction par fenêtre.

        Returns:
            x_hat      : (batch, L, f)
            recon_loss : (batch,)   — MSE par fenêtre (pour le scoring d'anomalie)
            alpha      : (batch, L)
        """
        x_hat, alpha, _ = self.forward(x)
        recon_loss = ((x - x_hat) ** 2).mean(dim=(1, 2))
        return x_hat, recon_loss, alpha


# =============================================================================
# TEST RAPIDE
# =============================================================================

if __name__ == "__main__":
    print("=" * 60)
    print("TEST : TAAE (model.py) — avec vecteur de contexte c")
    print("=" * 60)

    BATCH = 80
    L     = 128
    F     = 9

    model = TAAE(
        input_features=F,
        hidden1=64,
        hidden2=32,
        hidden3=32,
        hidden4=64,
        d_latent=16,
        d_attn=32,
        dropout=0.2,
        seq_len=L,
    )

    x = torch.randn(BATCH, L, F)
    print(f"\nInput X shape : {x.shape}  (batch, L, features)")

    model.eval()
    with torch.no_grad():
        x_hat, alpha, z = model(x)

    print(f"X_hat shape   : {x_hat.shape}  — attendu: ({BATCH}, {L}, {F})")
    print(f"alpha shape   : {alpha.shape}  — attendu: ({BATCH}, {L})")
    print(f"z shape       : {z.shape}      — attendu: ({BATCH}, 16)")

    # Vérifications dimensions
    assert x_hat.shape == (BATCH, L, F), f"x_hat shape: {x_hat.shape}"
    assert alpha.shape == (BATCH, L),    f"alpha shape: {alpha.shape}"
    assert z.shape     == (BATCH, 16),   f"z shape: {z.shape}"

    # Vérification Sigmoid
    assert x_hat.min() >= 0.0 and x_hat.max() <= 1.0, "Sigmoid hors [0,1]"

    # Vérification attention
    alpha_sum = alpha.sum(dim=-1)
    assert torch.allclose(alpha_sum, torch.ones(BATCH), atol=1e-5), \
        "attention weights ne somment pas à 1"

    # Test reconstruct()
    with torch.no_grad():
        x_hat2, recon_loss, alpha2 = model.reconstruct(x)
    assert recon_loss.shape == (BATCH,), f"recon_loss shape: {recon_loss.shape}"
    print(f"\nrecon_loss shape : {recon_loss.shape}  — attendu: ({BATCH},)")
    print(f"recon_loss mean  : {recon_loss.mean():.6f}")

    # Test encode()
    with torch.no_grad():
        z2, alpha3, c2 = model.encode(x)
    assert z2.shape == (BATCH, 16),  f"z2 shape: {z2.shape}"
    assert c2.shape == (BATCH, 64),  f"c2 shape: {c2.shape}"
    print(f"encode() → z: {z2.shape}, c: {c2.shape}")

    # Paramètres totaux
    total = sum(p.numel() for p in model.parameters())
    enc_p = sum(p.numel() for p in model.encoder.parameters())
    dec_p = sum(p.numel() for p in model.decoder.parameters())
    print(f"\nParamètres encodeur : {enc_p:,}")
    print(f"Paramètres décodeur : {dec_p:,}")
    print(f"Paramètres totaux   : {total:,}")

    print("\n✅ Toutes les vérifications passées — model.py prêt")
