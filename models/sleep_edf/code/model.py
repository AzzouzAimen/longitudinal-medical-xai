import torch
import torch.nn as nn

class TemporalAttentionAutoencoder(nn.Module):
    def __init__(self, input_dim=1, hidden=64, latent=16):
        super().__init__()

        # ENCODER (BiLSTM 1)
        self.enc1 = nn.LSTM(input_dim, 64, batch_first=True, bidirectional=True)

        # ENCODER (BiLSTM 2)
        self.enc2 = nn.LSTM(128, 32, batch_first=True, bidirectional=True)

        # ATTENTION
        self.attn = nn.Linear(64, 1)

        # LATENT
        self.latent = nn.Linear(64, latent)
        self.latent_out = nn.Linear(latent, 64)

        # DECODER
        self.dec1 = nn.LSTM(64, 32, batch_first=True, bidirectional=True)
        self.dec2 = nn.LSTM(64, 64, batch_first=True, bidirectional=True)

        self.out = nn.Linear(128, input_dim)

    def forward(self, x):

        # -------- Encoder --------
        h1, _ = self.enc1(x)
        h2, _ = self.enc2(h1)

        # -------- Attention --------
        e = self.attn(h2)
        alpha = torch.softmax(e, dim=1)

        context = torch.sum(alpha * h2, dim=1)

        # -------- Latent --------
        z = self.latent(context)
        z = self.latent_out(z)

        z = z.unsqueeze(1).repeat(1, x.shape[1], 1)

        # -------- Decoder --------
        d1, _ = self.dec1(z)
        d2, _ = self.dec2(d1)

        recon = self.out(d2)

        return recon, alpha