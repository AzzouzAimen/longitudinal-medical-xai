import torch
from torch import nn
import torch.nn.functional as F


class DiagonalS4Encoder(nn.Module):
    def __init__(self, input_dim=1, d_model=128, d_state=64, latent_dim=16, num_layers=4, dropout=0.2):
        super().__init__()
        self.input = nn.Linear(input_dim, d_model)
        self.norm = nn.LayerNorm(d_model)
        self.layers = nn.ModuleList([DiagonalS4Layer(d_model, d_state, dropout) for _ in range(num_layers)])
        self.attn = nn.Sequential(nn.Linear(d_model, 32), nn.Tanh(), nn.Linear(32, 1, bias=False))
        self.latent = nn.Linear(d_model, latent_dim)

    def forward(self, x):
        h = self.norm(self.input(x))
        for layer in self.layers:
            h = layer(h)
        scores = self.attn(h).squeeze(-1)
        alpha = torch.softmax(scores, dim=1)
        context = torch.sum(h * alpha.unsqueeze(-1), dim=1)
        z = self.latent(context)
        return z, alpha


class DiagonalS4Layer(nn.Module):
    def __init__(self, d_model, d_state, dropout):
        super().__init__()
        self.a_log = nn.Parameter(torch.randn(d_state))
        self.b = nn.Parameter(torch.randn(d_model, d_state) * 0.02)
        self.c = nn.Parameter(torch.randn(d_state, d_model) * 0.02)
        self.d = nn.Parameter(torch.randn(d_model) * 0.02)
        self.delta = nn.Parameter(torch.ones(d_model))
        self.dropout = nn.Dropout(dropout)
        self.output = nn.Linear(d_model, d_model)

    def forward(self, h):
        batch, length, d_model = h.shape
        a = -torch.exp(self.a_log)
        delta = F.softplus(self.delta).mean()
        a_bar = torch.exp(delta * a)
        b_bar = ((a_bar - 1.0) / a.clamp(max=-1e-6)).unsqueeze(0) * self.b
        state = h.new_zeros(batch, self.b.size(1))
        outputs = []
        for t in range(length):
            drive = torch.matmul(h[:, t, :], b_bar)
            state = a_bar * state + drive
            y = torch.matmul(state, self.c) + self.d * h[:, t, :]
            outputs.append(y)
        y = torch.stack(outputs, dim=1)
        return self.dropout(self.output(y)) + h
