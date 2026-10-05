import torch
from torch import nn


# Modele autoencodeur temporel avec attention pour reconstruire les fenetres EEG.
class TemporalAttentionAutoencoder(nn.Module):
    def __init__(self, input_dim=1, latent_dim=32, attention_dim=64, dropout=0.0):
        super().__init__()
        self.encoder1 = nn.LSTM(input_dim, 64, batch_first=True, bidirectional=True)
        self.dropout1 = nn.Dropout(dropout)
        self.encoder2 = nn.LSTM(128, 32, batch_first=True, bidirectional=True)
        self.dropout2 = nn.Dropout(dropout)
        self.attention = nn.Sequential(
            nn.Linear(64, attention_dim),
            nn.Tanh(),
            nn.Linear(attention_dim, 1, bias=False),
        )
        self.latent = nn.Linear(64, latent_dim)
        self.latent_norm = nn.LayerNorm(latent_dim)
        self.project = nn.Sequential(nn.Linear(64 + latent_dim, 64), nn.ReLU())
        self.decoder1 = nn.LSTM(64, 32, batch_first=True, bidirectional=True)
        self.dropout3 = nn.Dropout(dropout)
        self.decoder2 = nn.LSTM(64, 64, batch_first=True, bidirectional=True)
        self.dropout4 = nn.Dropout(dropout)
        self.output = nn.Linear(128, input_dim)
        self.reset_parameters()

    # Initialise les poids recurrentiels et lineaires de facon stable.
    def reset_parameters(self):
        for name, parameter in self.named_parameters():
            if parameter.dim() > 1:
                nn.init.xavier_uniform_(parameter)
            elif "bias" in name:
                nn.init.zeros_(parameter)

    # Calcule la reconstruction et les poids d'attention pour chaque pas temporel.
    def forward(self, x):
        h1, _ = self.encoder1(x)
        h1 = self.dropout1(h1)
        h2, _ = self.encoder2(h1)
        h2 = self.dropout2(h2)

        # L'attention resume les instants les plus informatifs dans un vecteur latent.
        energy = self.attention(h2).squeeze(-1)
        alpha = torch.softmax(energy, dim=1)
        context = torch.sum(h2 * alpha.unsqueeze(-1), dim=1)
        z = self.latent_norm(self.latent(context))

        # Le decodeur reconstruit la sequence complete a partir du latent et des etats temporels.
        z_repeated = z.unsqueeze(1).repeat(1, x.size(1), 1)
        decoder_input = self.project(torch.cat([h2, z_repeated], dim=-1))
        h3, _ = self.decoder1(decoder_input)
        h3 = self.dropout3(h3)
        h4, _ = self.decoder2(h3)
        h4 = self.dropout4(h4)
        decoded = torch.sigmoid(self.output(h4))

        # La connexion residuelle garde la forme saine du signal et laisse le decodeur corriger l'ecart.
        x_hat = 0.75 * x + 0.25 * decoded
        return x_hat, alpha
