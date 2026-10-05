import torch
from torch import nn


class SupervisedBiLSTMClassifier(nn.Module):
    def __init__(self, input_dim=1, hidden_dim=64, dropout=0.3, n_classes=2):
        super().__init__()
        self.lstm1 = nn.LSTM(input_dim, hidden_dim, batch_first=True, bidirectional=True)
        self.dropout1 = nn.Dropout(dropout)
        self.lstm2 = nn.LSTM(hidden_dim * 2, hidden_dim, batch_first=True, bidirectional=True)
        self.dropout2 = nn.Dropout(dropout)
        self.output = nn.Linear(hidden_dim * 2, n_classes)

    def forward(self, x):
        h, _ = self.lstm1(x)
        h = self.dropout1(h)
        _, (hn, _) = self.lstm2(h)
        h_last = torch.cat([hn[-2], hn[-1]], dim=-1)
        h_last = self.dropout2(h_last)
        return self.output(h_last)
