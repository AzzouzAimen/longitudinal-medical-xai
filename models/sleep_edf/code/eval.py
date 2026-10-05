import torch
import numpy as np
import pandas as pd
from model import TemporalAttentionAutoencoder

data = np.load("data/sleep_edf_processed.npy")
data = torch.tensor(data, dtype=torch.float32)

model = TemporalAttentionAutoencoder()
model.load_state_dict(torch.load("results/model.pth"))
model.eval()

errors = []
attentions = []

with torch.no_grad():
    for i in range(len(data)):

        x = data[i:i+1].unsqueeze(-1)

        recon, alpha = model(x)

        err = torch.mean((x - recon) ** 2).item()

        errors.append(err)
        attentions.append(alpha.mean().item())

errors = np.array(errors)

# ---------- THRESHOLD ----------
tau = np.percentile(errors, 85)
pred = errors > tau

# ---------- SAVE ----------
np.save("results/errors.npy", errors)
np.save("results/pred.npy", pred)

print("Threshold:", tau)