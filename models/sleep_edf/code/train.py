import torch
from model import TemporalAttentionAutoencoder
from loss import cp_loss
import numpy as np

data = np.load("data/sleep_edf_processed.npy")
data = torch.tensor(data, dtype=torch.float32).unsqueeze(-1)

model = TemporalAttentionAutoencoder()
optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)

train_losses = []

for epoch in range(20):  # Compact reference training loop.
    epoch_loss = 0

    for i in range(0, len(data), 32):
        batch = data[i:i+32]

        recon, _ = model(batch)

        loss = cp_loss(batch, recon)

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        epoch_loss += loss.item()

    avg = epoch_loss / len(data)
    train_losses.append(avg)

    print(f"Epoch {epoch}: {avg}")

np.save("results/train_losses.npy", train_losses)
torch.save(model.state_dict(), "results/model.pth")
