import torch
import logging
import os
from thermal_tsdb_dataset import ThermalTSDBDataset, ThermalTSDBDataModule
from model import TAAE                          # ← TAAE, pas AnomalyTransformer
from attention_store import AttentionStore

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
log = logging.getLogger("A3")

# 1. Charger le modèle
model = TAAE(n_channels=3, window_size=60)      # ← TAAE
model_path = "results/best_model.pt"

if os.path.exists(model_path):
    try:
        model.load_state_dict(torch.load(model_path, map_location="cpu"))
        log.info(f"Loaded weights from {model_path}")
    except Exception as e:
        log.error(f"Failed to load weights: {e}. Using random weights.")
else:
    log.warning(f"Model file {model_path} not found. Using random weights.")

model.eval()

# 2. Connexion DB
store = AttentionStore()

# 3. Dataset test
dm = ThermalTSDBDataModule(batch_size=1)
train_ids, val_ids, test_ids = dm.split()
dataset = ThermalTSDBDataset(dsn=dm.dsn, patient_ids=test_ids)

# 4. Inférence + stockage
log.info(f"Running inference on {min(50, len(dataset))} windows...")
for i in range(min(50, len(dataset))):
    try:
        sample = dataset.get_sample_with_meta(i)
        x = sample["signal"].unsqueeze(0)          # (1, C, T)

        with torch.no_grad():
            _, alpha = model(x)                    # TAAE retourne (x_hat, alpha)
            weights = alpha[0].cpu().numpy()       # (T,)

        store.insert(
            window_id  = sample["window_id"],
            patient_id = sample["patient_id"],
            timestamps = sample["timestamps"],
            weights    = weights,
            epoch      = 0
        )

        if (i + 1) % 10 == 0:
            log.info(f"  Processed {i+1} windows...")

    except Exception as e:
        log.error(f"Error processing window {i}: {e}")
        continue

print(f"\nA3 DONE: Attention maps stored in DB")
print(f"Total rows in attention_maps: {store.count()}")