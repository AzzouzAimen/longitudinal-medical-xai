from dataclasses import dataclass
from pathlib import Path
import numpy as np
import torch
from torch.utils.data import DataLoader, TensorDataset
from medattnaid.data import fetch_windows, split_healthy_windows
from medattnaid.losses import ClinicalPatternLoss, MSEPatternLoss, MSETrendLoss
from medattnaid.model import TemporalAttentionAutoencoder


# Parametres principaux utilises pour reproduire l'entrainement du modele.
@dataclass
class TrainConfig:
    subject: str = "chb01"
    channel: str = "ALL"
    window_length: int = 100
    stride: int = 100
    limit: int = 240000
    batch_size: int = 80
    learning_rate: float = 0.001
    weight_decay: float = 0.000001
    max_epochs: int = 80
    patience: int = 20
    output_dir: Path = Path("results/model")
    loss_variant: str = "mse"
    noise_std: float = 0.0
    seed: int = 42


# Entraine l'autoencodeur uniquement sur les fenetres saines puis sauvegarde le meilleur modele.
def train_model(args: TrainConfig):
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    batch = fetch_windows(args.subject, args.channel, args.window_length, args.stride, args.limit)
    train_idx, val_idx, _ = split_healthy_windows(batch)

    # Les donnees d'entrainement et de validation restent separees pour eviter le surapprentissage.
    train_loader = DataLoader(
        TensorDataset(torch.tensor(batch.windows[train_idx])),
        batch_size=args.batch_size,
        shuffle=True,
    )
    val_loader = DataLoader(
        TensorDataset(torch.tensor(batch.windows[val_idx])),
        batch_size=args.batch_size,
        shuffle=False,
    )
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = TemporalAttentionAutoencoder(input_dim=batch.windows.shape[-1]).to(device)

    # Le choix de la perte sert aussi aux etudes d'ablation demandees dans le rapport.
    if args.loss_variant == "mse":
        criterion = torch.nn.MSELoss()
    elif args.loss_variant == "mse_pattern":
        criterion = MSEPatternLoss()
    elif args.loss_variant == "mse_trend":
        criterion = MSETrendLoss()
    else:
        criterion = ClinicalPatternLoss()
    optimizer = torch.optim.AdamW(model.parameters(), lr=args.learning_rate, weight_decay=args.weight_decay)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, factor=0.5, patience=5)
    best_val = np.inf
    wait = 0
    history = []
    args.output_dir.mkdir(parents=True, exist_ok=True)
    for epoch in range(1, args.max_epochs + 1):
        model.train()
        train_losses = []

        # Boucle d'optimisation standard: reconstruction, perte, retropropagation, mise a jour.
        for (x,) in train_loader:
            x = x.to(device)
            x_in = torch.clamp(x + torch.randn_like(x) * args.noise_std, 0.0, 1.0) if args.noise_std > 0 else x
            optimizer.zero_grad()
            x_hat, _ = model(x_in)
            loss = criterion(x, x_hat)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()
            train_losses.append(float(loss.detach().cpu()))
        model.eval()
        val_losses = []

        # La validation sert a choisir le checkpoint conserve pour l'evaluation finale.
        with torch.no_grad():
            for (x,) in val_loader:
                x = x.to(device)
                x_hat, _ = model(x)
                val_losses.append(float(criterion(x, x_hat).cpu()))
        train_loss = float(np.mean(train_losses))
        val_loss = float(np.mean(val_losses))
        lr = optimizer.param_groups[0]["lr"]
        history.append((epoch, train_loss, val_loss, lr))
        scheduler.step(val_loss)
        if val_loss < best_val:
            best_val = val_loss
            wait = 0
            torch.save(model.state_dict(), args.output_dir / "best_model.pt")
        else:
            wait += 1
        if wait >= args.patience:
            break
    np.savetxt(
        args.output_dir / "training_history.csv",
        np.asarray(history),
        delimiter=",",
        header="epoch,train_loss,val_loss,learning_rate",
        comments="",
    )


if __name__ == "__main__":
    train_model(TrainConfig())
