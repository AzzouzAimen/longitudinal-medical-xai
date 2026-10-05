from pathlib import Path

from medattnaid.train import TrainConfig, train_model


def main():
    variants = [
        ("mse_pattern", "mse_pattern"),
        ("mse_trend", "mse_trend"),
        ("cploss", "cploss"),
    ]
    for variant, dirname in variants:
        train_model(
            TrainConfig(
                loss_variant=variant,
                output_dir=Path("results/ablations") / dirname,
                learning_rate=0.0007,
                weight_decay=0.00005,
                max_epochs=25,
                patience=8,
            )
        )


if __name__ == "__main__":
    main()
