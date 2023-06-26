from torch_geometric.datasets import Planetoid
from model.GAT.model import GAT

import torch

import pytorch_lightning as pl
from pytorch_lightning.callbacks import EarlyStopping
from pytorch_lightning.callbacks import ModelCheckpoint
from pytorch_lightning.loggers import TensorBoardLogger
import torch_geometric.transforms as T

import random
import numpy as np

import pandas as pd
import os


# main function
def main(seed=1):
    # initialize seed
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)

    transform = T.Compose([T.NormalizeFeatures(), T.AddSelfLoops()])
    dataset = Planetoid(root="./data", name="Cora", transform=transform)

    cross_entropy_early_stop_callback = EarlyStopping(
        monitor="val/cross_entropy", mode="min", patience=100
    )
    accuracy_early_stop_callback = EarlyStopping(
        monitor="val/acc", mode="max", patience=100
    )

    checkpoint_callback = ModelCheckpoint(
        save_top_k=1, monitor="val/cross_entropy", mode="min"
    )

    logger = TensorBoardLogger("tb_logs", name="GAT", version=f"seed_{seed}")
    trainer = pl.Trainer(
        callbacks=[
            cross_entropy_early_stop_callback,
            accuracy_early_stop_callback,
            checkpoint_callback,
        ],
        logger=logger,
        max_epochs=10000,
        accelerator="auto",
    )

    model = GAT(
        num_of_layers=2,
        num_heads_per_layer=[8, 1],
        num_features_per_layer=[dataset.num_features, 8, dataset.num_classes],
        add_skip_connection=False,
        bias=True,
        dropout=0.6,
        log_attention_weights=False,
    )
    trainer.fit(model, train_dataloaders=dataset, val_dataloaders=dataset)

    # best_model = GAT.load_from_checkpoint(checkpoint_callback.best_model_path)
    return trainer.test(model, dataloaders=dataset)


if __name__ == "__main__":
    for seed in range(100):
        test_result = main(seed)[0]
        test_accuracy = [
            {
                "seed": seed,
                "accuracy": test_result["test/acc"],
                "cross_entropy": test_result["test/cross_entropy"],
            }
        ]
        if os.path.exists("test_accuracy.csv"):
            test_accuracy_df = pd.read_csv("test_accuracy.csv", index_col="seed")
            pd.concat(
                [test_accuracy_df, pd.DataFrame(test_accuracy).set_index("seed")]
            ).to_csv("test_accuracy.csv")
        else:
            pd.DataFrame(test_accuracy).set_index("seed").to_csv("test_accuracy.csv")
