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



# main function
def main(seed=1):
    
    # initialize seed
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    
    transform = T.Compose([T.ToUndirected(), T.AddSelfLoops(), T.ToDense()])
    dataset = Planetoid(root='./data', name='Cora', transform=transform)
    
    cross_entropy_early_stop_callback = EarlyStopping(monitor='val/cross_entropy', mode='min', patience=100)
    accuracy_early_stop_callback = EarlyStopping(monitor='val/acc', mode='max', patience=100)
    
    checkpoint_callback = ModelCheckpoint(save_top_k=2, monitor="val/cross_entropy", mode="min")
    
    logger = TensorBoardLogger("tb_logs", name="GAT", version=f'seed_{seed}')
    trainer = pl.Trainer(callbacks=[
                                    cross_entropy_early_stop_callback, 
                                    accuracy_early_stop_callback,
                                    checkpoint_callback
                                    ], logger=logger, max_epochs=1000)
    
    model = GAT(nfeat=dataset.num_features, nhid=8, nclass=dataset.num_classes, 
                dropout=0.6, alpha=0.2, nheads=8)
    trainer.fit(model, train_dataloaders=dataset, val_dataloaders=dataset)
    
    best_model = GAT.load_from_checkpoint(checkpoint_callback.best_model_path)
    trainer.test(best_model, dataloaders=dataset)


if __name__ == '__main__':
    for seed in range(1, 100):
        print(f"Train model with seed: {seed}")
        main(seed)
    