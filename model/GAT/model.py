import torch
import torch.nn as nn
import pytorch_lightning as pl
from model.GAT.layer import GATLayerImp3
from utils import accuracy


class GAT(pl.LightningModule):
    """
    I've added 3 GAT implementations - some are conceptually easier to understand some are more efficient.

    The most interesting and hardest one to understand is implementation #3.
    Imp1 and imp2 differ in subtle details but are basically the same thing.

    Tip on how to approach this:
        understand implementation 2 first, check out the differences it has with imp1, and finally tackle imp #3.

    """

    def __init__(
        self,
        num_of_layers,
        num_heads_per_layer,
        num_features_per_layer,
        add_skip_connection=True,
        bias=True,
        dropout=0.6,
        log_attention_weights=False,
    ):
        super().__init__()
        assert (
            num_of_layers == len(num_heads_per_layer) == len(num_features_per_layer) - 1
        ), f"Enter valid arch params."

        num_heads_per_layer = [
            1
        ] + num_heads_per_layer  # trick - so that I can nicely create GAT layers below

        gat_layers = []  # collect GAT layers
        for i in range(num_of_layers):
            layer = GATLayerImp3(
                num_in_features=num_features_per_layer[i]
                * num_heads_per_layer[i],  # consequence of concatenation
                num_out_features=num_features_per_layer[i + 1],
                num_of_heads=num_heads_per_layer[i + 1],
                concat=True
                if i < num_of_layers - 1
                else False,  # last GAT layer does mean avg, the others do concat
                activation=nn.ELU()
                if i < num_of_layers - 1
                else None,  # last layer just outputs raw scores
                dropout_prob=dropout,
                add_skip_connection=add_skip_connection,
                bias=bias,
                log_attention_weights=log_attention_weights,
            )
            gat_layers.append(layer)

        self.gat_net = nn.Sequential(
            *gat_layers,
        )
        self.loss_fn = nn.CrossEntropyLoss()

    # data is just a (in_nodes_features, topology) tuple, I had to do it like this because of the nn.Sequential:
    # https://discuss.pytorch.org/t/forward-takes-2-positional-arguments-but-3-were-given-for-nn-sqeuential-with-linear-layers/65698
    def forward(self, data):
        return self.gat_net(data)[0]

    def training_step(self, batch, batch_idx):
        x, labels, edge_index = batch.x, batch.y, batch.edge_index
        graph_data = (x, edge_index)
        logits = self(graph_data)
        loss = self.loss_fn(logits[batch.train_mask], labels[batch.train_mask])
        self.log("train/cross_entropy", loss)
        return loss

    def validation_step(self, batch, batch_idx):
        x, labels, edge_index = batch.x, batch.y, batch.edge_index
        graph_data = (x, edge_index)
        logits = self(graph_data)
        acc = accuracy(logits[batch.val_mask], labels[batch.val_mask])
        loss = self.loss_fn(logits[batch.val_mask], labels[batch.val_mask])
        self.log("val/acc", acc)
        self.log("val/cross_entropy", loss)

    def test_step(self, batch, batch_idx):
        x, labels, edge_index = batch.x, batch.y, batch.edge_index
        graph_data = (x, edge_index)
        logits = self(graph_data)
        acc = accuracy(logits[batch.test_mask], labels[batch.test_mask])
        loss = self.loss_fn(logits[batch.test_mask], labels[batch.test_mask])
        self.log("test/acc", acc)
        self.log("test/cross_entropy", loss)

    def configure_optimizers(self):
        optimizer = torch.optim.Adam(self.parameters(), lr=0.005, weight_decay=5e-4)
        return optimizer
