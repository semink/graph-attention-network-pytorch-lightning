import torch
import torch.nn as nn
import torch.nn.functional as F
import pytorch_lightning as pl
from model.GAT.layer import GraphAttentionLayer
from utils import accuracy

class GAT(pl.LightningModule):
    def __init__(self, nfeat, nhid, nclass, dropout, alpha, nheads):
        """Dense version of GAT."""
        super(GAT, self).__init__()
        self.dropout = dropout

        self.attentions =nn.ModuleList([GraphAttentionLayer(nfeat, nhid, dropout=dropout, alpha=alpha, concat=True) for _ in range(nheads)])
        for i, attention in enumerate(self.attentions):
            self.add_module('attention_{}'.format(i), attention)

        self.out_att = GraphAttentionLayer(nhid * nheads, nclass, dropout=dropout, alpha=alpha, concat=False)
        self.save_hyperparameters()

    def forward(self, x, adj):
        x = F.dropout(x, self.dropout, training=self.training)
        x = torch.cat([att(x, adj) for att in self.attentions], dim=1)
        x = F.dropout(x, self.dropout, training=self.training)
        x = F.elu(self.out_att(x, adj))
        return F.log_softmax(x, dim=1)
    
    def training_step(self, batch, batch_idx):
        x, labels, adj = batch.x, batch.y, batch.adj
        logits = self(x, adj)
        loss = F.nll_loss(logits[batch.train_mask], labels[batch.train_mask])
        self.log('train/cross_entropy', loss)
        return loss
    
    
    def validation_step(self, batch, batch_idx):
        x, labels, adj = batch.x, batch.y, batch.adj
        logits = self(x, adj)
        acc = accuracy(logits[batch.val_mask], labels[batch.val_mask])
        loss = F.nll_loss(logits[batch.val_mask], labels[batch.val_mask])
        self.log('val/acc', acc)
        self.log('val/cross_entropy', loss)
    
    def test_step(self, batch, batch_idx):
        x, labels, adj = batch.x, batch.y, batch.adj
        logits = self(x, adj)
        acc = accuracy(logits[batch.test_mask], labels[batch.test_mask])
        loss = F.nll_loss(logits[batch.test_mask], labels[batch.test_mask])
        self.log('test/acc', acc)
        self.log('test/cross_entropy', loss)
    
    def configure_optimizers(self):
        optimizer = torch.optim.Adam(self.parameters(), lr=0.005, weight_decay=5e-4)
        return optimizer
    