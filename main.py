# -*- coding: utf-8 -*-
"""
@author: 34753
"""

from neuro_match import OrderEmbedder
from e2e_model import E2eModel
from graphsim import SimGNN
from dataset import SynDataset
import torch
import torch.nn.functional as F
import networkx as nx
from torch_geometric.utils import from_networkx

import pytorch_lightning as pl

feat_dim = 10
hidden_dim = 2*feat_dim
root = "Exp"

num_layers = 2

from torch_geometric.loader import DataLoader

dataset = SynDataset(root, size=100, positive=50, graph_type=["ER","WS"], graph_sizes=20,\
    pos_subgraph_type="subtree", neg_subgraph_type="ER", subgraph_sizes=5,
    feat_type="random", feat_dim=feat_dim)

loader = DataLoader(dataset, batch_size=10, follow_batch=['x_s', 'x_t'], shuffle=True)

#model = E2eModel(feat_dim, hidden_dim)
model = OrderEmbedder(2,feat_dim,hidden_dim,0.1)
#model = SimGNN(2, feat_dim, hidden_dim, hidden_dim*2, hidden_dim)

opt = torch.optim.Adam(model.parameters(),lr = 3e-4)

train_loader = loader

trainer = pl.Trainer(max_epochs=100000)
trainer.fit(model, train_loader, train_loader)