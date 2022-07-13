# -*- coding: utf-8 -*-
"""
@author: 34753
"""

from neuro_match import OrderEmbedder
from e2e_model import E2eModel
from graphsim import SimGNN
from iso_net import ISONET
from dataset import SynDataset
import torch
import torch.nn.functional as F
import networkx as nx
from torch_geometric.utils import from_networkx

import pytorch_lightning as pl

feat_dim = 10
hidden_dim = 2*feat_dim
root = "Exp"
feat_type = "ones"
device = "cuda"
num_layers = 2

from torch_geometric.loader import DataLoader

dataset = SynDataset(root, size=500, positive=250, graph_type=["ER","WS"], graph_sizes=20,\
    pos_subgraph_type="subtree", neg_subgraph_type="ER", subgraph_sizes=5,
    feat_type=feat_type, feat_dim=feat_dim)

val_dataset = SynDataset(root, size=200, positive=100, graph_type=["ER","WS"], graph_sizes=20,\
    pos_subgraph_type="subtree", neg_subgraph_type="ER", subgraph_sizes=5,
    feat_type=feat_type, feat_dim=feat_dim)

train_loader = DataLoader(dataset, batch_size=10, follow_batch=['x_s', 'x_t'], shuffle=True)
val_loader = DataLoader(val_dataset, batch_size=10, follow_batch=['x_s', 'x_t'], shuffle=True)
#model = E2eModel(feat_dim, hidden_dim)
#model = SimGNN(2, feat_dim, hidden_dim, hidden_dim*2, hidden_dim, conv_type="GIN")
#model = OrderEmbedder(2,feat_dim,hidden_dim,0.1, conv_type = "GIN")
model = ISONET(feat_dim, hidden_dim, hidden_dim, hidden_dim)

trainer = pl.Trainer(max_epochs=1000, accelerator = 'gpu', gpus=[3])
trainer.fit(model, train_loader, val_loader)
