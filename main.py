# -*- coding: utf-8 -*-
"""
@author: 34753
"""
from subtree_model import SubtreeModel
from neuro_match import OrderEmbedder
from e2e_model import E2eModel
from graphsim import SimGNN
from iso_net import ISONET
from dataset import SynDataset
import torch
import torch.nn.functional as F
import networkx as nx
import torch_geometric.utils as pyg_utils

import pytorch_lightning as pl

feat_dim = 10
hidden_dim = 2*feat_dim
root = "Exp"
feat_type = "ones"
device = "cuda"
num_layers = 2

from torch_geometric.loader import DataLoader

dataset = SynDataset(root, size=500, positive=250, graph_type=["ER","WS"], graph_sizes=40,\
    pos_subgraph_type="subtree", neg_subgraph_type="ER", subgraph_sizes=10,
    feat_type=feat_type, feat_dim=feat_dim, name = "40_10-500")

val_dataset = SynDataset(root, size=400, positive=200, graph_type=["ER","WS"], graph_sizes=40,\
    pos_subgraph_type="subtree", neg_subgraph_type="ER", subgraph_sizes=10,
    feat_type=feat_type, feat_dim=feat_dim, name = "40_10-400")

train_loader = DataLoader(dataset, batch_size=10, follow_batch=['x_s', 'x_t'], shuffle=True)
val_loader = DataLoader(val_dataset, batch_size=50, follow_batch=['x_s', 'x_t'])
#model = E2eModel(feat_dim, hidden_dim, inner_layers=5, inter_type = "Multi", lr=3e-4)
#model = E2eModel(feat_dim, hidden_dim, inner_layers=5, inter_type = "GNN", lr=3e-4)
#model = SimGNN(5, feat_dim, hidden_dim, hidden_dim*2, hidden_dim, conv_type="GIN", lr=3e-4)
#model = OrderEmbedder(5,feat_dim,hidden_dim,0.1, conv_type = "GIN", lr=3e-4)
#model = ISONET(feat_dim, hidden_dim, hidden_dim, hidden_dim, prop_layers=5, lr=3e-4)
model = SubtreeModel(num_iter = 5, sample_size=50,test_time=20)


#result = model(dataset[301])
result = []
for i in range(len(dataset)):
    
    result.append(model(dataset[i]))
    print(i,":",result[-1])
result = torch.tensor(result)
label = [i.y for i in dataset]
label = torch.tensor(label)
correct = (result==label)
print("Acc:{}".format(sum(correct)/len(correct)))

#trainer = pl.Trainer(max_epochs=1000, accelerator = 'gpu', gpus=[3])
#trainer.fit(model, train_loader, val_loader)
