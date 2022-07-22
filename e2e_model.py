# -*- coding: utf-8 -*-
"""
Created on Sun Jun 26 03:01:36 2022

@author: 34753
"""

from functools import reduce

import networkx as nx
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.data import Data
import torch_geometric.nn as pyg_nn
import torch_geometric.utils as pyg_utils


import pytorch_lightning as pl
import torchmetrics

from gnns import GeneralGNN, EdgeGNN, DecoupledGNN
from utils import Nomial

class E2eModel(pl.LightningModule):
    def __init__(self, input_dim, hidden_dim, inner_layers = 2, inter_layers = 2, inter_type = "GNN", aggr_type = "avg", dropout = 0.5, lr = 0.001):
        super(E2eModel, self).__init__()
        self.inner_layers = inner_layers
        self.input_dim = input_dim
        self.hidden_dim = hidden_dim
        self.inter_type = inter_type
        self.aggr_type = aggr_type
        if self.inter_type == "GNN":
            self.inter_gnn = EdgeGNN(inter_layers, hidden_dim,2*hidden_dim, hidden_dim, hidden_dim)
            self.post_layer = nn.Sequential(
                nn.Linear(hidden_dim, 2*hidden_dim), nn.Dropout(dropout),
                nn.ReLU(),
                nn.Linear(2*hidden_dim, 2))
        elif self.inter_type == "Sim":
            self.inter_sim = InterSim(hidden_dim)
            self.clf_layer = nn.Sequential(nn.Linear(1,hidden_dim), nn.Dropout(dropout), nn.ReLU(), nn.Linear(hidden_dim,2))
        elif self.inter_type == "Multi":
            self.inter_mat = InterMat(inner_layers, hidden_dim)
            self.clf_layer = nn.Sequential(nn.Linear(1,hidden_dim), nn.Dropout(dropout), nn.ReLU(), nn.Linear(hidden_dim,2))
        
        if inter_type == "Multi":
            self.target_gnn = DecoupledGNN(Nomial(inner_layers), inner_layers, input_dim, hidden_dim, hidden_dim)
            self.sub_gnn = DecoupledGNN(Nomial(inner_layers), inner_layers, input_dim, hidden_dim, hidden_dim)
        else:
            self.target_gnn = GeneralGNN(inner_layers, input_dim, hidden_dim, hidden_dim, conv_type="GIN")
            self.sub_gnn = GeneralGNN(inner_layers, input_dim, hidden_dim, hidden_dim, conv_type="GIN")

        self.train_acc = torchmetrics.Accuracy()
        self.val_acc = torchmetrics.Accuracy()
        self.lr = lr
        
    def forward(self, data_pair):
        data_t = Data(data_pair.x_t, data_pair.edge_index_t)
        data_s = Data(data_pair.x_s, data_pair.edge_index_s)

        x_t = self.target_gnn(data_t)
        x_s = self.sub_gnn(data_s)
        
        '''
        t_batch = data_pair.x_t_batch
        s_batch = data_pair.x_s_batch

        emb_t = pyg_nn.global_add_pool(x_t, t_batch)
        emb_s = pyg_nn.global_add_pool(x_s, s_batch)

        return F.log_softmax(self.post_layer(emb_t-emb_s))
        '''
        if self.inter_type == "GNN":
            begin_s = x_t.size(0)
            begin_t = 0
            inter_edge_index = []
            inter_edge_batch = []
            inter_x = []
            for i in range(len(data_pair.s_nodes)):
                row = torch.cat([torch.arange(begin_t, begin_t + data_pair.t_nodes[i])]*data_pair.s_nodes[i])
                col = torch.stack([torch.arange(begin_s, begin_s + data_pair.s_nodes[i])]*data_pair.t_nodes[i]).T.reshape(-1)

                edge_index = torch.cat([torch.stack([row, col]), torch.stack([col, row])], dim=1)
                
                inter_edge_index.append(edge_index)

                inter_edge_batch.append(torch.full([edge_index.size(1)],i))

                begin_t += data_pair.t_nodes[i]
                begin_s += data_pair.s_nodes[i]

            inter_edge_index = torch.cat(inter_edge_index, dim=1)
            inter_edge_batch = torch.cat(inter_edge_batch).to(x_t.device)
            inter_x = torch.cat([x_t, x_s], dim=0)

            inter_graph = Data(x=inter_x, edge_index=inter_edge_index).to(inter_x.device)
            inter_graph.edge_feat = torch.zeros([inter_graph.num_edges, self.hidden_dim]).to(inter_x.device)
            
            out_x, edge_emb = self.inter_gnn(inter_graph)
            
            if self.aggr_type == "avg":
                score = pyg_nn.global_mean_pool(edge_emb, inter_edge_batch)
            score = self.post_layer(score)
        elif self.inter_type == "Sim":
            x_t, x_t_mask = pyg_utils.to_dense_batch(x_t, data_pair.x_t_batch)
            x_s, x_s_mask = pyg_utils.to_dense_batch(x_s, data_pair.x_s_batch)
            
            out = self.inter_sim(x_t,x_s)
            out = x_t_mask.unsqueeze(2) * out * x_s_mask.unsqueeze(1)
            if self.aggr_type == "avg":
                score = torch.sum(torch.sum(out, dim=1)/data_pair.s_nodes.view(-1,1), dim=1)
            elif self.aggr_type == "max":
                score = torch.sum(torch.max(out, dim=1)[0]/data_pair.s_nodes.view(-1,1), dim=1)
            score = self.clf_layer(score.unsqueeze(1))
            #print(score)
        elif self.inter_type == "Multi":
            x_t, x_t_mask = pyg_utils.to_dense_batch(x_t, data_pair.x_t_batch)
            x_s, x_s_mask = pyg_utils.to_dense_batch(x_s, data_pair.x_s_batch)

            x_t = torch.stack(torch.split(x_t, x_t.size(-1)//self.inner_layers, dim=-1))
            x_s = torch.stack(torch.split(x_s, x_s.size(-1)//self.inner_layers, dim=-1))

            out = self.inter_mat(x_t,x_s)
            out = x_t_mask.unsqueeze(2) * out * x_s_mask.unsqueeze(1)
            #x_s/x_t: basis_num * batch * node_num * hidden_dim
            if self.aggr_type == "avg":
                score = torch.sum(torch.sum(out, dim=1)/data_pair.s_nodes.view(-1,1), dim=1)
            elif self.aggr_type == "max":
                score = torch.sum(torch.max(out, dim=1)[0]/data_pair.s_nodes.view(-1,1), dim=1)
            score = self.clf_layer(score.unsqueeze(1))

        return F.log_softmax(score)

    def training_step(self, batch, batch_idx):

        out = self(batch)
        #print(out)
        loss = F.nll_loss(out, batch.y)

        self.log("train_loss", loss)
        self.log("train acc", self.train_acc(out, batch.y))
        #print(self.target_gnn.basis_coef.data)
        return loss
    
    def training_epoch_end(self, out):
        self.log("total train acc", self.train_acc.compute(), prog_bar=True)
        self.train_acc.reset()

    def validation_step(self, batch, batch_idx):
        out = self(batch)
        self.log("val acc", self.val_acc(out, batch.y))
    
    def validation_epoch_end(self, out):
        self.log("total val acc", self.val_acc.compute(), prog_bar=True)
        self.val_acc.reset()

    def configure_optimizers(self):
        opt = torch.optim.Adam(self.parameters(), lr=self.lr)
        return opt


class InterSim(nn.Module):
    def __init__(self, input_dim, kernel_dim = None, hidden_dim = None, lr_diff = True):
        super(InterSim, self).__init__()
        if kernel_dim is None:
            kernel_dim = input_dim
        if hidden_dim is None:
            hidden_dim = 2*kernel_dim
        self.left_trans = pyg_nn.MLP([input_dim, hidden_dim, kernel_dim], batch_norm=False)
        if lr_diff:
            self.right_trans = pyg_nn.MLP([input_dim, hidden_dim, kernel_dim], batch_norm=False)
        else:
            self.right_trans = self.left_trans
        
        self.kernel = torch.nn.Parameter(torch.FloatTensor(kernel_dim, kernel_dim))
        torch.nn.init.xavier_uniform_(self.kernel)
    
    def forward(self, left_mat, right_mat):
        left_mat = self.left_trans(left_mat)
        right_mat = self.right_trans(right_mat)
        out = left_mat @ self.kernel @ right_mat.permute(0,2,1)

        return out

class InterMat(nn.Module):
    def __init__(self, num_mat, input_dim, kernel_dim = None, hidden_dim = None, lr_diff = True, init_scale = 0.1):
        super(InterMat, self).__init__()
        if kernel_dim is None:
            kernel_dim = input_dim
        if hidden_dim is None:
            hidden_dim = 2*kernel_dim
        
        self.left_trans = pyg_nn.MLP([input_dim, hidden_dim, kernel_dim], batch_norm=False)
        if lr_diff:
            self.right_trans = pyg_nn.MLP([input_dim, hidden_dim, kernel_dim], batch_norm=False)
        else:
            self.right_trans = self.left_trans
        
        self.inter_kernel = torch.nn.Parameter(torch.FloatTensor(kernel_dim, kernel_dim))
        torch.nn.init.xavier_uniform_(self.inter_kernel)

        self.mat_kernel = torch.nn.Parameter(init_scale*torch.randn(num_mat))


    def forward(self, left_mat_list, right_mat_list):
        #mat_list: basis_num * batch * node_num * hidden_dim
        left_mat_list = self.left_trans(left_mat_list)
        right_mat_list = self.right_trans(right_mat_list)
        out = left_mat_list @ self.inter_kernel @ right_mat_list.permute(0,1,3,2)
        out = out*self.mat_kernel.view(-1,1,1,1)

        out = torch.sum(out, dim=0)
        return out
