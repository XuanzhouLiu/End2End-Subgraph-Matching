# -*- coding: utf-8 -*-
"""
Created on Sun Jun 26 03:01:36 2022

@author: 34753
"""

from functools import reduce
import random

import networkx as nx
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.data import Data
import torch_geometric.nn as pyg_nn
import torch_geometric.utils as pyg_utils
from torch_geometric.nn import MessagePassing

import pytorch_lightning as pl
import torchmetrics

from gnn_convs import build_conv_model

class E2eModel(pl.LightningModule):
    def __init__(self, input_dim, hidden_dim, inner_layers = 2, inter_layers = 2, lr = 0.001):
        super(E2eModel, self).__init__()
        self.input_dim = input_dim
        self.hidden_dim = hidden_dim
        
        self.inter_gnn = InterGNN(inter_layers, hidden_dim,2*hidden_dim, hidden_dim, hidden_dim)
        self.target_gnn = InnerGNN(inner_layers, input_dim, hidden_dim, hidden_dim, conv_type="GIN")
        self.sub_gnn = InnerGNN(inner_layers, input_dim, hidden_dim, hidden_dim, conv_type="GIN")

        self.post_layer = nn.Sequential(
            nn.Linear(hidden_dim, 2*hidden_dim), nn.Dropout(0.5),
            nn.ReLU(),
            nn.Linear(2*hidden_dim, 2))

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
        inter_x = torch.cat([x_t,x_s], dim=0)

        inter_graph = Data(x=inter_x, edge_index=inter_edge_index).to(inter_x.device)
        inter_graph.edge_feat = torch.zeros([inter_graph.num_edges, self.hidden_dim]).to(inter_x.device)
        
        out_x, edge_emb = self.inter_gnn(inter_graph)
        
        edge_emb = pyg_nn.global_mean_pool(edge_emb, inter_edge_batch)
        output = self.post_layer(edge_emb)
        return F.log_softmax(output)

    def training_step(self, batch, batch_idx):

        out = self(batch)
        #print(out)
        loss = F.nll_loss(out, batch.y)

        self.log("train_loss", loss)
        self.log("train acc", self.train_acc(out, batch.y))
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


class InterGNN(nn.Module):
    def __init__(self, num_layers, input_dim, hidden_dim, output_dim, edge_dim, dropout = 0.5):
        super(InterGNN, self).__init__()
        self.dropout = dropout
        self.n_layers = num_layers
        
        self.convs = nn.ModuleList()
        for l in range(num_layers-1):
            self.convs.append(EdgeConv(input_dim, hidden_dim, hidden_dim, edge_dim))
            input_dim = hidden_dim
        self.convs.append(EdgeConv(input_dim, hidden_dim, output_dim, edge_dim))
        
        
        
    def forward(self, data):
        x, edge_index, edge_feat = data.x, data.edge_index, data.edge_feat
        for i in range(len(self.convs)):
            x, edge_feat = self.convs[i](x, edge_index, edge_feat)
        
        return x, edge_feat


class PolyGNN(nn.Module):
    def __init__(self, order, input_dim, hidden_dim, output_dim, dropout=0.5):
        super(PolyGNN, self).__init__()
        self.dropout = dropout
        self.order = order
        
        self.pre_mp = pyg_nn.MLP(num_layers = 2, in_channels = input_dim, hidden_channels = hidden_dim, out_channels = output_dim)

class InnerGNN(nn.Module):
    def __init__(self, num_layers, input_dim, hidden_dim, output_dim, dropout=0.5, conv_type="GCN", skip="all"):
        super(InnerGNN, self).__init__()
        self.dropout = dropout
        self.n_layers = num_layers
        self.skip = skip
        self.conv_type = conv_type
        
        self.pre_mp = nn.Sequential(nn.Linear(input_dim, hidden_dim))

        conv_model = build_conv_model(conv_type, 1)

        self.convs = nn.ModuleList()

        if skip == 'learnable':
            self.learnable_skip = nn.Parameter(torch.ones(self.n_layers,
                self.n_layers))

        for l in range(num_layers):
            if skip == 'all' or skip == 'learnable':
                hidden_input_dim = hidden_dim * (l + 1)
            else:
                hidden_input_dim = hidden_dim

            self.convs.append(conv_model(hidden_input_dim, hidden_dim))

        post_input_dim = hidden_dim * (num_layers + 1)

        self.post_mp = nn.Sequential(
            nn.Linear(post_input_dim, hidden_dim), nn.Dropout(dropout),
            nn.LeakyReLU(0.1),
            nn.Linear(hidden_dim, output_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, 256), nn.ReLU(),
            nn.Linear(256, hidden_dim))
        #self.batch_norm = nn.BatchNorm1d(output_dim, eps=1e-5, momentum=0.1)
        

    def forward(self, data):
        x, edge_index = data.x, data.edge_index
        x = self.pre_mp(x)

        all_emb = x.unsqueeze(1)
        emb = x
        for i in range(len(self.convs)):
            if self.skip == 'learnable':
                skip_vals = self.learnable_skip[i,
                    :i+1].unsqueeze(0).unsqueeze(-1)
                curr_emb = all_emb * torch.sigmoid(skip_vals)
                curr_emb = curr_emb.view(x.size(0), -1)

                x = self.convs[i](curr_emb, edge_index)
            elif self.skip == 'all':
                x = self.convs[i](emb, edge_index)
            else:
                x = self.convs[i](x, edge_index)
            x = F.relu(x)
            x = F.dropout(x, p=self.dropout, training=self.training)
            emb = torch.cat((emb, x), 1)
            if self.skip == 'learnable':
                all_emb = torch.cat((all_emb, x.unsqueeze(1)), 1)

        emb = self.post_mp(emb)
        return emb

    def loss(self, pred, label):
        return F.nll_loss(pred, label)

class EdgeConv(MessagePassing):
    def __init__(self, input_dim, hidden_dim, output_dim, edge_dim):
        super().__init__(aggr='sum') #  "Max" aggregation.
        self.edge_mlp = nn.Sequential(nn.Linear(edge_dim + 2*input_dim, 2*edge_dim),
                       nn.ReLU(),
                       nn.Linear(2*edge_dim, edge_dim))
        self.node_mlp = nn.Sequential(nn.Linear(2*input_dim+edge_dim, hidden_dim),
                       nn.ReLU(),
                       nn.Linear(hidden_dim, output_dim))
    
    def edge_update(self, x_i, x_j, edge_feat):
        emb = torch.cat([x_i, x_j, edge_feat], dim=1)
        edge_feat = self.edge_mlp(emb)
        return edge_feat

    def forward(self, x, edge_index, edge_feat):
        
        #edge_index, _ = pyg_utils.add_self_loops(edge_index, num_nodes=x.size(0))
        edge_feat = self.edge_updater(edge_index, x=x, edge_feat=edge_feat)

        out = self.propagate(edge_index, x=x, edge_feat=edge_feat)
        
        return out, edge_feat
    def message(self, x_i, x_j, edge_feat):
        tmp = torch.cat([x_i, x_j, edge_feat], dim=1)
        return self.node_mlp(tmp)