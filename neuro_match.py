"""Defines all graph embedding models"""
from functools import reduce
import random

import networkx as nx
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch_geometric.nn as pyg_nn
import torch_geometric.utils as pyg_utils

import pytorch_lightning as pl
import torchmetrics


# GNN -> concat -> MLP graph classification baseline
class BaselineMLP(nn.Module):
    def __init__(self,num_layers, input_dim, hidden_dim, dropout = 0.5, skip = "learnable", conv_type = "GCN"):
        super(BaselineMLP, self).__init__()
        self.emb_model = SkipLastGNN(num_layers, input_dim, hidden_dim, hidden_dim, dropout = dropout, skip = skip, conv_type = conv_type)
        self.mlp = nn.Sequential(nn.Linear(2 * hidden_dim, 256), nn.ReLU(),
            nn.Linear(256, 2))

    def forward(self, emb_motif, emb_motif_mod):
        pred = self.mlp(torch.cat((emb_motif, emb_motif_mod), dim=1))
        pred = F.log_softmax(pred, dim=1)
        return pred

    def predict(self, pred):
        return pred#.argmax(dim=1)

    def criterion(self, pred, _, label):
        return F.nll_loss(pred, label)

# Order embedder model -- contains a graph embedding model `emb_model`
class OrderEmbedder(pl.LightningModule):
    def __init__(self, num_layers, input_dim, hidden_dim, margin, dropout = 0.5, skip = "learnable", conv_type = "GCN", \
        lr = 0.001, clf_lr = 0.001):
        super().__init__()
        self.automatic_optimization = False
        self.lr = lr
        self.clf_lr = clf_lr

        self.train_acc = torchmetrics.Accuracy()
        self.val_acc = torchmetrics.Accuracy()
        #self.loss = torchmetrics.SumMetric()
        #self.clf_loss = torchmetrics.SumMetric()

        self.emb_model = SkipLastGNN(num_layers, input_dim, hidden_dim, hidden_dim, dropout = dropout, skip = skip, conv_type = conv_type)
        self.margin = margin
        self.use_intersection = False

        self.clf_model = nn.Sequential(nn.Linear(1, 2), nn.LogSoftmax(dim=-1))

    def forward(self, data_pair):
        pos_a = data_pair.target_data()
        pos_b = data_pair.subgraph_data()

        emb_a = self.emb_model(pos_a)
        emb_b = self.emb_model(pos_b)

        out = (emb_a, emb_b)
        return out

    def training_step(self, batch, batch_idx):
        opt, clf_opt = self.optimizers()
        
        out = self(batch)
        #print(out)
        loss = self.criterion(out, batch.y)

        opt.zero_grad()
        self.manual_backward(loss)
        opt.step()

        with torch.no_grad():
            pred = self.predict(out)
        
        pred = self.clf_model(pred.unsqueeze(1))
        criterion = nn.NLLLoss()
        clf_loss = criterion(pred, batch.y)
        

        self.clf_model.zero_grad()
        self.manual_backward(clf_loss)
        clf_opt.step()

        #self.loss.update(loss.item())
        #self.clf_loss.update(clf_loss.item())
        
        self.log("loss", loss, prog_bar=True)
        self.log("clf loss", clf_loss, prog_bar=True)
        self.log("train acc", self.train_acc(pred, batch.y), prog_bar=True)

    def training_epoch_end(self, out):
        self.log("total train acc", self.train_acc.compute(), prog_bar=True)
        self.train_acc.reset()
        #self.loss.reset()
        #self.clf_loss.reset()


    def validation_step(self, batch, batch_idx):
        out = self(batch)
        pred = self.predict(out)

        self.log("val acc", self.val_acc(pred, batch.y))

    def validation_epoch_end(self, result):
        self.log("total val acc", self.val_acc.compute())
        self.val_acc.reset()

    def configure_optimizers(self):
        opt = torch.optim.Adam(self.parameters(), lr=self.lr)
        clf_opt = torch.optim.Adam(self.clf_model.parameters(), lr=self.lr)
        return opt, clf_opt

    def predict(self, pred):
        """Predict if b is a subgraph of a (batched), where emb_as, emb_bs = pred.

        pred: list (emb_as, emb_bs) of embeddings of graph pairs

        Returns: list of bools (whether a is subgraph of b in the pair)
        """
        emb_as, emb_bs = pred

        e = torch.sum(torch.max(torch.zeros_like(emb_as,
            device=emb_as.device), emb_bs - emb_as)**2, dim=1)
        return e

    def criterion(self, pred, labels):
        """Loss function for order emb.
        The e term is the amount of violation (if b is a subgraph of a).
        For positive examples, the e term is minimized (close to 0); 
        for negative examples, the e term is trained to be at least greater than self.margin.

        pred: lists of embeddings outputted by forward
        intersect_embs: not used
        labels: subgraph labels for each entry in pred
        """
        emb_as, emb_bs = pred
        e = torch.sum(torch.max(torch.zeros_like(emb_as), emb_bs - emb_as)**2, dim=1)

        margin = self.margin
        e[labels == 0] = torch.max(torch.tensor(0.0), margin - e)[labels == 0]

        relation_loss = torch.sum(e)

        return relation_loss

class SkipLastGNN(nn.Module):
    def __init__(self,num_layers, input_dim, hidden_dim, output_dim, dropout = 0.5, skip = "learnable", conv_type = "GCN"):
        super(SkipLastGNN, self).__init__()
        self.dropout = dropout
        self.n_layers = num_layers

        self.pre_mp = nn.Sequential(nn.Linear(input_dim, 3*hidden_dim if
            conv_type == "PNA" else hidden_dim))

        conv_model = self.build_conv_model(conv_type, 1)
        if conv_type == "PNA":
            self.convs_sum = nn.ModuleList()
            self.convs_mean = nn.ModuleList()
            self.convs_max = nn.ModuleList()
        else:
            self.convs = nn.ModuleList()

        if skip == 'learnable':
            self.learnable_skip = nn.Parameter(torch.ones(self.n_layers,
                self.n_layers))

        for l in range(num_layers):
            if skip == 'all' or skip == 'learnable':
                hidden_input_dim = hidden_dim * (l + 1)
            else:
                hidden_input_dim = hidden_dim
            if conv_type == "PNA":
                self.convs_sum.append(conv_model(3*hidden_input_dim, hidden_dim))
                self.convs_mean.append(conv_model(3*hidden_input_dim, hidden_dim))
                self.convs_max.append(conv_model(3*hidden_input_dim, hidden_dim))
            else:
                self.convs.append(conv_model(hidden_input_dim, hidden_dim))

        post_input_dim = hidden_dim * (num_layers + 1)
        if conv_type == "PNA":
            post_input_dim *= 3
        self.post_mp = nn.Sequential(
            nn.Linear(post_input_dim, hidden_dim), nn.Dropout(dropout),
            nn.LeakyReLU(0.1),
            nn.Linear(hidden_dim, output_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, 256), nn.ReLU(),
            nn.Linear(256, hidden_dim))
        #self.batch_norm = nn.BatchNorm1d(output_dim, eps=1e-5, momentum=0.1)
        self.skip = skip
        self.conv_type = conv_type

    def build_conv_model(self, model_type, n_inner_layers):
        if model_type == "GCN":
            return pyg_nn.GCNConv
        elif model_type == "GIN":
            #return lambda i, h: pyg_nn.GINConv(nn.Sequential(
            #    nn.Linear(i, h), nn.ReLU()))
            return lambda i, h: GINConv(nn.Sequential(
                nn.Linear(i, h), nn.ReLU(), nn.Linear(h, h)
                ))
        elif model_type == "SAGE":
            return SAGEConv
        elif model_type == "graph":
            return pyg_nn.GraphConv
        elif model_type == "GAT":
            return pyg_nn.GATConv
        elif model_type == "gated":
            return lambda i, h: pyg_nn.GatedGraphConv(h, n_inner_layers)
        elif model_type == "PNA":
            return SAGEConv
        else:
            print("unrecognized model type")

    def forward(self, data):
        #if data.x is None:
        #    data.x = torch.ones((data.num_nodes, 1), device=utils.get_device())

        #x = self.pre_mp(x)
        x, edge_index, batch = data.x, data.edge_index, data.batch
        x = self.pre_mp(x)

        all_emb = x.unsqueeze(1)
        emb = x
        for i in range(len(self.convs_sum) if self.conv_type=="PNA" else
            len(self.convs)):
            if self.skip == 'learnable':
                skip_vals = self.learnable_skip[i,
                    :i+1].unsqueeze(0).unsqueeze(-1)
                curr_emb = all_emb * torch.sigmoid(skip_vals)
                curr_emb = curr_emb.view(x.size(0), -1)
                if self.conv_type == "PNA":
                    x = torch.cat((self.convs_sum[i](curr_emb, edge_index),
                        self.convs_mean[i](curr_emb, edge_index),
                        self.convs_max[i](curr_emb, edge_index)), dim=-1)
                else:
                    x = self.convs[i](curr_emb, edge_index)
            elif self.skip == 'all':
                if self.conv_type == "PNA":
                    x = torch.cat((self.convs_sum[i](emb, edge_index),
                        self.convs_mean[i](emb, edge_index),
                        self.convs_max[i](emb, edge_index)), dim=-1)
                else:
                    x = self.convs[i](emb, edge_index)
            else:
                x = self.convs[i](x, edge_index)
            x = F.relu(x)
            x = F.dropout(x, p=self.dropout, training=self.training)
            emb = torch.cat((emb, x), 1)
            if self.skip == 'learnable':
                all_emb = torch.cat((all_emb, x.unsqueeze(1)), 1)

        # x = pyg_nn.global_mean_pool(x, batch)
        emb = pyg_nn.global_add_pool(emb, batch)
        emb = self.post_mp(emb)
        #emb = self.batch_norm(emb)   # TODO: test
        #out = F.log_softmax(emb, dim=1)
        return emb

    def loss(self, pred, label):
        return F.nll_loss(pred, label)

class SAGEConv(pyg_nn.MessagePassing):
    def __init__(self, in_channels, out_channels, aggr="add"):
        super(SAGEConv, self).__init__(aggr=aggr)

        self.in_channels = in_channels
        self.out_channels = out_channels

        self.lin = nn.Linear(in_channels, out_channels)
        self.lin_update = nn.Linear(out_channels + in_channels,
            out_channels)

    def forward(self, x, edge_index, edge_weight=None, size=None,
                res_n_id=None):
        """
        Args:
            res_n_id (Tensor, optional): Residual node indices coming from
                :obj:`DataFlow` generated by :obj:`NeighborSampler` are used to
                select central node features in :obj:`x`.
                Required if operating in a bipartite graph and :obj:`concat` is
                :obj:`True`. (default: :obj:`None`)
        """
        #edge_index, edge_weight = add_remaining_self_loops(
        #    edge_index, edge_weight, 1, x.size(self.node_dim))
        edge_index, _ = pyg_utils.remove_self_loops(edge_index)

        return self.propagate(edge_index, size=size, x=x,
                              edge_weight=edge_weight, res_n_id=res_n_id)

    def message(self, x_j, edge_weight):
        #return x_j if edge_weight is None else edge_weight.view(-1, 1) * x_j
        return self.lin(x_j)

    def update(self, aggr_out, x, res_n_id):
        aggr_out = torch.cat([aggr_out, x], dim=-1)

        aggr_out = self.lin_update(aggr_out)
        #aggr_out = torch.matmul(aggr_out, self.weight)

        #if self.bias is not None:
        #    aggr_out = aggr_out + self.bias

        #if self.normalize:
        #    aggr_out = F.normalize(aggr_out, p=2, dim=-1)

        return aggr_out

    def __repr__(self):
        return '{}({}, {})'.format(self.__class__.__name__, self.in_channels,
                                   self.out_channels)

# pytorch geom GINConv + weighted edges
class GINConv(pyg_nn.MessagePassing):
    def __init__(self, nn, eps=0, train_eps=False, **kwargs):
        super(GINConv, self).__init__(aggr='add', **kwargs)
        self.nn = nn
        self.initial_eps = eps
        if train_eps:
            self.eps = torch.nn.Parameter(torch.Tensor([eps]))
        else:
            self.register_buffer('eps', torch.Tensor([eps]))
        self.reset_parameters()

    def reset_parameters(self):
        #reset(self.nn)
        self.eps.data.fill_(self.initial_eps)

    def forward(self, x, edge_index, edge_weight=None):
        """"""
        x = x.unsqueeze(-1) if x.dim() == 1 else x
        edge_index, edge_weight = pyg_utils.remove_self_loops(edge_index,
            edge_weight)
        out = self.nn((1 + self.eps) * x + self.propagate(edge_index, x=x,
            edge_weight=edge_weight))
        return out

    def message(self, x_j, edge_weight):
        return x_j if edge_weight is None else edge_weight.view(-1, 1) * x_j

    def __repr__(self):
        return '{}(nn={})'.format(self.__class__.__name__, self.nn)

