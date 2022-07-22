import torch
import torch_geometric.nn as pyg_nn
import torch.nn.functional as F
import torch.nn as nn
from torch.nn.utils.rnn import  pad_sequence 
from torch_geometric.data import Data, Batch
import torch_geometric.utils as pyg_utils


import pytorch_lightning as pl
import torchmetrics

from gnn_convs import SingleConv


class SubtreeModel(pl.LightningModule):
    def __init__(self, num_iter=3, sample_size=0, test_time=3):
        """
        """
        super(SubtreeModel, self).__init__()
        self.sample_size = sample_size
        self.num_iter = num_iter
        self.test_time = test_time

        self.rw_max_conv = SingleConv(conv_type="random", aggr="max")
        self.sum_conv = SingleConv(aggr="add")
        self.max_conv = SingleConv(aggr="max")
        self.mean_conv = SingleConv(aggr="mean")

    def init_assign(self, data_t, data_s):
        degree_t = pyg_utils.degree(data_t.edge_index[0], num_nodes=data_t.x.size(0))
        degree_s = pyg_utils.degree(data_s.edge_index[0], num_nodes=data_s.x.size(0))

        assign_mat = [degree_t >= degree_s[i] for i in range(degree_s.size(0))]
        assign_mat = torch.stack(assign_mat)
        assign_mat = assign_mat.float().T
        return assign_mat

    def iter_assign(self, assign_mat, edge_index_t, edge_index_s):
        w1 = self.rw_max_conv(assign_mat.T, edge_index_s).T
        s1 = self.sum_conv(w1, edge_index_t)
        match_idx_1 = s1>=1

        w2 = self.max_conv(assign_mat, edge_index_t)
        s2 = self.mean_conv(w2.T, edge_index_s)
        match_idx_2 = s2.T>=1

        match_idx = match_idx_1 * match_idx_2
        
        result = torch.zeros(assign_mat.size())
        result[match_idx] = 1

        return result

    def forward(self, data_pair):
        data_t = Data(data_pair.x_t, data_pair.edge_index_t)
        data_s = Data(data_pair.x_s, data_pair.edge_index_s)

        assign_mat = self.init_assign(data_t, data_s)

        for i in range(self.num_iter):
            assign_mat_1 = self.iter_assign(assign_mat, data_t.edge_index, data_s.edge_index)
            
            if data_s.edge_index.size(1)>=1:
                for j in range(self.sample_size):
                    edge_size = torch.randint(1, data_s.edge_index.size(1)-1, [1])
                    edge_idx = torch.randperm(data_s.edge_index.size(1))[0:edge_size]
                    sampled_edge_index = data_s.edge_index[:,edge_idx]
                    assign_mat_2 = self.iter_assign(assign_mat, data_t.edge_index, sampled_edge_index)

                    valid_id = sampled_edge_index[1].unique()
                    assign_mat_1[:,valid_id] = assign_mat_1[:,valid_id] * assign_mat_2[:,valid_id]
            assign_mat = assign_mat_1
        
        result = torch.sum(torch.max(assign_mat, dim=1)[0])>=assign_mat.size(1)
        result = result * (torch.sum(torch.max(assign_mat, dim=0)[0])==assign_mat.size(1))

        for i in range(self.test_time):
            size = torch.randint(1, assign_mat.size(1)-1, [1])
            idx = torch.randperm(assign_mat.size(1))[0:size]
            sampled_assign_mat = assign_mat[:,idx]

            result = result * torch.sum(torch.max(sampled_assign_mat, dim=1)[0])>=size
        print(assign_mat)
        return result