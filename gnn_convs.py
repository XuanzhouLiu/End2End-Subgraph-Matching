
import torch.nn as nn
import torch

import torch_geometric.nn as pyg_nn
import torch_geometric.utils as pyg_utils
from torch_geometric.nn import MessagePassing

def build_conv_model(model_type, n_inner_layers):
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


class SingleConv(MessagePassing):
    def __init__(self, conv_type = "sum", self_loop = False, aggr = "add"):
        super().__init__(aggr=aggr)
        self.conv_type = conv_type
        self.self_loop = self_loop

    def forward(self, x, edge_index):
        if self.self_loop:
            edge_index, _ = pyg_utils.add_self_loops(edge_index, num_nodes=x.size(0))

        if self.conv_type == "sum":
            x = self.propagate(edge_index, x=x)
        elif self.conv_type == "normalized":
            row, col = edge_index
            deg = pyg_utils.degree(col, x.size(0), dtype=x.dtype)
            deg_inv_sqrt = deg.pow(-0.5)
            deg_inv_sqrt[deg_inv_sqrt == float('inf')] = 0
            norm = deg_inv_sqrt[row] * deg_inv_sqrt[col]

            x = self.propagate(edge_index, x=x, edge_weight=norm)
        elif self.conv_type == "random":
            row, col = edge_index
            deg = pyg_utils.degree(col, x.size(0), dtype=x.dtype)
            deg_inv_sqrt = deg.pow(-1)
            deg_inv_sqrt[deg_inv_sqrt == float('inf')] = 0

            norm = deg_inv_sqrt[col]
            x = self.propagate(edge_index, x=x, edge_weight=norm)
        return x


    def message(self, x_j, edge_weight = None):
        return x_j if edge_weight is None else edge_weight.view(-1,1) * x_j