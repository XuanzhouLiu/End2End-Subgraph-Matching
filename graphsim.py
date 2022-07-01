import torch
import torch_geometric.nn as pyg_nn
import torch.nn.functional as F
from torch.nn.utils.rnn import  pad_sequence 
from torch_geometric.data import Data, Batch
from torch_geometric.utils import to_dense_batch

import pytorch_lightning as pl
import torchmetrics

class SimGNN(pl.LightningModule):
    def __init__(self,gnn_layers, input_dim, hidden_dim, tensor_neurons, bottle_neck_neurons, histogram = False, bins = 10, dropout=0, \
        lr = 0.01):
        """
        """
        super(SimGNN, self).__init__()

        self.input_dim = input_dim
        self.tensor_neurons = tensor_neurons
        self.dropout = dropout
        self.gnn_layers = gnn_layers
        self.bottle_neck_neurons = bottle_neck_neurons
        self.histogram = histogram
        self.bins = bins
        self.lr = lr

        self.mse_loss = torchmetrics.MeanSquaredError()
        self.train_acc = torchmetrics.Accuracy(num_classes=2)
        self.val_acc = torchmetrics.Accuracy(num_classes=2)

        #Conv layers
        if not isinstance(hidden_dim, list):
            hidden_dim = [hidden_dim]*gnn_layers
        self.hidden_dim = hidden_dim
        assert(gnn_layers == len(hidden_dim))

        self.convs = torch.nn.ModuleList()
        for dim in hidden_dim:
            self.convs.append(pyg_nn.GCNConv(input_dim, dim))
            input_dim = dim
        
        #Attention
        self.attention_layer = torch.nn.Linear(self.hidden_dim[-1], self.hidden_dim[-1], bias=False)
        torch.nn.init.xavier_uniform_(self.attention_layer.weight)
        #NTN
        self.ntn_a = torch.nn.Bilinear(self.hidden_dim[-1], self.hidden_dim[-1], tensor_neurons, bias=False)
        torch.nn.init.xavier_uniform_(self.ntn_a.weight)
        self.ntn_b = torch.nn.Linear(2*self.hidden_dim[-1], tensor_neurons, bias=False)
        torch.nn.init.xavier_uniform_(self.ntn_b.weight)
        self.ntn_bias = torch.nn.Parameter(torch.Tensor(tensor_neurons,1))
        torch.nn.init.xavier_uniform_(self.ntn_bias)
        #Final FC
        feature_count = (tensor_neurons+self.bins) if self.histogram else self.tensor_neurons
        self.fc1 = torch.nn.Linear(feature_count, self.bottle_neck_neurons)
        self.fc2 = torch.nn.Linear(self.bottle_neck_neurons, 1)

    def GNN (self, data):
        """
        """
        x, edge_index = data.x, data.edge_index
        for i in range(self.gnn_layers-1):
            x = F.relu(self.convs[i](x, edge_index))
            x = F.dropout(x, p=self.dropout)
        x = self.convs[-1](x, edge_index)

        return x

    def forward(self, data_pair):
        """
          batch_adj is unused
        """
        c_graphs = data_pair.target_data()
        q_graphs = data_pair.subgraph_data()
        #q_graphs,c_graphs = zip(*batch_data)
        #a,b = zip(*batch_data_sizes)
        #qgraph_sizes = cudavar(self.av,torch.tensor(a))
        #cgraph_sizes = cudavar(self.av,torch.tensor(b))
        #query_batch = Batch.from_data_list(q_graphs)
        q_graphs_x = self.GNN(q_graphs)
        q, _ = to_dense_batch(q_graphs_x, q_graphs.batch)
        #query_gnode_embeds = [g.x for g in q_graphs.to_data_list()]
        #query_gnode_embeds = []
        #for i in torch.unique(q_graphs.batch):
        #    query_gnode_embeds.append(q_graphs.x[q_graphs.batch == i])
        #qgraph_sizes = torch.tensor([g.size(0) for g in query_gnode_embeds])
        #q = pad_sequence(query_gnode_embeds,batch_first=True)
        c_graphs_x = self.GNN(c_graphs)
        c, _ = to_dense_batch(c_graphs_x, c_graphs.batch)
        #corpus_gnode_embeds = [g.x for g in c_graphs.to_data_list()]
        #context = torch.tanh(torch.div(torch.sum(self.attention_layer(q),dim=1).T,qgraph_sizes).T)
        context = torch.tanh(pyg_nn.global_mean_pool(self.attention_layer(q_graphs_x), q_graphs.batch))
        sigmoid_scores = torch.sigmoid(q @ context.unsqueeze(2))
        e1 = (q.permute(0,2,1)@sigmoid_scores).squeeze()
        #c = pad_sequence(corpus_gnode_embeds,batch_first=True)
        context = torch.tanh(pyg_nn.global_mean_pool(self.attention_layer(c_graphs_x), c_graphs.batch))
        sigmoid_scores = torch.sigmoid(c @ context.unsqueeze(2))
        e2 = (c.permute(0,2,1)@sigmoid_scores).squeeze()
        
        scores = torch.nn.functional.relu(self.ntn_a(e1,e2) +self.ntn_b(torch.cat((e1,e2),dim=-1))+self.ntn_bias.squeeze())

        #TODO: Figure out how to tensorize this
        if self.histogram == True:
          h = torch.histc(q@c.permute(0,2,1),bins=self.bins)
          h = h/torch.sum(h)

          scores = torch.cat((scores, h),dim=1)
        
        preds = []
        scores = torch.nn.functional.relu(self.fc1(scores))
        score = torch.sigmoid(self.fc2(scores))
        #preds.append(score)
        #p = torch.stack(preds).squeeze()
        return score#p

    def training_step(self, batch, batch_idx):

        out = self(batch)
        #print(out)
        #print(out)
        loss = F.mse_loss(out.squeeze(), batch.y.float())

        pred = (out>0.5).squeeze(1)

        self.log("train_loss", loss)
        self.log("train acc", self.train_acc(pred, batch.y))
        return loss
    
    def training_epoch_end(self, out):
        self.log("total train acc", self.train_acc.compute(), prog_bar=True)
        self.train_acc.reset()

    def validation_step(self, batch, batch_idx):
        out = self(batch)
        pred = (out>0.5)
        self.log("val acc", self.val_acc(pred.squeeze(), batch.y))
    
    def validation_epoch_end(self, out):
        self.log("total val acc", self.val_acc.compute(), prog_bar=True)
        self.val_acc.reset()

    def configure_optimizers(self):
        opt = torch.optim.Adam(self.parameters(), lr=self.lr)
        return opt