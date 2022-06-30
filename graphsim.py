import torch
import torch_geometric.nn as pyg_nn
import torch.nn.functional as F


class SimGNN(torch.nn.Module):
    def __init__(self,gnn_layers, input_dim, hidden_dim, tensor_neurons, dropout=0):
        """
        """
        super(SimGNN, self).__init__()

        self.input_dim = input_dim
        self.tensor_neurons = tensor_neurons
        self.dropout = dropout
        self.gnn_layers = gnn_layers

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
        feature_count = (tensor_neurons+self.av.bins) if self.av.histogram else self.av.tensor_neurons
        self.fc1 = torch.nn.Linear(feature_count, self.av.bottle_neck_neurons)
        self.fc2 = torch.nn.Linear(self.av.bottle_neck_neurons, 1)

    def GNN (self, data):
        """
        """
        x, edge_index = data.x, data.edge_index
        for i in range(self.gnn_layers-1):
            x = F.relu(self.conv[i](x, edge_index))
            x = F.dropout(x, p=self.dropout)
        x = self.conv[-1](x, edge_index)

        return x

    def forward(self, datapair):
        """
          batch_adj is unused
        """
        c_graphs = data_pair.target_data()
        q_graphs = data_pair.subgraph_data()
        #q_graphs,c_graphs = zip(*batch_data)
        a,b = zip(*batch_data_sizes)
        qgraph_sizes = cudavar(self.av,torch.tensor(a))
        cgraph_sizes = cudavar(self.av,torch.tensor(b))
        query_batch = Batch.from_data_list(q_graphs)
        query_batch.x = self.GNN(query_batch)
        query_gnode_embeds = [g.x for g in query_batch.to_data_list()]
        
        corpus_batch = Batch.from_data_list(c_graphs)
        corpus_batch.x = self.GNN(corpus_batch)
        corpus_gnode_embeds = [g.x for g in corpus_batch.to_data_list()]

        preds = []
        q = pad_sequence(query_gnode_embeds,batch_first=True)
        context = torch.tanh(torch.div(torch.sum(self.attention_layer(q),dim=1).T,qgraph_sizes).T)
        sigmoid_scores = torch.sigmoid(q@context.unsqueeze(2))
        e1 = (q.permute(0,2,1)@sigmoid_scores).squeeze()

        c = pad_sequence(corpus_gnode_embeds,batch_first=True)
        context = torch.tanh(torch.div(torch.sum(self.attention_layer(c),dim=1).T,cgraph_sizes).T)
        sigmoid_scores = torch.sigmoid(c@context.unsqueeze(2))
        e2 = (c.permute(0,2,1)@sigmoid_scores).squeeze()
        
        scores = torch.nn.functional.relu(self.ntn_a(e1,e2) +self.ntn_b(torch.cat((e1,e2),dim=-1))+self.ntn_bias.squeeze())
        

        #TODO: Figure out how to tensorize this
        if self.av.histogram == True:
          h = torch.histc(q@c.permute(0,2,1),bins=self.av.bins)
          h = h/torch.sum(h)

          scores = torch.cat((scores, h),dim=1)

        scores = torch.nn.functional.relu(self.fc1(scores))
        score = torch.sigmoid(self.fc2(scores))
        preds.append(score)
        p = torch.stack(preds).squeeze()
        return p