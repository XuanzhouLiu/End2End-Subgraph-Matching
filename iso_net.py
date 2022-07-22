import torch
import torch_geometric.nn as pyg_nn
import torch.nn.functional as F
import torch.nn as nn
from torch.nn.utils.rnn import  pad_sequence 
from torch_geometric.data import Data, Batch
from torch_geometric.nn import global_add_pool
from torch_geometric.nn import MessagePassing

import pytorch_lightning as pl
import torchmetrics

class ISONET(pl.LightningModule):
  def __init__(self, input_dim, node_hidden_dim, input_edge_dim, edge_hidden_dim, prop_layers = 2, node_layers = 2, edge_layers = 2,\
    margin = 0.1, lr = 0.001, clf_lr = 0.001, **kwargs):
      """
      """
      super(ISONET, self).__init__()
      self.automatic_optimization = False

      self.config = kwargs
      self.input_dim = input_dim
      self.edge_dim = input_edge_dim

      self.node_hidden_dim = node_hidden_dim
      self.edge_hidden_dim = edge_hidden_dim

      self.node_layers = node_layers
      self.edge_layers = edge_layers
      self.prop_layers = prop_layers
      self.MAX_EDGES = 1000
      self.margin = 0.1
      self.lr = lr
      self.clf_lr = clf_lr

      self.train_acc = torchmetrics.Accuracy()
      self.val_acc = torchmetrics.Accuracy()
      
      self.build_masking_utility()
      self.build_layers()
      self.diagnostic_mode = False

      self.clf_model = nn.Sequential(nn.Linear(1, 2), nn.LogSoftmax(dim=-1))
      
  def build_masking_utility(self):
      self.max_set_size = self.MAX_EDGES
      #this mask pattern sets bottom last few rows to 0 based on padding needs
      self.graph_size_to_mask_map = [torch.cat((torch.tensor([1]).repeat(x,1).repeat(1,self.edge_hidden_dim), \
      torch.tensor([0]).repeat(self.max_set_size-x,1).repeat(1,self.edge_hidden_dim))) for x in range(0,self.max_set_size+1)]
      # Mask pattern sets top left (k)*(k) square to 1 inside arrays of size n*n. Rest elements are 0
      self.set_size_to_mask_map = [torch.cat((torch.repeat_interleave(torch.tensor([1,0]),torch.tensor([x,self.max_set_size-x])).repeat(x,1),
                            torch.repeat_interleave(torch.tensor([1,0]),torch.tensor([0,self.max_set_size])).repeat(self.max_set_size-x,1)))
                            for x in range(0,self.max_set_size+1)]

      
  def fetch_edge_counts(self, to_idx, from_idx, graph_idx, num_graphs):
      #HACK - since I'm not storing edge sizes of each graph (only storing node sizes)
      #and no. of nodes is not equal to no. of edges
      #so a hack to obtain no of edges in each graph from available info
      tt = global_add_pool(torch.ones(len(to_idx)).to(self.device), to_idx, len(graph_idx))
      tt1 = global_add_pool(torch.ones(len(from_idx)).to(self.device), from_idx, len(graph_idx))
      edge_counts = global_add_pool(tt, graph_idx, num_graphs)
      edge_counts1 = global_add_pool(tt1, graph_idx, num_graphs)
      assert(edge_counts == edge_counts1).all()
      assert(sum(edge_counts)== len(to_idx))
      return list(map(int,edge_counts.tolist()))

  def build_layers(self):

      self.encoder = GraphEncoder(self.input_dim, self.edge_dim, [self.node_hidden_dim]*self.node_layers, [self.edge_hidden_dim]*self.edge_layers)
      self.prop_layer = GraphPropLayer(self.node_hidden_dim, [self.edge_hidden_dim]*self.edge_layers, [self.node_hidden_dim]*self.node_layers)
      
      #NOTE:FILTERS_3 is 10 for now - hardcoded into config
      self.fc_transform1 = nn.Linear(self.edge_hidden_dim, self.edge_hidden_dim)
      self.relu1 = nn.ReLU()
      self.fc_transform2 = nn.Linear(self.edge_hidden_dim, self.edge_hidden_dim)
      
      #self.edge_score_fc = nn.Linear(self.prop_layer._message_net[-1].out_features, 1)
      
  def get_graph(self, batch):
      node_features = torch.cat([batch.x_t,batch.x_s], dim=0)
      if hasattr(batch, "edge_feature_t"):
        edge_features = torch.cat([batch.edge_feature_t,batch.edge_feature_s]).to(self.device)
      else:
        edge_features = torch.ones([batch.edge_index_t.size(1)+ batch.edge_index_s.size(1), self.edge_dim]).to(self.device)
      edge_index = torch.cat([batch.edge_index_t, batch.edge_index_s + len(batch.x_t)], dim=1)
      from_idx = edge_index[0]
      to_idx = edge_index[1]
      graph_idx = torch.cat([batch.x_t_batch, len(batch.t_nodes) + batch.x_s_batch])
      return node_features, edge_features, from_idx, to_idx, graph_idx    
  
  def criterion(self, pred, labels):
    pred = -pred
    pred[labels == 0] = torch.max(torch.tensor(0.0,device=pred.device), self.margin - pred)[labels == 0]
    return torch.sum(pred)

  def forward(self, batch):
      """
      """
      node_features, edge_features, from_idx, to_idx, graph_idx = self.get_graph(batch)
  
      #先编码节点和边信息
      node_features_enc, edge_features_enc = self.encoder(node_features, edge_features)

      #多次传播得到更新后的节点特征
      for i in range(self.prop_layers) :
          #node_feature_enc = self.prop_layer(node_features_enc, from_idx, to_idx,edge_features_enc)
          node_features_enc = self.prop_layer(node_features_enc, from_idx, to_idx, edge_features_enc)
      
      source_node_enc = node_features_enc[from_idx]
      dest_node_enc  = node_features_enc[to_idx]
      forward_edge_input = torch.cat((source_node_enc,dest_node_enc,edge_features_enc),dim=-1)
      backward_edge_input = torch.cat((dest_node_enc,source_node_enc,edge_features_enc),dim=-1)

      #再编码边信息
      forward_edge_msg = self.prop_layer._message_net(forward_edge_input)
      backward_edge_msg = self.prop_layer._reverse_message_net(backward_edge_input)
      edge_features_enc = forward_edge_msg + backward_edge_msg
      
      edge_counts  = self.fetch_edge_counts(to_idx,from_idx,graph_idx,2*batch.num_graphs)
      qgraph_edge_sizes = torch.tensor(edge_counts[0:len(batch.s_nodes)])
      cgraph_edge_sizes = torch.tensor(edge_counts[len(batch.s_nodes):])

      edge_feature_enc_split = torch.split(edge_features_enc, edge_counts, dim=0)
      edge_feature_enc_query = edge_feature_enc_split[0:len(batch.s_nodes)]
      edge_feature_enc_corpus = edge_feature_enc_split[len(batch.s_nodes):]  
      
      
      stacked_qedge_emb = torch.stack([F.pad(x, pad=(0,0,0,self.max_set_size-x.shape[0])) \
                                        for x in edge_feature_enc_query])
      stacked_cedge_emb = torch.stack([F.pad(x, pad=(0,0,0,self.max_set_size-x.shape[0])) \
                                        for x in edge_feature_enc_corpus])


      transformed_qedge_emb = self.fc_transform2(self.relu1(self.fc_transform1(stacked_qedge_emb)))
      transformed_cedge_emb = self.fc_transform2(self.relu1(self.fc_transform1(stacked_cedge_emb)))
      #由于padding了边，所以需要把这些边去掉
      qgraph_mask = torch.stack([self.graph_size_to_mask_map[i] for i in qgraph_edge_sizes])
      cgraph_mask = torch.stack([self.graph_size_to_mask_map[i] for i in cgraph_edge_sizes])
      masked_qedge_emb = torch.mul(qgraph_mask.to(transformed_qedge_emb.device),transformed_qedge_emb)
      masked_cedge_emb = torch.mul(cgraph_mask.to(transformed_qedge_emb.device),transformed_cedge_emb)
      #三维，batch,edge,dim
      sinkhorn_input = torch.matmul(masked_qedge_emb,masked_cedge_emb.permute(0,2,1))
      transport_plan = pytorch_sinkhorn_iters(sinkhorn_input)

      if self.diagnostic_mode:
          return transport_plan

      scores = -torch.sum(torch.maximum(stacked_qedge_emb - transport_plan@stacked_cedge_emb,\
            torch.tensor([0]).to(stacked_qedge_emb.device)),\
          dim=(1,2))
      
      return scores
  
  def training_step(self, batch, batch_idx):
    opt, clf_opt = self.optimizers()

    pred = self(batch)
      #Pairwise ranking loss
    #print(pred)
    #predPos = prediction[batch.y>0.5]
    #predNeg = prediction[batch.y<0.5]
    loss = self.criterion(pred, batch.y)#pairwise_ranking_loss_similarity(predPos.unsqueeze(1),predNeg.unsqueeze(1), 0.5)
      #losses = torch.nn.functional.mse_loss(target, prediction,reduction="sum")

    opt.zero_grad()
    self.manual_backward(loss)
    opt.step()

    
    pred = pred.detach()
    pred = self.clf_model(pred.unsqueeze(1))
    criterion = nn.NLLLoss()
    clf_loss = criterion(pred, batch.y)

    self.clf_model.zero_grad()
    self.manual_backward(clf_loss)
    clf_opt.step()
    
    self.log("loss", loss, prog_bar=True)
    self.log("clf loss", clf_loss, prog_bar=True)
    self.log("train acc", self.train_acc(pred, batch.y), prog_bar=True)
    

  def training_epoch_end(self, out):
    self.log("total train acc", self.train_acc.compute(), prog_bar=True)
    self.train_acc.reset()

  def validation_step(self, batch, batch_idx):
    pred = self(batch)
    pred = self.clf_model(pred.unsqueeze(1))
    self.log("val acc", self.val_acc(pred, batch.y), prog_bar=True)
    

  def validation_epoch_end(self, out):
    self.log("total val acc", self.val_acc.compute(), prog_bar=True)
    self.val_acc.reset()

  def configure_optimizers(self):
    opt = torch.optim.Adam(self.parameters(), lr=self.lr)
    clf_opt = torch.optim.Adam(self.clf_model.parameters(), lr=self.clf_lr)
    return opt, clf_opt

def pairwise_ranking_loss_similarity(predPos, predNeg, margin):
    
    n_1 = predPos.shape[0]
    n_2 = predNeg.shape[0]
    dim = predPos.shape[1]

    expanded_1 = predPos.unsqueeze(1).expand(n_1, n_2, dim)
    expanded_2 = predNeg.unsqueeze(0).expand(n_1, n_2, dim)
    ell = margin + expanded_2 - expanded_1
    hinge = torch.nn.ReLU()
    loss = hinge(ell)
    sum_loss =  torch.sum(loss,dim= [0, 1])
    return sum_loss/(n_1*n_2)

class GraphEncoder(nn.Module):
    """Encoder module that projects node and edge features to some embeddings."""

    def __init__(self,
                 node_feature_dim,
                 edge_feature_dim,
                 node_hidden_sizes=None,
                 edge_hidden_sizes=None,
                 name='graph-encoder'):
        """Constructor.

        Args:
          node_hidden_sizes: if provided should be a list of ints, hidden sizes of
            node encoder network, the last element is the size of the node outputs.
            If not provided, node features will pass through as is.
          edge_hidden_sizes: if provided should be a list of ints, hidden sizes of
            edge encoder network, the last element is the size of the edge outptus.
            If not provided, edge features will pass through as is.
          name: name of this module.
        """
        super(GraphEncoder, self).__init__()

        # this also handles the case of an empty list
        self._node_feature_dim = node_feature_dim
        self._edge_feature_dim = edge_feature_dim
        self._node_hidden_sizes = node_hidden_sizes
        self._edge_hidden_sizes = edge_hidden_sizes if edge_hidden_sizes else None
        self._build_model()

    def _build_model(self):
        layer = []
        layer.append(nn.Linear(self._node_feature_dim, self._node_hidden_sizes[0]))
        for i in range(1, len(self._node_hidden_sizes)):
            layer.append(nn.ReLU())
            layer.append(nn.Linear(self._node_hidden_sizes[i - 1], self._node_hidden_sizes[i]))
        self.MLP1 = nn.Sequential(*layer)

        if self._edge_hidden_sizes is not None:
            layer = []
            layer.append(nn.Linear(self._edge_feature_dim, self._edge_hidden_sizes[0]))
            for i in range(1, len(self._node_hidden_sizes)):
                layer.append(nn.ReLU())
                layer.append(nn.Linear(self._edge_hidden_sizes[i - 1], self._edge_hidden_sizes[i]))
            self.MLP2 = nn.Sequential(*layer)
        else:
            self.MLP2 = None

    def forward(self, node_features, edge_features=None):
        """Encode node and edge features.

        Args:
          node_features: [n_nodes, node_feat_dim] float tensor.
          edge_features: if provided, should be [n_edges, edge_feat_dim] float
            tensor.

        Returns:
          node_outputs: [n_nodes, node_embedding_dim] float tensor, node embeddings.
          edge_outputs: if edge_features is not None and edge_hidden_sizes is not
            None, this is [n_edges, edge_embedding_dim] float tensor, edge
            embeddings; otherwise just the input edge_features.
        """
        if self._node_hidden_sizes is None:
            node_outputs = node_features
        else:
            node_outputs = self.MLP1(node_features)
        if edge_features is None or self._edge_hidden_sizes is None:
            edge_outputs = edge_features
        else:
            edge_outputs = self.MLP2(edge_features)

        return node_outputs, edge_outputs

class ConvLayer(MessagePassing):
    def __init__(self, node_input_dim, node_hidden_dim, node_output_dim, edge_input_dim, edge_hidden_dim, edge_output_dim):
        super().__init__(aggr='sum') #  "Max" aggregation.
        self.edge_mlp = nn.Sequential(nn.Linear(edge_input_dim + 2*input_dim, 2*edge_dim),
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

class GraphPropLayer(pl.LightningModule):
    """Implementation of a graph propagation (message passing) layer."""

    def __init__(self,
                 node_state_dim,
                 edge_hidden_sizes,  # int
                 node_hidden_sizes,  # int
                 edge_net_init_scale=0.1,
                 node_update_type='residual',
                 use_reverse_direction=True,
                 reverse_dir_param_different=True,
                 layer_norm=False,
                 prop_type='embedding',
                 name='graph-net'):
        """Constructor.

        Args:
          node_state_dim: int, dimensionality of node states.
          edge_hidden_sizes: list of ints, hidden sizes for the edge message
            net, the last element in the list is the size of the message vectors.
          node_hidden_sizes: list of ints, hidden sizes for the node update
            net.
          edge_net_init_scale: initialization scale for the edge networks.  This
            is typically set to a small value such that the gradient does not blow
            up.
          node_update_type: type of node updates, one of {mlp, gru, residual}.
          use_reverse_direction: set to True to also propagate messages in the
            reverse direction.
          reverse_dir_param_different: set to True to have the messages computed
            using a different set of parameters than for the forward direction.
          layer_norm: set to True to use layer normalization in a few places.
          name: name of this module.
        """
        super(GraphPropLayer, self).__init__()

        self._node_state_dim = node_state_dim
        self._edge_hidden_sizes = edge_hidden_sizes[:]

        # output size is node_state_dim
        self._node_hidden_sizes = node_hidden_sizes[:] + [node_state_dim]
        self._edge_net_init_scale = edge_net_init_scale
        self._node_update_type = node_update_type

        self._use_reverse_direction = use_reverse_direction
        self._reverse_dir_param_different = reverse_dir_param_different

        self._layer_norm = layer_norm
        self._prop_type = prop_type
        self.build_model()

        if self._layer_norm:
            self.layer_norm1 = nn.LayerNorm()
            self.layer_norm2 = nn.LayerNorm()

    def build_model(self):
        layer = []
        layer.append(nn.Linear(self._edge_hidden_sizes[0] + 2*self._node_hidden_sizes[0], self._edge_hidden_sizes[0]))
        for i in range(1, len(self._edge_hidden_sizes)):
            layer.append(nn.ReLU())
            layer.append(nn.Linear(self._edge_hidden_sizes[i - 1], self._edge_hidden_sizes[i]))
        self._message_net = nn.Sequential(*layer)

        # optionally compute message vectors in the reverse direction
        if self._use_reverse_direction:
            if self._reverse_dir_param_different:
                layer = []
                layer.append(nn.Linear(self._edge_hidden_sizes[0] + 2*self._node_hidden_sizes[0], self._edge_hidden_sizes[0]))
                for i in range(1, len(self._edge_hidden_sizes)):
                    layer.append(nn.ReLU())
                    layer.append(nn.Linear(self._edge_hidden_sizes[i - 1], self._edge_hidden_sizes[i]))
                self._reverse_message_net = nn.Sequential(*layer)
            else:
                self._reverse_message_net = self._message_net

        if self._node_update_type == 'gru':
            if self._prop_type == 'embedding':
                self.GRU = nn.GRU(self._node_state_dim * 2, self._node_state_dim)
            elif self._prop_type == 'matching':
                self.GRU = nn.GRU(self._node_state_dim * 3, self._node_state_dim)
        else:
            layer = []
            if self._prop_type == 'embedding':
                layer.append(nn.Linear(self._node_state_dim * 2, self._node_hidden_sizes[0]))
            elif self._prop_type == 'matching':
                layer.append(nn.Linear(self._node_state_dim * 3, self._node_hidden_sizes[0]))
            for i in range(1, len(self._node_hidden_sizes)):
                layer.append(nn.ReLU())
                layer.append(nn.Linear(self._node_hidden_sizes[i - 1], self._node_hidden_sizes[i]))
            self.MLP = nn.Sequential(*layer)

    def _compute_aggregated_messages(
            self, node_states, from_idx, to_idx, edge_features=None):
        """Compute aggregated messages for each node.

        Args:
          node_states: [n_nodes, input_node_state_dim] float tensor, node states.
          from_idx: [n_edges] int tensor, from node indices for each edge.
          to_idx: [n_edges] int tensor, to node indices for each edge.
          edge_features: if not None, should be [n_edges, edge_embedding_dim]
            tensor, edge features.

        Returns:
          aggregated_messages: [n_nodes, aggregated_message_dim] float tensor, the
            aggregated messages for each node.
        """

        aggregated_messages = self.graph_prop_once(
            node_states,
            from_idx,
            to_idx,
            self._message_net,
            aggregation_module=None,
            edge_features=edge_features)

        # optionally compute message vectors in the reverse direction
        if self._use_reverse_direction:
            reverse_aggregated_messages = self.graph_prop_once(
                node_states,
                to_idx,
                from_idx,
                self._reverse_message_net,
                aggregation_module=None,
                edge_features=edge_features)

            aggregated_messages += reverse_aggregated_messages

        if self._layer_norm:
            aggregated_messages = self.layer_norm1(aggregated_messages)

        return aggregated_messages

    def _compute_node_update(self,
                             node_states,
                             node_state_inputs,
                             node_features=None):
        """Compute node updates.

        Args:
          node_states: [n_nodes, node_state_dim] float tensor, the input node
            states.
          node_state_inputs: a list of tensors used to compute node updates.  Each
            element tensor should have shape [n_nodes, feat_dim], where feat_dim can
            be different.  These tensors will be concatenated along the feature
            dimension.
          node_features: extra node features if provided, should be of size
            [n_nodes, extra_node_feat_dim] float tensor, can be used to implement
            different types of skip connections.

        Returns:
          new_node_states: [n_nodes, node_state_dim] float tensor, the new node
            state tensor.

        Raises:
          ValueError: if node update type is not supported.
        """
        if self._node_update_type in ('mlp', 'residual'):
            node_state_inputs.append(node_states)
        if node_features is not None:
            node_state_inputs.append(node_features)

        if len(node_state_inputs) == 1:
            node_state_inputs = node_state_inputs[0]
        else:
            node_state_inputs = torch.cat(node_state_inputs, dim=-1)

        if self._node_update_type == 'gru':
            node_state_inputs = torch.unsqueeze(node_state_inputs, 0)
            node_states = torch.unsqueeze(node_states, 0)
            _, new_node_states = self.GRU(node_state_inputs, node_states)
            new_node_states = torch.squeeze(new_node_states)
            return new_node_states
        else:
            mlp_output = self.MLP(node_state_inputs)
            if self._layer_norm:
                mlp_output = nn.self.layer_norm2(mlp_output)
            if self._node_update_type == 'mlp':
                return mlp_output
            elif self._node_update_type == 'residual':
                return node_states + mlp_output
            else:
                raise ValueError('Unknown node update type %s' % self._node_update_type)

    def forward(self,
                node_states,
                from_idx,
                to_idx,
                edge_features=None,
                node_features=None):
        """Run one propagation step.

        Args:
          node_states: [n_nodes, input_node_state_dim] float tensor, node states.
          from_idx: [n_edges] int tensor, from node indices for each edge.
          to_idx: [n_edges] int tensor, to node indices for each edge.
          edge_features: if not None, should be [n_edges, edge_embedding_dim]
            tensor, edge features.
          node_features: extra node features if provided, should be of size
            [n_nodes, extra_node_feat_dim] float tensor, can be used to implement
            different types of skip connections.

        Returns:
          node_states: [n_nodes, node_state_dim] float tensor, new node states.
        """
        aggregated_messages = self._compute_aggregated_messages(
            node_states, from_idx, to_idx, edge_features=edge_features)

        return self._compute_node_update(node_states,
                                         [aggregated_messages],
                                         node_features=node_features)

    def graph_prop_once(self, node_states,
                        from_idx,
                        to_idx,
                        message_net,
                        aggregation_module=None,
                        edge_features=None):
        """One round of propagation (message passing) in a graph.

        Args:
          node_states: [n_nodes, node_state_dim] float tensor, node state vectors, one
            row for each node.
          from_idx: [n_edges] int tensor, index of the from nodes.
          to_idx: [n_edges] int tensor, index of the to nodes.
          message_net: a network that maps concatenated edge inputs to message
            vectors.
          aggregation_module: a module that aggregates messages on edges to aggregated
            messages for each node.  Should be a callable and can be called like the
            following,
            `aggregated_messages = aggregation_module(messages, to_idx, n_nodes)`,
            where messages is [n_edges, edge_message_dim] tensor, to_idx is the index
            of the to nodes, i.e. where each message should go to, and n_nodes is an
            int which is the number of nodes to aggregate into.
          edge_features: if provided, should be a [n_edges, edge_feature_dim] float
            tensor, extra features for each edge.

        Returns:
          aggregated_messages: an [n_nodes, edge_message_dim] float tensor, the
            aggregated messages, one row for each node.
        """
        from_states = node_states[from_idx]
        to_states = node_states[to_idx]
        edge_inputs = [from_states, to_states]

        if edge_features is not None:
            edge_inputs.append(edge_features)

        edge_inputs = torch.cat(edge_inputs, dim=-1)
        messages = message_net(edge_inputs)

        tensor = global_add_pool(messages, to_idx, node_states.shape[0])
        return tensor


def pytorch_sinkhorn_iters(log_alpha,temp=0.1,noise_factor=1.0, n_iters = 20):
    batch_size = log_alpha.size()[0]
    n = log_alpha.size()[1]
    log_alpha = log_alpha.view(-1, n, n)
    noise = pytorch_sample_gumbel([batch_size, n, n])*noise_factor
    log_alpha = log_alpha + noise.to(log_alpha.device)
    log_alpha = torch.div(log_alpha,temp)

    for i in range(n_iters):
      log_alpha = log_alpha - (torch.logsumexp(log_alpha, dim=2, keepdim=True)).view(-1, n, 1)

      log_alpha = log_alpha - (torch.logsumexp(log_alpha, dim=1, keepdim=True)).view(-1, 1, n)
    return torch.exp(log_alpha)

def pytorch_sample_gumbel(shape, eps=1e-20):
  #Sample from Gumbel(0, 1)
  U = torch.rand(shape).float()
  return -torch.log(eps - torch.log(U + eps))
