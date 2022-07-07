import torch
import torch_geometric.nn as pyg_nn
import torch.nn.functional as F
import torch.nn as nn
from torch.nn.utils.rnn import  pad_sequence 
from torch_geometric.data import Data, Batch

import pytorch_lightning as pl
import torchmetrics




class ISONET(pl.LightningModule):
    def __init__(self, input_dim, **kwargs):
        """
        """
        super(ISONET, self).__init__()
        self.config = kwargs
        self.input_dim = input_dim
        self.build_masking_utility()
        self.build_layers()
        self.diagnostic_mode = False
        
    def build_masking_utility(self):
        self.max_set_size = self.av.MAX_EDGES
        #this mask pattern sets bottom last few rows to 0 based on padding needs
        self.graph_size_to_mask_map = [torch.cat((torch.tensor([1]).repeat(x,1).repeat(1,self.av.transform_dim), \
        torch.tensor([0]).repeat(self.max_set_size-x,1).repeat(1,self.av.transform_dim))) for x in range(0,self.max_set_size+1)]
        # Mask pattern sets top left (k)*(k) square to 1 inside arrays of size n*n. Rest elements are 0
        self.set_size_to_mask_map = [torch.cat((torch.repeat_interleave(torch.tensor([1,0]),torch.tensor([x,self.max_set_size-x])).repeat(x,1),
                             torch.repeat_interleave(torch.tensor([1,0]),torch.tensor([0,self.max_set_size])).repeat(self.max_set_size-x,1)))
                             for x in range(0,self.max_set_size+1)]

        
    def fetch_edge_counts(self,to_idx,from_idx,graph_idx,num_graphs):
        #HACK - since I'm not storing edge sizes of each graph (only storing node sizes)
        #and no. of nodes is not equal to no. of edges
        #so a hack to obtain no of edges in each graph from available info
        tt = unsorted_segment_sum(cudavar(self.av,torch.ones(len(to_idx))), to_idx, len(graph_idx))
        tt1 = unsorted_segment_sum(cudavar(self.av,torch.ones(len(from_idx))), from_idx, len(graph_idx))
        edge_counts = unsorted_segment_sum(tt, graph_idx, num_graphs)
        edge_counts1 = unsorted_segment_sum(tt1, graph_idx, num_graphs)
        assert(edge_counts == edge_counts1).all()
        assert(sum(edge_counts)== len(to_idx))
        return list(map(int,edge_counts.tolist()))

    def build_layers(self):

        self.encoder = GraphEncoder(**self.config['encoder'])
        prop_config = self.config['graph_embedding_net'].copy()
        prop_config.pop('n_prop_layers',None)
        prop_config.pop('share_prop_params',None)
        self.prop_layer = GraphPropLayer(**prop_config)
        
        #NOTE:FILTERS_3 is 10 for now - hardcoded into config
        self.fc_transform1 = nn.Linear(2*self.av.filters_3, self.av.transform_dim)
        self.relu1 = nn.ReLU()
        self.fc_transform2 = nn.Linear(self.av.transform_dim, self.av.transform_dim)
        
        #self.edge_score_fc = nn.Linear(self.prop_layer._message_net[-1].out_features, 1)
        
    def get_graph(self, batch):
        graph = batch
        node_features = cudavar(self.av,torch.from_numpy(graph.node_features))
        edge_features = cudavar(self.av,torch.from_numpy(graph.edge_features))
        from_idx = cudavar(self.av,torch.from_numpy(graph.from_idx).long())
        to_idx = cudavar(self.av,torch.from_numpy(graph.to_idx).long())
        graph_idx = cudavar(self.av,torch.from_numpy(graph.graph_idx).long())
        return node_features, edge_features, from_idx, to_idx, graph_idx    
    

    def forward(self, batch_data,batch_data_sizes,batch_adj):
        """
        """
        #a,b = zip(*batch_data_sizes)
        #qgraph_sizes = cudavar(self.av,torch.tensor(a))
        #cgraph_sizes = cudavar(self.av,torch.tensor(b))
        #A
        #a, b = zip(*batch_adj)
        #q_adj = torch.stack(a)
        #c_adj = torch.stack(b)
        

        node_features, edge_features, from_idx, to_idx, graph_idx = self.get_graph(batch_data)
    
        #先编码节点和边信息
        node_features_enc, edge_features_enc = self.encoder(node_features, edge_features)

        #多次传播得到更新后的节点特征
        for i in range(self.config['graph_embedding_net'] ['n_prop_layers']) :
            #node_feature_enc = self.prop_layer(node_features_enc, from_idx, to_idx,edge_features_enc)
            node_features_enc = self.prop_layer(node_features_enc, from_idx, to_idx,edge_features_enc)
        
        source_node_enc = node_features_enc[from_idx]
        dest_node_enc  = node_features_enc[to_idx]
        forward_edge_input = torch.cat((source_node_enc,dest_node_enc,edge_features_enc),dim=-1)
        backward_edge_input = torch.cat((dest_node_enc,source_node_enc,edge_features_enc),dim=-1)

        #再编码边信息
        forward_edge_msg = self.prop_layer._message_net(forward_edge_input)
        backward_edge_msg = self.prop_layer._reverse_message_net(backward_edge_input)
        edge_features_enc = forward_edge_msg + backward_edge_msg
        
        edge_counts  = self.fetch_edge_counts(to_idx,from_idx,graph_idx,2*len(batch_data_sizes))
        qgraph_edge_sizes = torch.tensor(edge_counts[0::2])
        cgraph_edge_sizes = torch.tensor(edge_counts[1::2])

        edge_feature_enc_split = torch.split(edge_features_enc, edge_counts, dim=0)
        edge_feature_enc_query = edge_feature_enc_split[0::2]
        edge_feature_enc_corpus = edge_feature_enc_split[1::2]  
        
        
        stacked_qedge_emb = torch.stack([F.pad(x, pad=(0,0,0,self.max_set_size-x.shape[0])) \
                                         for x in edge_feature_enc_query])
        stacked_cedge_emb = torch.stack([F.pad(x, pad=(0,0,0,self.max_set_size-x.shape[0])) \
                                         for x in edge_feature_enc_corpus])


        transformed_qedge_emb = self.fc_transform2(self.relu1(self.fc_transform1(stacked_qedge_emb)))
        transformed_cedge_emb = self.fc_transform2(self.relu1(self.fc_transform1(stacked_cedge_emb)))
        #由于padding了边，所以需要把这些边去掉
        qgraph_mask = cudavar(self.av,torch.stack([self.graph_size_to_mask_map[i] for i in qgraph_edge_sizes]))
        cgraph_mask = cudavar(self.av,torch.stack([self.graph_size_to_mask_map[i] for i in cgraph_edge_sizes]))
        masked_qedge_emb = torch.mul(qgraph_mask,transformed_qedge_emb)
        masked_cedge_emb = torch.mul(cgraph_mask,transformed_cedge_emb)
 
        sinkhorn_input = torch.matmul(masked_qedge_emb,masked_cedge_emb.permute(0,2,1))
        transport_plan = pytorch_sinkhorn_iters(self.av,sinkhorn_input)
 
        if self.diagnostic_mode:
            return transport_plan

        scores = -torch.sum(torch.maximum(stacked_qedge_emb - transport_plan@stacked_cedge_emb,\
              cudavar(self.av,torch.tensor([0]))),\
           dim=(1,2))
        
        return scores


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
            edge_outputs = self.MLP2(node_features)

        return node_outputs, edge_outputs

class GraphPropLayer(nn.Module):
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
        layer.append(nn.Linear(self._edge_hidden_sizes[0] + 1, self._edge_hidden_sizes[0]))
        for i in range(1, len(self._edge_hidden_sizes)):
            layer.append(nn.ReLU())
            layer.append(nn.Linear(self._edge_hidden_sizes[i - 1], self._edge_hidden_sizes[i]))
        self._message_net = nn.Sequential(*layer)

        # optionally compute message vectors in the reverse direction
        if self._use_reverse_direction:
            if self._reverse_dir_param_different:
                layer = []
                layer.append(nn.Linear(self._edge_hidden_sizes[0] + 1, self._edge_hidden_sizes[0]))
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
                layer.append(nn.Linear(self._node_state_dim * 3, self._node_hidden_sizes[0]))
            elif self._prop_type == 'matching':
                layer.append(nn.Linear(self._node_state_dim * 4, self._node_hidden_sizes[0]))
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

        aggregated_messages = graph_prop_once(
            node_states,
            from_idx,
            to_idx,
            self._message_net,
            aggregation_module=None,
            edge_features=edge_features)

        # optionally compute message vectors in the reverse direction
        if self._use_reverse_direction:
            reverse_aggregated_messages = graph_prop_once(
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


def graph_prop_once(node_states,
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

    tensor = unsorted_segment_sum(messages, to_idx, node_states.shape[0])
    return tensor

def unsorted_segment_sum(data, segment_ids, num_segments):
    """
    Computes the sum along segments of a tensor. Analogous to tf.unsorted_segment_sum.

    :param data: A tensor whose segments are to be summed.
    :param segment_ids: The segment indices tensor.
    :param num_segments: The number of segments.
    :return: A tensor of same data type as the data argument.
    """

    assert all([i in data.shape for i in segment_ids.shape]), "segment_ids.shape should be a prefix of data.shape"

    # Encourage to use the below code when a deterministic result is
    # needed (reproducibility). However, the code below is with low efficiency.

    # tensor = torch.zeros(num_segments, data.shape[1]).cuda()
    # for index in range(num_segments):
    #     tensor[index, :] = torch.sum(data[segment_ids == index, :], dim=0)
    # return tensor

    if len(segment_ids.shape) == 1:
        s = torch.prod(torch.tensor(data.shape[1:])).long().cuda()
        segment_ids = segment_ids.repeat_interleave(s).view(segment_ids.shape[0], *data.shape[1:])

    assert data.shape == segment_ids.shape, "data.shape and segment_ids.shape should be equal"

    shape = [num_segments] + list(data.shape[1:])
    tensor = torch.zeros(*shape).cuda().scatter_add(0, segment_ids, data)
    tensor = tensor.type(data.dtype)
    return tensor

def cudavar(av, x):
  """Adapt to CUDA or CUDA-less runs.  Annoying av arg may become
  useful for multi-GPU settings."""
  return x.cuda() if av.has_cuda and av.want_cuda else x

def pytorch_sinkhorn_iters(av, log_alpha,temp=0.1,noise_factor=1.0, n_iters = 20):
    noise_factor = av.NOISE_FACTOR
    batch_size = log_alpha.size()[0]
    n = log_alpha.size()[1]
    log_alpha = log_alpha.view(-1, n, n)
    noise = pytorch_sample_gumbel(av,[batch_size, n, n])*noise_factor
    log_alpha = log_alpha + noise
    log_alpha = torch.div(log_alpha,temp)

    for i in range(n_iters):
      log_alpha = log_alpha - (torch.logsumexp(log_alpha, dim=2, keepdim=True)).view(-1, n, 1)

      log_alpha = log_alpha - (torch.logsumexp(log_alpha, dim=1, keepdim=True)).view(-1, 1, n)
    return torch.exp(log_alpha)

def pytorch_sample_gumbel(av,shape, eps=1e-20):
  #Sample from Gumbel(0, 1)
  U = cudavar(av,torch.rand(shape).float())
  return -torch.log(eps - torch.log(U + eps))
