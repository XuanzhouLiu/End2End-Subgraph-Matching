# -*- coding: utf-8 -*-

import numpy as np
import networkx as nx
import logging
import random



class Generator(object):
    def __init__(self, sizes, size_prob=None):
        self.set_sizes(sizes, size_prob)

    def set_sizes(self, sizes, size_prob=None):
        self.sizes = sizes
        if sizes is not None:
            if size_prob is None:
                self.size_prob = np.ones(len(sizes)) / len(sizes)
            else:
                self.size_prob = size_prob

    def _get_size(self, size=None):
        if size is None:
            return np.random.choice(
                self.sizes, size=1, replace=True, p=self.size_prob
            )[0]
        else:
            return size
    
    def generate(self, size=None):
        raise NotImplementedError
    

class GraphGenerator(Generator):
    def __init__(self,sizes, size_prob=None, feat_type=None, feat_dim = 0):
        super(GraphGenerator, self).__init__(sizes, size_prob)
        self.set_features(feat_type, feat_dim)
        
    def set_features(self, feat_type, feat_dim):
        self.feat_type = feat_type
        self.feat_dim = feat_dim
        
    def _generate_graph(self, size):
        return nx.Graph()
        
    def _generate_feat(self, size):
        if self.feat_type == None:
            return np.zeros([size,1]).astype(np.float32)
        elif self.feat_type == "random":
            return np.random.randn(size,self.feat_dim).astype(np.float32)
        elif self.feat_type == "ones":
            return np.ones([size,self.feat_dim]).astype(np.float32)
        return np.zeros(size,1).astype(np.float32)
    
    def generate(self, size=None):
        size = self._get_size(size)
        graph = self._generate_graph(size)
        feat = self._generate_feat(size)
        
        feat_dict = {i:feat[i] for i in range(size)}
        nx.set_node_attributes(graph, feat_dict, name = "feature")
        
        return graph

class ERGenerator(GraphGenerator):
    def __init__(self, sizes, p_alpha=1.3, **kwargs):
        super(ERGenerator, self).__init__(sizes, **kwargs)
        self.p_alpha = p_alpha

    def _generate_graph(self, size):
        num_nodes = size
        # p follows beta distribution with mean = log2(num_graphs) / num_graphs
        alpha = self.p_alpha
        mean = np.log2(num_nodes) / num_nodes
        beta = alpha / mean - alpha
        p = np.random.beta(alpha, beta)
        graph = nx.gnp_random_graph(num_nodes, p)

        while not nx.is_connected(graph):
            p = np.random.beta(alpha, beta)
            graph = nx.gnp_random_graph(num_nodes, p)
        #logging.debug('Generated {}-node E-R graphs with average p: {}'.format(
        #       num_nodes, mean))
        return graph

class WSGenerator(GraphGenerator):
    def __init__(self, sizes, density_alpha=1.3, 
            rewire_alpha=2, rewire_beta=2, **kwargs):
        super(WSGenerator, self).__init__(sizes, **kwargs)
        self.density_alpha = density_alpha
        self.rewire_alpha = rewire_alpha
        self.rewire_beta = rewire_beta
    
    def _generate_graph(self, size=None):
        num_nodes = self._get_size(size)
        curr_num_graphs = 0

        density_alpha = self.density_alpha
        density_mean = np.log2(num_nodes) / num_nodes
        density_beta = density_alpha / density_mean - density_alpha

        rewire_alpha = self.rewire_alpha
        rewire_beta = self.rewire_beta
        while curr_num_graphs < 1:
            k = int(np.random.beta(density_alpha, density_beta) * num_nodes)
            k = max(k, 2)
            p = np.random.beta(rewire_alpha, rewire_beta)
            try:
                graph = nx.connected_watts_strogatz_graph(num_nodes, k, p)
                curr_num_graphs += 1
            except:
                pass
        logging.debug('Generated {}-node W-S graph with average density: {}'.format(
                num_nodes, density_mean))
        return graph


class SubgraphGenerator(Generator):
    def __init__(self, sizes, size_prob=None):
        super().__init__(sizes, size_prob)
    
    def generate(self, graph, size=None, **kwargs):
        raise NotImplementedError


class SubTreeGenerator(SubgraphGenerator):
    def __init__(self, sizes, size_prob=None):
        super(SubTreeGenerator, self).__init__(sizes, size_prob)
        
    def generate(self, graph, size=None, **kwargs):
        size = self._get_size(size)
        if size < 1:
            size = np.round(len(graph.nodes)*size)
        
        node = random.choice(range(len(graph.nodes)))
        neigh = self.subgraph_tree(graph, node, size)
        subgraph = nx.subgraph(graph,neigh)
        subgraph = nx.convert_node_labels_to_integers(subgraph)
        return subgraph
    
    def subgraph_tree(self, graph, node, size):
        start_node = node
        neigh = [start_node]
        frontier = list(set(graph.neighbors(start_node)) - set(neigh))
        visited = set([start_node])
        while len(neigh) < size and frontier:
            new_node = random.choice(list(frontier))
            assert new_node not in neigh
            neigh.append(new_node)
            visited.add(new_node)
            frontier += list(graph.neighbors(new_node))
            frontier = [x for x in frontier if x not in visited]
        if len(neigh) == size:
            return neigh
        else:
            print("No big enough subgraph tree")
            return neigh

class SubERGenerator(SubgraphGenerator, ERGenerator):
    def __init__(self, sizes, p_alpha=1.3, **kwargs):
        SubgraphGenerator.__init__(self, sizes)
        ERGenerator.__init__(self, sizes, p_alpha, **kwargs)
    def generate(self, graph, size=None, **kwargs):
        feat_type = self.feat_type
        if "feat_type" in kwargs:
            feat_type = kwargs["feat_type"]
        feat_dim = 0
        node = random.choice(graph.nodes)
        if "feature" in node:
            feat_dim = node["feature"].size
        self.set_features(feat_type, feat_dim)
        
        return ERGenerator.generate(self)