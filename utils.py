import numpy as np
from torch_cluster import neighbor_sampler
import torch_geometric.utils as pyg_utils
from torch_geometric.data import Data
import torch
import networkx as nx


def Nomial(order):
    x = np.poly1d([1,0])
    return [x**i for i in range(order)]

def rooted_tree(data: Data, node, depth=3):
    num_nodes = data.x.size(0)
    if not isinstance(node,torch.Tensor):
        node = torch.tensor([node])

    tree_edge = []
    node_index = 0
    leaf_index = 1

    for i in range(depth):
        neighbors = []
        for n in node:
            mask = pyg_utils.index_to_mask(n, num_nodes)
            neighbor = data.edge_index[1][mask[data.edge_index[0]]]
            tree_edge.append(torch.tensor([[node_index]*len(neighbor),list(range(leaf_index, leaf_index+len(neighbor)))]))
            leaf_index += len(neighbor)
            node_index += 1
            neighbors.append(neighbor)
        
        node = torch.cat(neighbors)

    x = torch.ones(node_index,1)
    edge_index_1 = torch.cat(tree_edge, dim=1)
    edge_index_2 = edge_index_1[[1,0]]
    edge_index = torch.cat([edge_index_1,edge_index_2],dim=1)
    return pyg_utils.to_networkx(Data(x=x,edge_index=edge_index))

edge_index = torch.tensor([[0,0,0,1,1,2,2,3],[1,2,3,0,2,0,1,0]])
x = torch.ones(4,1)
data = Data(x,edge_index)
g = rooted_tree(data, 0)
nx.draw(g)