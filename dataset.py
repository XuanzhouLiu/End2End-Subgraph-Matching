import random
import numpy as np

from pathlib import Path
from sklearn.utils import shuffle
import tqdm

import torch
from torch_geometric.data import InMemoryDataset
from torch_geometric.utils import from_networkx
from generator import ERGenerator, WSGenerator, SubTreeGenerator, SubERGenerator

from torch_geometric.data import Data
class PairData(Data):
    def __init__(self, data_s = None, data_t = None):
        super().__init__()
        self.t_nodes = 0
        self.s_nodes = 0
        if data_s is None:
            self.edge_index_s = torch.tensor([])
            self.x_s = torch.tensor([])
        else:
            self.edge_index_s = data_s.edge_index
            self.x_s = data_s.x
            if not data_s.x is None:
                self.s_nodes += len(data_s.x)
        
        if data_t is None:
            self.edge_index_t = None
            self.x_t = None
        else:
            self.edge_index_t = data_t.edge_index
            self.x_t = data_t.x
            if not data_s.x is None:
                self.t_nodes += len(data_t.x)
        
        self.num_nodes = self.s_nodes+self.t_nodes
        

    def __inc__(self, key, value, *args, **kwargs):
        if key == 'edge_index_s':
            return self.x_s.size(0)
        if key == 'edge_index_t':
            return self.x_t.size(0)
        else:
            return super().__inc__(key, value, *args, **kwargs)
    
    def target_data(self):
        data = Data(x = self.x_t, edge_index = self.edge_index_t)
        if hasattr(self, "x_t_batch"):
            data.batch = self.x_t_batch
        return data

    def subgraph_data(self):
        data = Data(x = self.x_s, edge_index = self.edge_index_s)
        if hasattr(self, "x_s_batch"):
            data.batch = self.x_s_batch
        return data

class SynDataset(InMemoryDataset):
    def __init__(self, root, size = 100, positive = 50, \
        graph_type = "ER", graph_sizes = 20,\
        pos_subgraph_type = "subtree", neg_subgraph_type = "ER", subgraph_sizes = 5, \
        feat_type = "random", feat_dim = 1, **kwargs):
        self.kargs = kwargs

        self.root = Path(root)
        if not self.root.exists():
            self.root.mkdir()

        self.size = size

        if not isinstance(graph_type, list):
            self.graph_type_list = [graph_type]
        else:
            self.graph_type_list = graph_type
        
        if not isinstance(graph_sizes, list):
            self.graph_sizes = [graph_sizes]
        else:
            self.graph_sizes = graph_sizes

        self.positive = positive
        if not isinstance(pos_subgraph_type, list):
            self.pos_subgraph_type_list = [pos_subgraph_type]
        else:
            self.pos_subgraph_type_list = pos_subgraph_type

        if not isinstance(neg_subgraph_type, list):
            self.neg_subgraph_type_list = [neg_subgraph_type]
        else:
            self.neg_subgraph_type_list = neg_subgraph_type

        if not isinstance(subgraph_sizes, list):
            self.subgraph_sizes = [subgraph_sizes]
        else:
            self.subgraph_sizes = subgraph_sizes
        

        self.feat_type = feat_type
        self.feat_dim = feat_dim

        super().__init__(root)
        self.data, self.slices = torch.load(self.processed_paths[0])


    @property
    def raw_file_names(self):
        return None

    @property
    def processed_file_names(self):
        if "name" in self.kargs:
            name = self.kargs["name"]
        else:
            name = "SynData-{}-{}{}-{}{}{}-{}{}.pt".format(self.size, \
            "_".join(self.graph_type_list), "_".join([str(i) for i in self.graph_sizes]), \
            "_".join(self.pos_subgraph_type_list), "_".join(self.neg_subgraph_type_list), "_".join([str(i) for i in self.subgraph_sizes]), \
            str(self.feat_type), self.feat_dim)
        return [name]

    def num_classes(self):
        return 2

    def process(self):
        print("Begin Generating Data")
        self.create_generator()

        data_list = []
        positive = self.positive
        if self.positive < 1:
            positive = np.round(self.size * self.positive)
        negative = self.size - positive
        sizes = [negative, positive]
        sub_generator_list = [self.neg_subgraph_generator_list, self.pos_subgraph_generator_list]
        for y in range(2):
            if y == 0:
                print("Generating Negative Samples")
            else:
                print("Generating Positive Samples")
            for j in tqdm.tqdm(range(sizes[y])):
                generator = random.choice(self.generator_list)
                sub_generator = random.choice(sub_generator_list[y])

                graph = generator.generate()
                subgraph = sub_generator.generate(graph)

                while subgraph is None:
                    graph = generator.generate()
                    subgraph = sub_generator.generate(graph)

                data_s = from_networkx(subgraph, ["feature"])
                data_t = from_networkx(graph, ["feature"])
                pair = PairData(data_s,data_t)
                pair.y = y
                data_list.append(pair)

        data, slices = self.collate(data_list)
        torch.save((data, slices), self.processed_paths[0])
        print("Done")

    def create_generator(self):
        self.generator_list = []
        for graph_type in self.graph_type_list:
            if graph_type == "ER":
                self.generator_list.append(ERGenerator(self.graph_sizes, feat_type = self.feat_type, feat_dim = self.feat_dim, **self.kargs))
            elif graph_type == "WS":
                self.generator_list.append(WSGenerator(self.graph_sizes, feat_type = self.feat_type, feat_dim = self.feat_dim, **self.kargs))
            else:
                raise RuntimeError("Graph Type {} Not Included".format(graph_type))

        self.pos_subgraph_generator_list = []
        for subgraph_type in self.pos_subgraph_type_list:
            if subgraph_type == "subtree":
                self.pos_subgraph_generator_list.append(SubTreeGenerator(self.subgraph_sizes, **self.kargs))
            else:
                raise RuntimeError("Positive Subgraph Type {} Not Included".format(subgraph_type))

        self.neg_subgraph_generator_list = []
        for subgraph_type in self.neg_subgraph_type_list:
            if subgraph_type == "ER":
                self.neg_subgraph_generator_list.append(SubERGenerator(self.subgraph_sizes, feat_type = self.feat_type, **self.kargs))
            else:
                raise RuntimeError("Negative Subgraph Type {} Not Included".format(subgraph_type))

if __name__ == "__main__":
    dataset = SynDataset(Path("Exp"), graph_type=["ER","WS"], feat_type = None)
    from torch_geometric.loader import DataLoader
    loader = DataLoader(dataset,batch_size = 4, follow_batch=['x_s', 'x_t'], shuffle = True)
    batch = next(iter(loader))
    print(batch)