# Copyright 2022 Twitter, Inc.
# SPDX-License-Identifier: Apache-2.0

import torch
import torch.nn as nn
import numpy as np
import os.path as osp
import torch_geometric.transforms as T
import networkx as nx

from typing import Optional, Callable, List, Union
from torch_sparse import SparseTensor, coalesce
from torch_geometric.data import InMemoryDataset, download_url, Data
from torch_geometric.utils.undirected import to_undirected
from torch_geometric.utils import remove_self_loops, from_networkx
from utils.classic import Planetoid
from definitions import ROOT_DIR
from scipy.stats import invwishart


class Actor(InMemoryDataset):
    r"""
    Code adapted from https://github.com/pyg-team/pytorch_geometric/blob/2.0.4/torch_geometric/datasets/actor.py

    The actor-only induced subgraph of the film-director-actor-writer
    network used in the
    `"Geom-GCN: Geometric Graph Convolutional Networks"
    <https://openreview.net/forum?id=S1e2agrFvS>`_ paper.
    Each node corresponds to an actor, and the edge between two nodes denotes
    co-occurrence on the same Wikipedia page.
    Node features correspond to some keywords in the Wikipedia pages.
    The task is to classify the nodes into five categories in term of words of
    actor's Wikipedia.

    Args:
        root (string): Root directory where the dataset should be saved.
        transform (callable, optional): A function/transform that takes in an
            :obj:`torch_geometric.data.Data` object and returns a transformed
            version. The data object will be transformed before every access.
            (default: :obj:`None`)
        pre_transform (callable, optional): A function/transform that takes in
            an :obj:`torch_geometric.data.Data` object and returns a
            transformed version. The data object will be transformed before
            being saved to disk. (default: :obj:`None`)
    """

    url = 'https://raw.githubusercontent.com/graphdml-uiuc-jlu/geom-gcn/master'

    def __init__(self, root: str, transform: Optional[Callable] = None,
                 pre_transform: Optional[Callable] = None):
        super().__init__(root, transform, pre_transform)
        self.data, self.slices = torch.load(self.processed_paths[0])

    @property
    def raw_file_names(self) -> List[str]:
        return ['out1_node_feature_label.txt', 'out1_graph_edges.txt'
                ] + [f'film_split_0.6_0.2_{i}.npz' for i in range(10)]

    @property
    def processed_file_names(self) -> str:
        return 'data.pt'

    def download(self):
        for f in self.raw_file_names[:2]:
            download_url(f'{self.url}/new_data/film/{f}', self.raw_dir)
        for f in self.raw_file_names[2:]:
            download_url(f'{self.url}/splits/{f}', self.raw_dir)

    def process(self):

        with open(self.raw_paths[0], 'r') as f:
            data = [x.split('\t') for x in f.read().split('\n')[1:-1]]

            rows, cols = [], []
            for n_id, col, _ in data:
                col = [int(x) for x in col.split(',')]
                rows += [int(n_id)] * len(col)
                cols += col
            x = SparseTensor(row=torch.tensor(rows), col=torch.tensor(cols))
            x = x.to_dense()

            y = torch.empty(len(data), dtype=torch.long)
            for n_id, _, label in data:
                y[int(n_id)] = int(label)

        with open(self.raw_paths[1], 'r') as f:
            data = f.read().split('\n')[1:-1]
            data = [[int(v) for v in r.split('\t')] for r in data]
            edge_index = torch.tensor(data, dtype=torch.long).t().contiguous()
            # Remove self-loops
            edge_index, _ = remove_self_loops(edge_index)
            # Make the graph undirected
            edge_index = to_undirected(edge_index)
            edge_index, _ = coalesce(edge_index, None, x.size(0), x.size(0))

        train_masks, val_masks, test_masks = [], [], []
        for f in self.raw_paths[2:]:
            tmp = np.load(f)
            train_masks += [torch.from_numpy(tmp['train_mask']).to(torch.bool)]
            val_masks += [torch.from_numpy(tmp['val_mask']).to(torch.bool)]
            test_masks += [torch.from_numpy(tmp['test_mask']).to(torch.bool)]
        train_mask = torch.stack(train_masks, dim=1)
        val_mask = torch.stack(val_masks, dim=1)
        test_mask = torch.stack(test_masks, dim=1)

        data = Data(x=x, edge_index=edge_index, y=y, train_mask=train_mask,
                    val_mask=val_mask, test_mask=test_mask)
        data = data if self.pre_transform is None else self.pre_transform(data)
        torch.save(self.collate([data]), self.processed_paths[0])


class WikipediaNetwork(InMemoryDataset):
    r"""
    Code adapted from https://github.com/pyg-team/pytorch_geometric/blob/2.0.4/torch_geometric/datasets/wikipedia_network.py

    The Wikipedia networks introduced in the
    `"Multi-scale Attributed Node Embedding"
    <https://arxiv.org/abs/1909.13021>`_ paper.
    Nodes represent web pages and edges represent hyperlinks between them.
    Node features represent several informative nouns in the Wikipedia pages.
    The task is to predict the average daily traffic of the web page.

    Args:
        root (string): Root directory where the dataset should be saved.
        name (string): The name of the dataset (:obj:`"chameleon"`,
            :obj:`"crocodile"`, :obj:`"squirrel"`).
        geom_gcn_preprocess (bool): If set to :obj:`True`, will load the
            pre-processed data as introduced in the `"Geom-GCN: Geometric
            Graph Convolutional Networks" <https://arxiv.org/abs/2002.05287>_`,
            in which the average monthly traffic of the web page is converted
            into five categories to predict.
            If set to :obj:`True`, the dataset :obj:`"crocodile"` is not
            available.
        transform (callable, optional): A function/transform that takes in an
            :obj:`torch_geometric.data.Data` object and returns a transformed
            version. The data object will be transformed before every access.
            (default: :obj:`None`)
        pre_transform (callable, optional): A function/transform that takes in
            an :obj:`torch_geometric.data.Data` object and returns a
            transformed version. The data object will be transformed before
            being saved to disk. (default: :obj:`None`)

    """

    def __init__(self, root: str, name: str,
                 transform: Optional[Callable] = None,
                 pre_transform: Optional[Callable] = None):
        self.name = name.lower()
        assert self.name in ['chameleon', 'squirrel']
        super().__init__(root, transform, pre_transform)
        self.data, self.slices = torch.load(self.processed_paths[0])

    @property
    def raw_dir(self) -> str:
        return osp.join(self.root, self.name, 'raw')

    @property
    def processed_dir(self) -> str:
        return osp.join(self.root, self.name, 'processed')

    @property
    def raw_file_names(self) -> Union[str, List[str]]:
        return ['out1_node_feature_label.txt', 'out1_graph_edges.txt']

    @property
    def processed_file_names(self) -> str:
        return 'data.pt'

    def download(self):
        pass

    def process(self):
        with open(self.raw_paths[0], 'r') as f:
            data = f.read().split('\n')[1:-1]
        x = [[float(v) for v in r.split('\t')[1].split(',')] for r in data]
        x = torch.tensor(x, dtype=torch.float)
        y = [int(r.split('\t')[2]) for r in data]
        y = torch.tensor(y, dtype=torch.long)

        with open(self.raw_paths[1], 'r') as f:
            data = f.read().split('\n')[1:-1]
            data = [[int(v) for v in r.split('\t')] for r in data]
        edge_index = torch.tensor(data, dtype=torch.long).t().contiguous()
        # Remove self-loops
        edge_index, _ = remove_self_loops(edge_index)
        # Make the graph undirected
        edge_index = to_undirected(edge_index)
        edge_index, _ = coalesce(edge_index, None, x.size(0), x.size(0))

        data = Data(x=x, edge_index=edge_index, y=y)

        if self.pre_transform is not None:
            data = self.pre_transform(data)

        torch.save(self.collate([data]), self.processed_paths[0])


class WebKB(InMemoryDataset):
    r"""
    Code adapted from https://github.com/pyg-team/pytorch_geometric/blob/2.0.4/torch_geometric/datasets/webkb.py

    The WebKB datasets used in the
    `"Geom-GCN: Geometric Graph Convolutional Networks"
    <https://openreview.net/forum?id=S1e2agrFvS>`_ paper.
    Nodes represent web pages and edges represent hyperlinks between them.
    Node features are the bag-of-words representation of web pages.
    The task is to classify the nodes into one of the five categories, student,
    project, course, staff, and faculty.
    Args:
        root (string): Root directory where the dataset should be saved.
        name (string): The name of the dataset (:obj:`"Cornell"`,
            :obj:`"Texas"` :obj:`"Washington"`, :obj:`"Wisconsin"`).
        transform (callable, optional): A function/transform that takes in an
            :obj:`torch_geometric.data.Data` object and returns a transformed
            version. The data object will be transformed before every access.
            (default: :obj:`None`)
        pre_transform (callable, optional): A function/transform that takes in
            an :obj:`torch_geometric.data.Data` object and returns a
            transformed version. The data object will be transformed before
            being saved to disk. (default: :obj:`None`)
    """

    url = ('https://raw.githubusercontent.com/graphdml-uiuc-jlu/geom-gcn/'
           '1c4c04f93fa6ada91976cda8d7577eec0e3e5cce/new_data')

    def __init__(self, root, name, transform=None, pre_transform=None):
        self.name = name.lower()
        assert self.name in ['cornell', 'texas', 'washington', 'wisconsin']

        super(WebKB, self).__init__(root, transform, pre_transform)
        self.data, self.slices = torch.load(self.processed_paths[0])

    @property
    def raw_dir(self):
        return osp.join(self.root, self.name, 'raw')

    @property
    def processed_dir(self):
        return osp.join(self.root, self.name, 'processed')

    @property
    def raw_file_names(self):
        return ['out1_node_feature_label.txt', 'out1_graph_edges.txt']

    @property
    def processed_file_names(self):
        return 'data.pt'

    def download(self):
        for name in self.raw_file_names:
            download_url(f'{self.url}/{self.name}/{name}', self.raw_dir)

    def process(self):
        with open(self.raw_paths[0], 'r') as f:
            data = f.read().split('\n')[1:-1]
            x = [[float(v) for v in r.split('\t')[1].split(',')] for r in data]
            x = torch.tensor(x, dtype=torch.float32)

            y = [int(r.split('\t')[2]) for r in data]
            y = torch.tensor(y, dtype=torch.long)

        with open(self.raw_paths[1], 'r') as f:
            data = f.read().split('\n')[1:-1]
            data = [[int(v) for v in r.split('\t')] for r in data]
            edge_index = torch.tensor(data, dtype=torch.long).t().contiguous()
            edge_index = to_undirected(edge_index)
            # We also remove self-loops in these datasets in order not to mess up with the Laplacian.
            edge_index, _ = remove_self_loops(edge_index)
            edge_index, _ = coalesce(edge_index, None, x.size(0), x.size(0))

        data = Data(x=x, edge_index=edge_index, y=y)
        data = data if self.pre_transform is None else self.pre_transform(data)
        torch.save(self.collate([data]), self.processed_paths[0])

    def __repr__(self):
        return '{}()'.format(self.name)

class GaussianGraph:
    def __init__(self, num_nodes, p, dist_dim, num_samples, samples_dim):
        self.num_nodes = num_nodes
        self.p = p
        self.dist_dim = dist_dim
        self.num_samples = num_samples
        self.samples_dim = samples_dim

    def generate_sample(self, generator):
        graph = self.generate_graph(generator)
        self.generate_features(graph)
        self.generate_labels(graph)
        self.resize_params(graph)
        dataset = from_networkx(graph)
        rns = T.RandomNodeSplit(num_test=0.2, num_val=0.2)
        dataset = rns(dataset)
        dataset.dist_dim = self.dist_dim
        dataset.num_samples = self.num_samples
        dataset.samples_dim = self.samples_dim
        #torch.save(self.collate([dataset]), self.processed_paths[0])
        return dataset
    
    def generate_graph(self, generator):
        if generator == 'erdos_renyi':
            graph = nx.erdos_renyi_graph(self.num_nodes, self.p)
        elif generator == 'barabasi_albert':
            graph = nx.barabasi_albert_graph(self.num_nodes, 25)
        elif generator == 'watts_strogatz':
            graph = nx.watts_strogatz_graph(self.num_nodes, 45, self.p)
        elif generator == 'random_geometric':
            graph = nx.random_geometric_graph(self.num_nodes, 0.3)
        elif generator == 'complete':
            graph = nx.complete_graph(self.num_nodes)
        #olhar loops

        assert nx.is_connected(graph)

        return graph

    def generate_features(self, graph):
        m = np.random.uniform(-1, 1, self.dist_dim)
        S = np.random.uniform(-1, 1, (self.dist_dim, self.dist_dim))
        S = S @ S.T

        assert np.all(np.linalg.eigvals(S) > 0)

        for n in range(graph.number_of_nodes()):
            mean = torch.tensor(np.random.multivariate_normal(m, S), dtype=torch.float64)
            cov = torch.tensor(invwishart.rvs(self.dist_dim, S), dtype=torch.float64) #+ 1e-6 * torch.eye(self.dist_dim, dtype=torch.float32)

            if self.dist_dim != 1:
                assert (torch.linalg.eigvals(cov).real >= 0).all()

            graph.nodes[n]['x'] = (mean, cov)

    def generate_labels(self, graph):
        C = 1
        for n in range(graph.number_of_nodes()):
            mean_y, cov_y = graph.nodes[n]['x'][0].clone(), graph.nodes[n]['x'][1].clone()
            neighbors_params = []
            KL_divs = []

            for neighbor in graph.neighbors(n):
                mean, cov = graph.nodes[neighbor]['x']
                neighbors_params.append((mean, cov, graph.degree(neighbor)))
                KL_divs.append(self.KL_div(mean_y, cov_y, mean, cov).item())

            #multiplicar v.a. por 1/sqrt(grau do no * grau do vizinho)
            KL_divs = [1 / KL_divs[i] for i in range(len(KL_divs))]
            m = max(KL_divs)
            KL_divs = [C * KL_divs[i] / m for i in range(len(KL_divs))]
            node_degree = graph.degree(n)

            mean_y = mean_y + sum([KL_divs[i] * neighbors_params[i][0] for i in range(len(KL_divs))])
            cov_y = cov_y + sum([KL_divs[i]**2 * neighbors_params[i][1] for i in range(len(KL_divs))])
            #mean_y += sum([C/(node_degree * neighbors_params[i][2])**0.5 * neighbors_params[i][0] for i in range(len(KL_divs))])
            #cov_y += sum([C**2/(node_degree * neighbors_params[i][2]) * neighbors_params[i][1] for i in range(len(KL_divs))])
            if self.samples_dim == 1:
                graph.nodes[n]['y'] = torch.distributions.normal.Normal(mean_y,cov_y).sample(torch.Size([self.num_samples])).flatten()
            else:
                graph.nodes[n]['y'] = torch.distributions.multivariate_normal.MultivariateNormal(mean_y,cov_y).sample(torch.Size([self.num_samples])).flatten()
    
    def KL_div(self, mean1, cov1, mean2, cov2):
        if self.dist_dim == 1:
            cov2_inv = 1/cov2
            return 0.5 * (cov2_inv * cov1 + (mean2 - mean1)**2 - 1 + torch.log(cov2) - torch.log(cov1))
        else:
            cov2_inv = torch.inverse(cov2)
            return 0.5 * (torch.trace(cov2_inv @ cov1) + (mean2 - mean1).t() @ cov2_inv @ (mean2 - mean1) - self.dist_dim + torch.logdet(cov2) - torch.logdet(cov1))
    
    def resize_params(self, graph):
        for n in range(graph.number_of_nodes()):
            mean, cov = graph.nodes[n]['x']
            graph.nodes[n]['x'] = torch.cat([mean, cov.flatten()])

def get_fixed_splits(data, dataset_name, seed):
    with np.load(f'splits/{dataset_name}_split_0.6_0.2_{seed}.npz') as splits_file:
        train_mask = splits_file['train_mask']
        val_mask = splits_file['val_mask']
        test_mask = splits_file['test_mask']

    data.train_mask = torch.tensor(train_mask, dtype=torch.bool)
    data.val_mask = torch.tensor(val_mask, dtype=torch.bool)
    data.test_mask = torch.tensor(test_mask, dtype=torch.bool)

    if dataset_name in {'cora', 'citeseer', 'pubmed'}:
        data.train_mask[data.non_valid_samples] = False
        data.test_mask[data.non_valid_samples] = False
        data.val_mask[data.non_valid_samples] = False
        print("Non zero masks", torch.count_nonzero(data.train_mask + data.val_mask + data.test_mask))
        print("Nodes", data.x.size(0))
        print("Non valid", len(data.non_valid_samples))
    else:
        assert torch.count_nonzero(data.train_mask + data.val_mask + data.test_mask) == data.x.size(0)

    return data

def get_dataset(name):
    data_root = osp.join(ROOT_DIR, 'datasets')
    if name in ['cornell', 'texas', 'wisconsin']:
        dataset = WebKB(root=data_root, name=name, transform=T.NormalizeFeatures())
    elif name in ['chameleon', 'squirrel']:
        dataset = WikipediaNetwork(root=data_root, name=name, transform=T.NormalizeFeatures())
    elif name == 'film':
        dataset = Actor(root=data_root, transform=T.NormalizeFeatures())
    elif name in ['cora', 'citeseer', 'pubmed']:
        dataset = Planetoid(root=data_root, name=name, transform=T.NormalizeFeatures())
    elif name in ['erdos_renyi']:
        dataset = GaussianGraph(num_nodes=200, p=0.5, dist_dim=2, num_samples=30, samples_dim=2).generate_sample('erdos_renyi')
        print(dataset)
    elif name in ['barabasi_albert']:
        dataset = GaussianGraph(num_nodes=200, p=0.5, dist_dim=2, num_samples=30, samples_dim=2).generate_sample('barabasi_albert')
        print(dataset)
    elif name in ['watts_strogatz']:
        dataset = GaussianGraph(num_nodes=200, p=0.5, dist_dim=2, num_samples=30, samples_dim=2).generate_sample('watts_strogatz')
        print(dataset)
    elif name in ['random_geometric']:
        dataset = GaussianGraph(num_nodes=200, p=0.5, dist_dim=2, num_samples=30, samples_dim=2).generate_sample('random_geometric')
        print(dataset)
    elif name in ['complete']:
        dataset = GaussianGraph(num_nodes=200, p=0.5, dist_dim=2, num_samples=30, samples_dim=2).generate_sample('complete')
        print(dataset)
    else:
        raise ValueError(f'dataset {name} not supported in dataloader')

    return dataset