# Copyright 2022 Twitter, Inc.
# SPDX-License-Identifier: Apache-2.0

import torch
import torch.nn.functional as F
import numpy as np

from typing import Tuple
from abc import abstractmethod
from torch import nn
from lib import laplace as lap
import ot


class SheafLearner(nn.Module):
    """Base model that learns a sheaf from the features and the graph structure."""
    def __init__(self):
        super(SheafLearner, self).__init__()
        self.L = None

    @abstractmethod
    def forward(self, x, edge_index):
        raise NotImplementedError()

    def set_L(self, weights):
        self.L = weights.clone().detach()


class LocalConcatSheafLearner(SheafLearner):
    """Learns a sheaf by concatenating the local node features and passing them through a linear layer + activation."""

    def __init__(self, in_channels: int, out_shape: Tuple[int, ...], sheaf_act="tanh"):
        super(LocalConcatSheafLearner, self).__init__()
        assert len(out_shape) in [1, 2]
        self.out_shape = out_shape
        self.linear1 = torch.nn.Linear(in_channels*2, int(np.prod(out_shape)), bias=False, dtype=torch.float64)

        if sheaf_act == 'id':
            self.act = lambda x: x
        elif sheaf_act == 'tanh':
            self.act = torch.tanh
        elif sheaf_act == 'elu':
            self.act = F.elu
        else:
            raise ValueError(f"Unsupported act {sheaf_act}")

    def forward(self, x, edge_index):
        row, col = edge_index
        x_row = torch.index_select(x, dim=0, index=row)
        x_col = torch.index_select(x, dim=0, index=col)
        maps = self.linear1(torch.cat([x_row, x_col], dim=1))
        maps = self.act(maps)

        # sign = maps.sign()
        # maps = maps.abs().clamp(0.05, 1.0) * sign

        if len(self.out_shape) == 2:
            return maps.view(-1, self.out_shape[0], self.out_shape[1])
        else:
            return maps.view(-1, self.out_shape[0])


class LocalConcatSheafLearnerVariant(SheafLearner):
    """Learns a sheaf by concatenating the local node features and passing them through a linear layer + activation."""

    def __init__(self, d: int, hidden_channels: int, out_shape: Tuple[int, ...], sheaf_act="tanh"):
        super(LocalConcatSheafLearnerVariant, self).__init__()
        assert len(out_shape) in [1, 2]
        self.out_shape = out_shape
        self.d = d
        self.hidden_channels = hidden_channels
        self.linear1 = torch.nn.Linear(hidden_channels * 2, int(np.prod(out_shape)), bias=False, dtype=torch.float64)
        # self.linear2 = torch.nn.Linear(self.d, 1, bias=False)

        # std1 = 1.414 * math.sqrt(2. / (hidden_channels * 2 + 1))
        # std2 = 1.414 * math.sqrt(2. / (d + 1))
        #
        # nn.init.normal_(self.linear1.weight, 0.0, std1)
        # nn.init.normal_(self.linear2.weight, 0.0, std2)

        if sheaf_act == 'id':
            self.act = lambda x: x
        elif sheaf_act == 'tanh':
            self.act = torch.tanh
        elif sheaf_act == 'elu':
            self.act = F.elu
        else:
            raise ValueError(f"Unsupported act {sheaf_act}")

    def forward(self, x, edge_index):
        row, col = edge_index

        x_row = torch.index_select(x, dim=0, index=row)
        x_col = torch.index_select(x, dim=0, index=col)
        x_cat = torch.cat([x_row, x_col], dim=-1)
        x_cat = x_cat.reshape(-1, self.d, self.hidden_channels * 2).sum(dim=1)

        x_cat = self.linear1(x_cat)

        # x_cat = x_cat.t().reshape(-1, self.d)
        # x_cat = self.linear2(x_cat)
        # x_cat = x_cat.reshape(-1, edge_index.size(1)).t()

        maps = self.act(x_cat)

        if len(self.out_shape) == 2:
            return maps.view(-1, self.out_shape[0], self.out_shape[1])
        else:
            return maps.view(-1, self.out_shape[0])

class LocalConcatGaussianSheafLearner(SheafLearner):
    """Learns a sheaf by concatenating the local node features and passing them through a linear layer + activation."""

    def __init__(self, in_channels: int, out_shape: Tuple[int, ...], rest_maps_mlp : Tuple[int, ...], sheaf_act="tanh"):
        super(LocalConcatGaussianSheafLearner, self).__init__()
        assert len(out_shape) in [1, 2]
        self.in_channels = in_channels
        self.out_shape = out_shape
        self.linear1 = torch.nn.Linear(in_channels*2, int(np.prod(out_shape)), bias=False)

        mlp_num_layers = rest_maps_mlp[0]
        mlp_hidden_channels = rest_maps_mlp[1]

        mlp_layers = [nn.Linear(in_channels * 2, mlp_hidden_channels, dtype=torch.float64)]
        for i in range(mlp_num_layers-1):
            mlp_layers.append(nn.ReLU())
            mlp_layers.append(nn.Linear(mlp_hidden_channels, mlp_hidden_channels, dtype=torch.float64))
            
        mlp_layers.append(nn.ReLU())
        mlp_layers.append(nn.Linear(mlp_hidden_channels, int(np.prod(out_shape)), dtype=torch.float64))
        #self.linear1 = torch.nn.Linear(hidden_channels * 4, int(np.prod(out_shape)), bias=False, dtype=torch.float64)
        self.mlp = nn.Sequential(*mlp_layers)

        if sheaf_act == 'id':
            self.act = lambda x: x
        elif sheaf_act == 'tanh':
            self.act = torch.tanh
        elif sheaf_act == 'elu':
            self.act = F.elu
        else:
            raise ValueError(f"Unsupported act {sheaf_act}")

    def forward(self, x, edge_index):
        row, col = edge_index
        row, col = edge_index
        x_mu = x[:, :self.in_channels]
        x_sig = x[:, self.in_channels:]

        x_row = torch.index_select(x_mu, dim=0, index=row)
        x_col = torch.index_select(x_mu, dim=0, index=col)
        x_cat = torch.cat([x_row, x_col], dim=-1)
        x_cat = x_cat.reshape(-1, self.in_channels * 2)

        x_sig_row = torch.index_select(x_sig, dim=0, index=row)
        x_sig_col = torch.index_select(x_sig, dim=0, index=col)
        x_sig_cat = torch.cat([x_sig_row, x_sig_col], dim=-1)
        x_sig_cat = x_sig_cat.reshape(-1, self.in_channels * 2)
        maps = self.mlp(torch.cat([x_row, x_col], dim=1))
        maps = self.act(maps)

        # sign = maps.sign()
        # maps = maps.abs().clamp(0.05, 1.0) * sign

        if len(self.out_shape) == 2:
            return maps.view(-1, self.out_shape[0], self.out_shape[1])
        else:
            return maps.view(-1, self.out_shape[0])

class LocalConcatGaussianSheafLearnerVariant(SheafLearner):
    """Learns a sheaf by concatenating the local node features and passing them through a linear layer + activation."""

    def __init__(self, d: int, hidden_channels: int, out_shape: Tuple[int, ...], rest_maps_mlp : Tuple[int, ...], sheaf_act="tanh"):
        super(LocalConcatGaussianSheafLearnerVariant, self).__init__()
        assert len(out_shape) in [1, 2]
        self.out_shape = out_shape
        self.d = d
        self.hidden_channels = hidden_channels

        mlp_num_layers = rest_maps_mlp[0]
        mlp_hidden_channels = rest_maps_mlp[1]

        mlp_layers = [nn.Linear(self.hidden_channels * 4, mlp_hidden_channels, dtype=torch.float64)]
        for i in range(mlp_num_layers-1):
            mlp_layers.append(nn.ReLU())
            mlp_layers.append(nn.Linear(mlp_hidden_channels, mlp_hidden_channels, dtype=torch.float64))
            
        mlp_layers.append(nn.ReLU())
        mlp_layers.append(nn.Linear(mlp_hidden_channels, int(np.prod(out_shape)), dtype=torch.float64))
        #self.linear1 = torch.nn.Linear(hidden_channels * 4, int(np.prod(out_shape)), bias=False, dtype=torch.float64)
        self.mlp = nn.Sequential(*mlp_layers)
        # self.linear2 = torch.nn.Linear(self.d, 1, bias=False)

        # std1 = 1.414 * math.sqrt(2. / (hidden_channels * 2 + 1))
        # std2 = 1.414 * math.sqrt(2. / (d + 1))
        #
        # nn.init.normal_(self.linear1.weight, 0.0, std1)
        # nn.init.normal_(self.linear2.weight, 0.0, std2)

        if sheaf_act == 'id':
            self.act = lambda x: x
        elif sheaf_act == 'tanh':
            self.act = torch.tanh
        elif sheaf_act == 'elu':
            self.act = F.elu
        else:
            raise ValueError(f"Unsupported act {sheaf_act}")

    def forward(self, x, edge_index):
        row, col = edge_index
        x_mu = x[:, :self.hidden_channels * self.d]
        x_sig = x[:, self.hidden_channels * self.d:]

        x_row = torch.index_select(x_mu, dim=0, index=row)
        x_col = torch.index_select(x_mu, dim=0, index=col)
        x_cat = torch.cat([x_row, x_col], dim=-1)
        x_cat = x_cat.reshape(-1, self.d, self.hidden_channels * 2).sum(dim=1)

        x_sig_row = torch.index_select(x_sig, dim=0, index=row)
        x_sig_col = torch.index_select(x_sig, dim=0, index=col)
        x_sig_cat = torch.cat([x_sig_row, x_sig_col], dim=-1)
        x_sig_cat = x_sig_cat.reshape(-1, self.d**2, self.hidden_channels * 2).sum(dim=1)

        #print(x_cat.shape, x_sig_cat.shape)

        x_cat = self.mlp(torch.cat([x_cat, x_sig_cat], dim=-1))
        #x_cat = self.linear1(torch.cat([x_cat, x_sig_cat], dim=-1))
        #x_cat = self.linear1(x_cat)

        # x_cat = x_cat.t().reshape(-1, self.d)
        # x_cat = self.linear2(x_cat)
        # x_cat = x_cat.reshape(-1, edge_index.size(1)).t()

        maps = self.act(x_cat)

        if len(self.out_shape) == 2:
            return maps.view(-1, self.out_shape[0], self.out_shape[1])
        else:
            return maps.view(-1, self.out_shape[0])

class AttentionSheafLearner(SheafLearner):

    def __init__(self, in_channels, d):
        super(AttentionSheafLearner, self).__init__()
        self.d = d
        self.linear1 = torch.nn.Linear(in_channels*2, d**2, bias=False)

    def forward(self, x, edge_index):
        row, col = edge_index
        x_row = torch.index_select(x, dim=0, index=row)
        x_col = torch.index_select(x, dim=0, index=col)
        maps = self.linear1(torch.cat([x_row, x_col], dim=1)).view(-1, self.d, self.d)

        id = torch.eye(self.d, device=edge_index.device, dtype=maps.dtype).unsqueeze(0)
        return id - torch.softmax(maps, dim=-1)


class EdgeWeightLearner(SheafLearner):
    """Learns a sheaf by concatenating the local node features and passing them through a linear layer + activation."""

    def __init__(self, in_channels: int, edge_index):
        super(EdgeWeightLearner, self).__init__()
        self.in_channels = in_channels
        self.linear1 = torch.nn.Linear(in_channels*2, 1, bias=False, dtype=torch.float64)
        self.full_left_right_idx, _ = lap.compute_left_right_map_index(edge_index, full_matrix=True)

    def forward(self, x, edge_index):
        _, full_right_idx = self.full_left_right_idx

        row, col = edge_index
        x_row = torch.index_select(x, dim=0, index=row)
        x_col = torch.index_select(x, dim=0, index=col)
        weights = self.linear1(torch.cat([x_row, x_col], dim=1))
        weights = torch.sigmoid(weights)

        edge_weights = weights * torch.index_select(weights, index=full_right_idx, dim=0)
        return edge_weights

    def update_edge_index(self, edge_index):
        self.full_left_right_idx, _ = lap.compute_left_right_map_index(edge_index, full_matrix=True)

class GaussianEdgeWeightLearner(SheafLearner):
    """Learns a sheaf by concatenating the local node features and passing them through a linear layer + activation."""

    def __init__(self, in_channels: int, rest_maps_mlp : Tuple[int, ...], edge_index, d):
        super(GaussianEdgeWeightLearner, self).__init__()
        self.in_channels = in_channels
        self.d = d
        self.linear1 = torch.nn.Linear(in_channels*self.d*4, 1, bias=False, dtype=torch.float64)
        self.full_left_right_idx, _ = lap.compute_left_right_map_index(edge_index, full_matrix=True)

        mlp_num_layers = rest_maps_mlp[0]
        mlp_hidden_channels = rest_maps_mlp[1]

        mlp_layers = [nn.Linear(in_channels * self.d * 4, mlp_hidden_channels, dtype=torch.float64)]
        for i in range(mlp_num_layers-1):
            mlp_layers.append(nn.ReLU())
            mlp_layers.append(nn.Linear(mlp_hidden_channels, mlp_hidden_channels, dtype=torch.float64))
            
        mlp_layers.append(nn.ReLU())
        mlp_layers.append(nn.Linear(mlp_hidden_channels, 1, dtype=torch.float64))
        #self.linear1 = torch.nn.Linear(hidden_channels * 4, int(np.prod(out_shape)), bias=False, dtype=torch.float64)
        self.mlp = nn.Sequential(*mlp_layers)

    def forward(self, x, edge_index):
        _, full_right_idx = self.full_left_right_idx

        row, col = edge_index
        x_mu = x[:, :self.in_channels * self.d]
        x_sig = x[:, self.in_channels * self.d:]

        x_row = torch.index_select(x_mu, dim=0, index=row)
        x_col = torch.index_select(x_mu, dim=0, index=col)
        x_cat = torch.cat([x_row, x_col], dim=-1)
        x_cat = x_cat.reshape(-1, self.d * self.in_channels * 2)

        x_sig_row = torch.index_select(x_sig, dim=0, index=row)
        x_sig_col = torch.index_select(x_sig, dim=0, index=col)
        x_sig_cat = torch.cat([x_sig_row, x_sig_col], dim=-1)
        x_sig_cat = x_sig_cat.reshape(-1, self.d, self.d * self.in_channels * 2).sum(dim=1)

        weights = self.mlp(torch.cat([x_cat, x_sig_cat], dim=1))
        #weights = self.linear1(torch.cat([x_cat, x_sig_cat], dim=1))
        weights = torch.sigmoid(weights)

        # row, col = edge_index
        # x_row = torch.index_select(x, dim=0, index=row)
        # x_col = torch.index_select(x, dim=0, index=col)
        # weights = self.linear1(torch.cat([x_row, x_col], dim=1))
        # weights = torch.sigmoid(weights)

        edge_weights = weights * torch.index_select(weights, index=full_right_idx, dim=0)
        return edge_weights

    def update_edge_index(self, edge_index):
        self.full_left_right_idx, _ = lap.compute_left_right_map_index(edge_index, full_matrix=True)

class OptMapsSheafLearner(SheafLearner):
    """Learns a sheaf by concatenating the local node features and passing them through a linear layer + activation."""

    def __init__(self, d: int, hidden_channels: int, out_shape: Tuple[int, ...], rest_maps_mlp : Tuple[int, ...], sheaf_act="tanh"):
        super(OptMapsSheafLearner, self).__init__()
        assert len(out_shape) in [1, 2]
        self.out_shape = out_shape
        self.d = d
        self.hidden_channels = hidden_channels

        mlp_num_layers = rest_maps_mlp[0]
        mlp_hidden_channels = rest_maps_mlp[1]

        mlp_layers = [nn.Linear(self.d ** 2 + self.d, mlp_hidden_channels, dtype=torch.float64)]
        for i in range(mlp_num_layers-1):
            mlp_layers.append(nn.ReLU())
            mlp_layers.append(nn.Linear(mlp_hidden_channels, mlp_hidden_channels, dtype=torch.float64))
            
        mlp_layers.append(nn.ReLU())
        mlp_layers.append(nn.Linear(mlp_hidden_channels, int(np.prod(out_shape)), dtype=torch.float64))
        #self.linear1 = torch.nn.Linear(hidden_channels * 4, int(np.prod(out_shape)), bias=False, dtype=torch.float64)
        self.mlp = nn.Sequential(*mlp_layers)
        # self.linear2 = torch.nn.Linear(self.d, 1, bias=False)

        # std1 = 1.414 * math.sqrt(2. / (hidden_channels * 2 + 1))
        # std2 = 1.414 * math.sqrt(2. / (d + 1))
        #
        # nn.init.normal_(self.linear1.weight, 0.0, std1)
        # nn.init.normal_(self.linear2.weight, 0.0, std2)

        if sheaf_act == 'id':
            self.act = lambda x: x
        elif sheaf_act == 'tanh':
            self.act = torch.tanh
        elif sheaf_act == 'elu':
            self.act = F.elu
        else:
            raise ValueError(f"Unsupported act {sheaf_act}")

    def forward(self, x, edge_index):
        row, col = edge_index
        x_mu, x_sig = x
        opt_maps = []
        biases = []
        
        x_mu_row = torch.index_select(x_mu, dim=1, index=row).sum(dim=0)
        x_mu_col = torch.index_select(x_mu, dim=1, index=col).sum(dim=0)
        #x_mu_cat = torch.cat([x_mu_row, x_mu_col], dim=-1).view(-1, self.d * 2)

        x_sig_row = torch.index_select(x_sig, dim=1, index=row).sum(dim=0)
        x_sig_col = torch.index_select(x_sig, dim=1, index=col).sum(dim=0)
        for i in range(x_sig_row.shape[0]):
            mu_1, mu_2 = x_mu_row[i].view(self.d), x_mu_col[i].view(self.d)
            opt = ot.gaussian.gaussian_gromov_wasserstein_mapping(mu_1, mu_2, x_sig_row[i], x_sig_col[i])
            opt_maps.append(opt[0])
            biases.append(opt[1])

        opt_maps = torch.stack(opt_maps).view(-1, self.d**2)
        biases = torch.stack(biases)

        maps = torch.cat([biases, opt_maps], dim=-1)
        #print(x_cat.shape, x_sig_cat.shape)

        #x_cat = self.mlp(maps)
        #x_cat = self.linear1(torch.cat([x_cat, x_sig_cat], dim=-1))
        #x_cat = self.linear1(x_cat)

        # x_cat = x_cat.t().reshape(-1, self.d)
        # x_cat = self.linear2(x_cat)
        # x_cat = x_cat.reshape(-1, edge_index.size(1)).t()

        #maps = self.act(x_cat)
        maps = opt_maps

        if len(self.out_shape) == 2:
            return maps.view(-1, self.out_shape[0], self.out_shape[1])
        else:
            return maps.view(-1, self.out_shape[0])


class QuadraticFormSheafLearner(SheafLearner):
    """Learns a sheaf by concatenating the local node features and passing them through a linear layer + activation."""

    def __init__(self, in_channels: int, out_shape: Tuple[int]):
        super(QuadraticFormSheafLearner, self).__init__()
        assert len(out_shape) in [1, 2]
        self.out_shape = out_shape

        tensor = torch.eye(in_channels).unsqueeze(0).tile(int(np.prod(out_shape)), 1, 1)
        self.tensor = nn.Parameter(tensor)

    def forward(self, x, edge_index):
        row, col = edge_index
        x_row = torch.index_select(x, dim=0, index=row)
        x_col = torch.index_select(x, dim=0, index=col)
        maps = self.map_builder(torch.cat([x_row, x_col], dim=1))

        if len(self.out_shape) == 2:
            return torch.tanh(maps).view(-1, self.out_shape[0], self.out_shape[1])
        else:
            return torch.tanh(maps).view(-1, self.out_shape[0])

