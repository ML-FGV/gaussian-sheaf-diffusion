import torch
import numpy as np
import torch.nn.functional as F
import torch_sparse

from torch import nn
from torch_geometric.nn import GCNConv
from models.sheaf_base import SheafDiffusion
from models import laplacian_builders as lb
from models.sheaf_models import LocalConcatSheafLearner, LocalConcatSheafLearnerVariant, GaussianEdgeWeightLearner, LocalConcatGaussianSheafLearner, LocalConcatGaussianSheafLearnerVariant, OptMapsSheafLearner

class GaussianMLP(SheafDiffusion):
    def __init__(self, edge_index, args):
        super(GaussianMLP, self).__init__(edge_index, args)

        self.mlp = self.set_mlps()

    def forward(self, x):
        samples = self.get_samples(x)
        samples = samples.transpose(2, 1)
        samples = self.mlp(samples).view(self.graph_size, -1)

        return samples
    
class GaussianGCN(SheafDiffusion):
    def __init__(self, edge_index, args):
        super(GaussianGCN, self).__init__(edge_index, args)

        self.convs = nn.ModuleList()
        if self.layers == 1:
            self.convs.append(GCNConv(self.dist_dim * self.num_samples, self.dist_dim * self.num_samples).type(torch.float64))
        else:
            self.convs.append(GCNConv(self.dist_dim * self.num_samples, self.hidden_channels).type(torch.float64))
            for i in range(1, self.layers - 1):
                self.convs.append(GCNConv(self.hidden_channels, self.hidden_channels).type(torch.float64))

        self.convs.append(GCNConv(self.hidden_channels, self.dist_dim * self.num_samples).type(torch.float64))

        self.mlp = self.set_mlps()

    def forward(self, x):
        samples = self.get_samples(x)
        samples = samples.view(self.graph_size, -1)
        for i in range(self.layers):
            samples = self.convs[i](samples, self.edge_index)

        samples = samples.view(self.graph_size, self.num_samples, self.dist_dim)
        samples = self.mlp(samples).view(self.graph_size, -1)

        return samples

class GaussianSheafDiffusion(SheafDiffusion):

    def __init__(self, edge_index, args):
        super(GaussianSheafDiffusion, self).__init__(edge_index, args)
        assert args['d'] > 1

        self.lin_right_weights = []
        self.lin_left_weights = []

        if self.right_weights:
            for i in range(self.layers):
                self.lin_right_weights.append(nn.Parameter(torch.zeros(self.hidden_channels, self.hidden_channels, device=self.device, dtype=torch.float64)))
                nn.init.orthogonal_(self.lin_right_weights[-1].data)
        if self.left_weights:
            for i in range(self.layers):
                self.lin_left_weights.append(nn.Parameter(torch.zeros(self.final_d, self.final_d, device=self.device, dtype=torch.float64)))
                nn.init.eye_(self.lin_left_weights[-1].data)
        
        self.weight_learners = nn.ModuleList()

        if self.rest_maps_type == 'diag':
            out_shape = (self.d,)
        elif self.rest_maps_type == 'orth':
            out_shape = (self.get_param_size(),)
        elif self.rest_maps_type == 'general':
            out_shape = (self.d, self.d)

        if self.sparse_learner:
            self.sheaf_learners = LocalConcatGaussianSheafLearnerVariant(
                self.final_d, self.hidden_channels, out_shape=out_shape, 
                rest_maps_mlp=(self.rest_maps_mlp_layers, self.rest_maps_mlp_hc), sheaf_act=self.sheaf_act)
        else:

            self.sheaf_learners = LocalConcatGaussianSheafLearner(
                    self.hidden_dim, out_shape=out_shape,
                    rest_maps_mlp=(self.rest_maps_mlp_layers, self.rest_maps_mlp_hc), sheaf_act=self.sheaf_act)

        if self.rest_maps_type == 'diag':
            self.laplacian_builder = lb.DiagLaplacianBuilder(
                self.graph_size, edge_index, d=self.d,
                normalised=self.normalised,
                deg_normalised=self.deg_normalised,
                add_hp=self.add_hp, add_lp=self.add_lp)
        elif self.rest_maps_type == 'orth': 
            self.laplacian_builder = lb.NormConnectionLaplacianBuilder(
                self.graph_size, edge_index, d=self.d,add_hp=self.add_hp,
                add_lp=self.add_lp, orth_map=self.orth_trans)
            
            if self.use_edge_weights:
                self.weight_learners.append(GaussianEdgeWeightLearner(
                    self.hidden_channels, (self.rest_maps_mlp_layers, self.rest_maps_mlp_hc),
                    edge_index, self.d))
        elif self.rest_maps_type == 'general':
            self.laplacian_builder = lb.GeneralLaplacianBuilder(
            self.graph_size, edge_index, d=self.d,
            add_lp=self.add_lp, add_hp=self.add_hp,
            normalised=self.normalised, deg_normalised=self.deg_normalised)

        self.epsilons = nn.Parameter(torch.zeros(self.final_d, 1, device=self.device, dtype=torch.float64))

        blocks1 = torch.zeros(self.d, self.dist_dim, dtype=torch.float64, device=self.device)
        blocks2 = torch.zeros(self.dist_dim, self.d, dtype=torch.float64, device=self.device)
        nn.init.orthogonal_(blocks1)
        nn.init.orthogonal_(blocks2)

        self.emb1 = nn.Parameter(torch.stack([blocks1 for _ in range(self.hidden_channels)]))
        self.emb2 = nn.Parameter(blocks2)

        vec1 = torch.zeros(self.d, dtype=torch.float64, device=self.device)
        vec2 = torch.zeros(self.dist_dim, dtype=torch.float64, device=self.device)

        self.vec1 = nn.Parameter(torch.stack([vec1 for _ in range(self.hidden_channels)]))
        self.vec2 = nn.Parameter(vec2)

        self.act = nn.Identity()

        self.mlp = self.set_mlps()
    
    def get_param_size(self):
        if self.orth_trans in ['matrix_exp', 'cayley']:
            return self.d * (self.d + 1) // 2
        else:
            return self.d * (self.d - 1) // 2
    
    def kronecker_decomposition(self, M, m, n, r, s):
        assert M.shape == (m * r, n * s) #aqui m=n=h e r=s=nd
        assert torch.any(M != 0)

        A, B = None, None
        blocks = []
        vec_blocks = []
        for i in range(m):
            for j in range(n):
                block = M[i*r:(i+1)*r, j*s:(j+1)*s]
                blocks.append(block)
                vec_blocks.append(block.flatten())

        print(torch.linalg.matrix_rank(torch.stack(vec_blocks, dim=1)).item())
        if torch.linalg.matrix_rank(torch.stack(vec_blocks, dim=1)).item() != 1:
            return "cannot decompose", "cannot decompose"

        for block in blocks:
            if torch.any(block != 0):
                B = block
                break
        
        a = []
        for i in range(m*n):
            #find a_i such that vec_blocks[i] = a_i * B.flatten()
            if torch.any(vec_blocks[i] != 0):
                for j in range(vec_blocks[i].shape[0]):
                    if vec_blocks[i][j] != 0:
                        a_i = vec_blocks[i][j] / B.flatten()[j]
                        break
            else:
                a_i = 0
            a.append(a_i)
        
        A = torch.tensor(a, device=self.device).view(m, n)
        assert torch.allclose(M, torch.kron(A,B))

        return A,B

    def forward(self, x):
        #x = F.dropout(x, p=self.input_dropout, training=self.training)
        x_mu = x[:, :self.dist_dim]
        x_sig = x[:, self.dist_dim:]
        x_sig = x_sig.view(-1, self.dist_dim, self.dist_dim)

        #Embedding dos parâmetros na dimensão do talo
        x_mu = self.emb1[:, None, ...] @ x_mu[..., None] #+ self.vec1[:, None, :, None]
        x_sig = self.emb1[:, None, ...] @ x_sig @ self.emb1[:, None, ...].transpose(3, 2) + 1e-6 * torch.eye(self.d, device=self.device)

        if self.use_act:
            x_mu = self.act(x_mu)
            x_sig = self.act(x_sig)

        #Criando o feixe/Laplaciano
        L = None
        edge_weights = None
        x_maps = torch.cat((x_mu.view(self.graph_size, -1), x_sig.view(self.graph_size, -1)), dim=1)
        x_maps = x_maps.reshape(self.graph_size, -1)
        maps = self.sheaf_learners(x_maps, self.edge_index)
        if self.use_edge_weights and self.rest_maps_type == 'orth':
            edge_weights = self.weight_learners[0](x_maps, self.edge_index)
            L, trans_maps = self.laplacian_builder(maps, edge_weights)
        else:
            L,trans_maps = self.laplacian_builder(maps)
        self.sheaf_learners.set_L(trans_maps)

        x_mu = x_mu.view(self.graph_size * self.final_d, -1)

        # x_mu0 = x_mu
        # x_sig0 = x_sig

        l = L
        for i in range(self.layers):
            if i != 0: 
                L = torch_sparse.spspmm(l[0], l[1], L[0], L[1], x_mu.size(0), x_mu.size(0), x_mu.size(0), coalesced=True)

            if self.left_weights:
                x_mu = x_mu.t().reshape(self.final_d, -1)
                x_mu = self.lin_left_weights[i] @ x_mu
                x_mu = x_mu.reshape(-1, self.graph_size * self.final_d).t()
                x_sig = self.lin_left_weights[i][None, None, ...] @ x_sig @ self.lin_left_weights[i].t()

            if self.right_weights:
                x_mu = x_mu @ self.lin_right_weights[i]
                # blocks = [x_sig[h] for h in range(self.hidden_channels)]
                # b = []
                # for block in blocks:
                #     a = [block[n] for n in range(self.graph_size)]
                #     b.append(torch.block_diag(*a))
                # b = torch.block_diag(*b)

                ### tentando usar matrix-normal distribution
                # V,U = self.kronecker_decomposition(b, self.hidden_channels, self.hidden_channels,
                #                              self.graph_size*self.d, self.graph_size*self.d)
                # if V == "cannot decompose":
                #     pass
                # else:
                #     V = self.lin_right_weights[i].t() @ V @ self.lin_right_weights[i]
                #     x_sig = torch.kron(V, U)
                #     retrived_blocks = []
                #     for n in range(self.graph_size):
                #         retrived_blocks.append(x_sig[n * self.hidden_channels * self.d : (n + 1) * self.hidden_channels * self.d,
                #                                     n * self.hidden_channels * self.d : (n + 1) * self.hidden_channels * self.d])
                #     to_be_x_sig = []
                #     for block in retrived_blocks:
                #         aux = []
                #         for h in range(self.hidden_channels):
                #             aux.append(block[h * self.d : (h + 1) * self.d, h * self.d : (h + 1) * self.d])
                #         aux = torch.stack(aux)
                #         to_be_x_sig.append(aux)
                #     x_sig = torch.stack(to_be_x_sig).transpose(1,0)
                

                # b = b.view(self.graph_size, self.d, self.hidden_channels, self.graph_size, self.d, self.hidden_channels).transpose(3, 2).transpose(4, 3)
                # x_sig = self.lin_right_weights[i].t()[None, None, None, None, ...] @ b @ self.lin_right_weights[i]
                # x_sig = x_sig.transpose(4, 3).transpose(3, 2).reshape(self.graph_size * self.d * self.hidden_channels, -1)
                
                # retrived_blocks = []
                # for h in range(self.hidden_channels):
                #     retrived_blocks.append(x_sig[h * self.graph_size * self.d : (h + 1) * self.graph_size * self.d,
                #                                  h * self.graph_size * self.d : (h + 1) * self.graph_size * self.d])
                # to_be_x_sig = []
                # for block in retrived_blocks:
                #     aux = []
                #     for node in range(self.graph_size):
                #         aux.append(block[node * self.d : (node + 1) * self.d, node * self.d : (node + 1) * self.d])
                #     aux = torch.stack(aux)
                #     to_be_x_sig.append(aux)
                # x_sig = torch.stack(to_be_x_sig)

                # usando n matrizes hdxhd
                # blocks = [x_sig[:, n] for n in range(self.graph_size)]
                # b = []
                # for block in blocks:
                #     a = [block[h] for h in range(self.hidden_channels)]
                #     b.append(torch.block_diag(*a))
                # b = torch.stack(b)
                # b = b.view(self.graph_size, self.d, self.hidden_channels, self.d, self.hidden_channels).transpose(3, 2)
                # x_sig = self.lin_right_weights[i].t()[None, None, None, ...] @ b @ self.lin_right_weights[i]
                # x_sig = x_sig.transpose(3, 2).reshape(self.graph_size, self.d * self.hidden_channels, -1)
                # retrived_blocks = []
                # for n in range(self.graph_size):
                #     aux = []
                #     for h in range(self.hidden_channels):
                #         aux.append(x_sig[n][h * self.d : (h + 1) * self.d,
                #                                     h * self.d : (h + 1) * self.d])
                #     aux = torch.stack(aux)
                #     retrived_blocks.append(aux)

                # x_sig = torch.stack(retrived_blocks).transpose(1,0)
                        

        x_mu = torch_sparse.spmm(L[0], L[1], x_mu.size(0), x_mu.size(0), x_mu)

        L = torch_sparse.to_torch_sparse(L[0], L[1], x_mu.size(0), x_mu.size(0)).coalesce().to_dense()
        L = L.view(self.graph_size, self.d, self.graph_size, self.d).transpose(2, 1)

        x_sig = torch.stack(
            [(L[i, :] @ x_sig @ L[i, :].transpose(2, 1)).sum(dim=1) 
             + 1e-6 * torch.eye(self.d, device=self.device) for i in range(self.graph_size)],
            dim=1)

        #assert torch.all(torch.tensor([(torch.linalg.eigvalsh(list_sigs[i]).real >= 0).all().item() for i in range(self.graph_size)]))

        if self.use_act:
            x_mu = self.act(x_mu)
            x_sig = self.act(x_sig)

        #Agregando as informações dos canais ocultos
        x_mu = x_mu.view(self.graph_size, self.d, self.hidden_channels).sum(dim=2)
        x_sig = x_sig.sum(dim=0)

        x_mu = self.emb2[None, ...] @ x_mu[..., None] #+ self.vec2[:, None]
        x_sig = self.emb2[None, ...] @ x_sig @ self.emb2[None, ...].transpose(2, 1)

        x_mu = x_mu.reshape(self.graph_size, -1)
        x_sig = torch.linalg.cholesky(x_sig + 1e-6 * torch.eye(self.dist_dim, device=self.device))

        rn = torch.randn((self.graph_size, self.dist_dim, self.num_samples), dtype=torch.float64, device=self.device)
        pred = x_mu[..., None] + x_sig @ rn
        pred = pred.view(self.graph_size, self.num_samples, self.dist_dim)
        pred = self.mlp(pred).view(self.graph_size, -1)

        return pred

### Modelo utilizando amostras na difusão
class SampledGaussianSheafDiffusion(SheafDiffusion):

    def __init__(self, edge_index, args):
        super(SampledGaussianSheafDiffusion, self).__init__(edge_index, args)
        assert args['d'] > 1

        self.lin_right_weights = []
        self.lin_left_weights = []

        if self.right_weights:
            self.lin_right_weights.append(nn.Linear(self.hidden_channels, self.hidden_channels, bias=False, dtype=torch.float64, device=self.device))
            nn.init.orthogonal_(self.lin_right_weights[-1].weight.data)
        if self.left_weights:
            self.lin_left_weights.append(nn.Linear(self.final_d, self.final_d, bias=False, dtype=torch.float64, device=self.device))
            nn.init.eye_(self.lin_left_weights[-1].weight.data)

        if self.sparse_learner:
            self.sheaf_learners = LocalConcatSheafLearnerVariant(self.final_d,
                    self.hidden_channels, out_shape=(self.d, self.d), sheaf_act=self.sheaf_act)
        else:
            self.sheaf_learners = LocalConcatSheafLearner(
                    self.hidden_dim, out_shape=(self.d, self.d), sheaf_act=self.sheaf_act)

        self.laplacian_builder = lb.GeneralLaplacianBuilder(
            self.graph_size, edge_index, d=self.d, add_lp=self.add_lp, add_hp=self.add_hp,
            normalised=self.normalised, deg_normalised=self.deg_normalised)

        self.epsilons = nn.Parameter(torch.zeros((self.final_d, 1)))

        #Embedding das amostras
        self.emb1 = nn.Linear(self.dist_dim * self.num_samples, self.hidden_dim, dtype=torch.float64)
        self.emb2 = nn.Linear(self.hidden_dim, self.dist_dim * self.num_samples, dtype=torch.float64)

        self.act = nn.ReLU()

        self.mlp = self.set_mlps()

    def forward(self, x):
        #x = F.dropout(x, p=self.input_dropout, training=self.training)
        samples = self.get_samples(x)
        samples = samples.view(self.graph_size, -1)

        samples = self.emb1(samples)

        if self.use_act:
            samples = self.act(samples)

        samples = samples.view(self.graph_size * self.d, -1)

        samples0 = samples

        L = None
        x_maps = samples
        x_maps = x_maps.reshape(self.graph_size, -1)
        maps = self.sheaf_learners(x_maps, self.edge_index)
        L, trans_maps = self.laplacian_builder(maps)
        self.sheaf_learners.set_L(trans_maps)

        l = L
        for layer in range(1, self.layers):
            L = torch_sparse.spspmm(l[0], l[1], L[0], L[1], samples.size(0), samples.size(0), samples.size(0), coalesced=True)
        
        if self.left_weights:
            samples = samples.t().reshape(-1, self.final_d)
            samples = self.lin_left_weights[0](samples)
            samples = samples.reshape(-1, self.graph_size * self.final_d).t()

        if self.right_weights:
            samples = self.lin_right_weights[0](samples)

        samples = torch_sparse.spmm(L[0], L[1], samples.size(0), samples.size(0), samples)
        

        if self.use_act:
            samples = self.act(samples)

        coeff = (1 + torch.tanh(self.epsilons).tile(self.graph_size, 1))
        samples0 = coeff * samples0 - samples
        samples = samples0

        # To detect the numerical instabilities of SVD.
        assert torch.all(torch.isfinite(samples))

        samples = samples.reshape(self.graph_size, -1)
        samples = self.emb2(samples)
        samples = samples.view(self.graph_size, self.num_samples, self.dist_dim)
        pred = self.mlp(samples).view(self.graph_size, -1)

        return pred