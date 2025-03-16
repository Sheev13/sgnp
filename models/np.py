import torch
from torch import nn
import torch.nn.functional as F
from networks.set_architectures import DeepSet, TrAgDeepSet, ConvDeepSet, Transformer, TrAgTransformer
from networks.base_architectures import MLP
# from sparse_gp.likelihoods import GaussianLikelihood

from typing import List, Optional

class ConvGNP(nn.Module):
    def __init__(self,
                 x_dim: int = None,
                 cnn_hidden_chans: List[int]=[32, 32],
                 cnn_kernel_size: int = 5,
                 nonlinearity: nn.Module = nn.ReLU(),
                 grid_spacing: float = 1e-2,
                 init_ls_multiplier=2,
                 learn_ls: bool = True,
                 d_k: int = 16,
                 use_unet: bool = False,
                 classification: bool = False,
                 tetouan_grid_spacing: Optional[List[float]] = None,
                ):
        super().__init__()
        self.x_dim = x_dim
        if tetouan_grid_spacing is not None:
            self.grid_spacing = tetouan_grid_spacing
        else:
            self.grid_spacing = [grid_spacing] * x_dim
        self.log_query_l = nn.Parameter(torch.log(init_ls_multiplier * torch.tensor(self.grid_spacing)),
                                        requires_grad=learn_ls)

        chans = [2] + cnn_hidden_chans
        # Naming of below networks follows that of "Efficient Gaussian Neural 
        # Process for Regression" (Markou et al 2021)
        self.f = ConvDeepSet(x_dim=x_dim,
                            grid_spacing_list=self.grid_spacing,
                            cnn_chans=chans+[1],
                            cnn_kernel_size=cnn_kernel_size,
                            l_multiplier=init_ls_multiplier,
                            learn_l=learn_ls,
                            nonlinearity=nonlinearity,
                            use_unet=use_unet,
                            )
        self.g = ConvDeepSet(x_dim=x_dim,
                            grid_spacing_list=self.grid_spacing,
                            cnn_chans=chans+[d_k],
                            cnn_kernel_size=cnn_kernel_size,
                            l_multiplier=init_ls_multiplier,
                            learn_l=learn_ls,
                            nonlinearity=nonlinearity,
                            use_unet=use_unet,
                            )
        self.v = ConvDeepSet(x_dim=x_dim,
                            grid_spacing_list=self.grid_spacing,
                            cnn_chans=chans+[1],
                            cnn_kernel_size=cnn_kernel_size,
                            l_multiplier=init_ls_multiplier,
                            learn_l=learn_ls,
                            nonlinearity=nonlinearity,
                            use_unet=use_unet,
                            )
        
        self.min_gridpoints = 1
        if use_unet:
            self.min_gridpoints = cnn_kernel_size * 2**self.f.rho.unet_depth
        
        self.log_l =nn.Parameter(torch.tensor(grid_spacing * init_ls_multiplier).log(), requires_grad=True)

        self.classification = classification

    @property
    def query_l(self):
        return self.log_query_l.exp()
    
    @property
    def l(self):
        return self.log_l.exp()

    def forward(self, X_c: torch.Tensor, y_c: torch.Tensor, X_t: torch.Tensor):
        if len(X_c.shape) < 2:
            X_c = X_c.unsqueeze(-1)
        if len(y_c.shape) < 2:
            y_c = y_c.unsqueeze(-1)
        if len(X_t.shape) < 2:
            X_t = X_t.unsqueeze(-1)

        t = self._construct_grid(X_c, X_t)

        m_encoding = self.f(X_c, y_c, t)
        m = self._query_encoding(m_encoding, X_t, t).squeeze(-1)

        # kvv covariance parameterisation (see Markou et al 2021)
        S_g_encoding = self.g(X_c, y_c, t)
        S_v_encoding = self.v(X_c, y_c, t)
        S_g = self._query_encoding(S_g_encoding, X_t, t) / self.l
        S_v = self._query_encoding(S_v_encoding, X_t, t)
        K_gg = torch.exp(-0.5 * torch.cdist(S_g, S_g).square())
        vv = torch.outer(S_v.squeeze(-1), S_v.squeeze(-1))
        S = K_gg * vv + torch.eye(X_t.shape[0]) * 1e-3

        if self.classification:
            vars = S.diagonal()
            # following Pattern Recognition and Machine Learning, Chris Bishop, pp.218-220
            # (our fn[i] is their a).
            # This is the probit approximation to the logistic function.
            return torch.distributions.Bernoulli(probs=torch.sigmoid(m / (1 + torch.pi*vars/8).pow(0.5)))
        else:
            return torch.distributions.MultivariateNormal(m, S)

    def _construct_grid(self, X_c, X_t):
        """a function to generate a grid of evenly spaced inputs that span the domain of X_c AND X_t"""
        # get min and max of union of X_c and X_t
        union = torch.cat((X_c, X_t), dim=0)
        t_min, t_max = union.min(dim=0)[0], union.max(dim=0)[0]
        # widen the span to avoid edge effects in convolutions later on
        t_min, t_max = t_min - 0.1, t_max + 0.1
        # handle variable shapes
        if len(t_min.shape) == 0: # occurs if each point in X_c and Z is 1 dimensional
            t_min, t_max = t_min.unsqueeze(0), t_max.unsqueeze(0)

        # if x's are d-dimensional, the below generates a d-dimensional image of d-dimensional
        # coordinates, i.e. if x_dim is 2, grid is shape (grid_width, grid_height, 2)
        aranges = [torch.arange(t_min[i].item(), t_max[i].item(), self.grid_spacing[i]) for i in range(len(t_max))]
        grid = torch.cat([t.unsqueeze(-1) for t in torch.meshgrid(*aranges)], dim=-1)
        grid = self._pad_grid_if_needed(grid)
        return grid # has shape (grid_dim_1, ..., grid_dim_x_dim, x_dim)
    
    def _pad_grid_if_needed(self, grid: torch.Tensor):
        # grid expected to have shape (grid_dim_1, ..., grid_dim_x_dim, x_dim)
        if any([gridpoints < self.min_gridpoints for gridpoints in grid.shape]):
            grid = grid.permute(-1, *range(len(grid.shape)-1)) # bring last dimension to first position
            pad = []
            for i in range(1, len(grid.shape)):
                if grid.shape[i] < self.min_gridpoints:
                    diff = self.min_gridpoints - grid.shape[i]
                    if diff % 2 == 0: # if difference is even:
                        pad += [diff//2, diff//2]
                    else: # if difference is odd:
                        pad += [diff//2 + 1, diff//2]
                else:
                    pad += [0, 0]
            pad = tuple(pad[::-1])
            grid = F.pad(grid, pad)
            return grid.permute(*range(1, len(grid.shape)), 0) # return first dimension to back position
        else:
            return grid

    def _query_encoding(self, grid_enc, X_t, t):
        # grid_enc is shape (grid_dim_1, ..., grid_dim_x_dim, cnn_out)
        # X_t is shape (batch_t, x_dim)
        # t is shape (grid_dim_1, ..., grid_dim_x_dim, x_dim)
        t = t.reshape((-1, self.x_dim)) # shape (total_gridpoints, x_dim)
        grid_enc = grid_enc.reshape((t.shape[0], -1)) # shape (total_gridpoints, cnn_out)
        dists = torch.cdist(X_t / self.query_l, t / self.query_l) # shape (batch_t, total_gridpoints)
        bases = torch.exp(-0.5 * dists)
        return (grid_enc.unsqueeze(0) * bases.unsqueeze(-1)).sum(1) # shape (batch_t, cnn_out)
    

    def loss(self, X_c, y_c, X_t, y_t, **redundant_kwargs):
            """Predictive log likelihood of targets given contexts"""
            predictive = self(X_c, y_c, X_t)
            ll = predictive.log_prob(y_t.squeeze(-1))
            if self.classification:
                ll = ll.sum()

            metrics = {
                "ll": ll.detach().item(),
            }
                
            return - ll, metrics
    

class ConvCNP(nn.Module):
    def __init__(self,
                 x_dim: int = None,
                 cnn_hidden_chans: List[int]=[32, 32],
                 cnn_kernel_size: int = 5,
                 nonlinearity: nn.Module = nn.ReLU(),
                 grid_spacing: float = 1e-2,
                 init_ls_multiplier=2,
                 learn_ls: bool = True,
                 use_unet: bool = False,
                 classification: bool = False,
                 tetouan_grid_spacing: Optional[List[float]] = None,
                ):
        super().__init__()
        self.x_dim = x_dim
        if tetouan_grid_spacing is not None:
            self.grid_spacing = tetouan_grid_spacing
        else:
            self.grid_spacing = [grid_spacing] * x_dim
        self.log_query_l = nn.Parameter(torch.log(init_ls_multiplier * torch.tensor(self.grid_spacing)),
                                        requires_grad=learn_ls)

        chans = [2] + cnn_hidden_chans
        if classification:
            out_chans = 1
        else:
            out_chans = 2
        chans += [out_chans]

        self.f = ConvDeepSet(x_dim=x_dim,
                            grid_spacing_list=self.grid_spacing,
                            cnn_chans=chans,
                            cnn_kernel_size=cnn_kernel_size,
                            l_multiplier=init_ls_multiplier,
                            learn_l=learn_ls,
                            nonlinearity=nonlinearity,
                            use_unet=use_unet,
                            )
        
        self.min_gridpoints = 1
        if use_unet:
            self.min_gridpoints = cnn_kernel_size * 2**self.f.rho.unet_depth

        self.classification = classification

    @property
    def query_l(self):
        return self.log_query_l.exp()

    def forward(self, X_c: torch.Tensor, y_c: torch.Tensor, X_t: torch.Tensor):
        if len(X_c.shape) < 2:
            X_c = X_c.unsqueeze(-1)
        if len(y_c.shape) < 2:
            y_c = y_c.unsqueeze(-1)
        if len(X_t.shape) < 2:
            X_t = X_t.unsqueeze(-1)

        t = self._construct_grid(X_c, X_t)

        encoding = self.f(X_c, y_c, t)
        pred_params = self._query_encoding(encoding, X_t, t).squeeze(-1) # shape (batch, 2) or (batch, 1)

        if self.classification:
            logits = pred_params.squeeze()
            return torch.distributions.Bernoulli(logits=logits)
        else:
            means, stds = pred_params[:,0], 0.02+0.98*nn.functional.softplus(pred_params[:,1])
            return torch.distributions.Normal(means, stds)

    def _construct_grid(self, X_c, X_t):
        """a function to generate a grid of evenly spaced inputs that span the domain of X_c AND X_t"""
        # get min and max of union of X_c and X_t
        union = torch.cat((X_c, X_t), dim=0)
        t_min, t_max = union.min(dim=0)[0], union.max(dim=0)[0]
        # widen the span to avoid edge effects in convolutions later on
        t_min, t_max = t_min - 0.1, t_max + 0.1
        # handle variable shapes
        if len(t_min.shape) == 0: # occurs if each point in X_c and Z is 1 dimensional
            t_min, t_max = t_min.unsqueeze(0), t_max.unsqueeze(0)

        # if x's are d-dimensional, the below generates a d-dimensional image of d-dimensional
        # coordinates, i.e. if x_dim is 2, grid is shape (grid_width, grid_height, 2)
        aranges = [torch.arange(t_min[i].item(), t_max[i].item(), self.grid_spacing[i]) for i in range(len(t_max))]
        grid = torch.cat([t.unsqueeze(-1) for t in torch.meshgrid(*aranges)], dim=-1)
        grid = self._pad_grid_if_needed(grid)
        return grid # has shape (grid_dim_1, ..., grid_dim_x_dim, x_dim)
    
    def _pad_grid_if_needed(self, grid: torch.Tensor):
        # grid expected to have shape (grid_dim_1, ..., grid_dim_x_dim, x_dim)
        if any([gridpoints < self.min_gridpoints for gridpoints in grid.shape]):
            grid = grid.permute(-1, *range(len(grid.shape)-1)) # bring last dimension to first position
            pad = []
            for i in range(1, len(grid.shape)):
                if grid.shape[i] < self.min_gridpoints:
                    diff = self.min_gridpoints - grid.shape[i]
                    if diff % 2 == 0: # if difference is even:
                        pad += [diff//2, diff//2]
                    else: # if difference is odd:
                        pad += [diff//2 + 1, diff//2]
                else:
                    pad += [0, 0]
            pad = tuple(pad[::-1])
            grid = F.pad(grid, pad)
            return grid.permute(*range(1, len(grid.shape)), 0) # return first dimension to back position
        else:
            return grid

    def _query_encoding(self, grid_enc, X_t, t):
        # grid_enc is shape (grid_dim_1, ..., grid_dim_x_dim, cnn_out)
        # X_t is shape (batch_t, x_dim)
        # t is shape (grid_dim_1, ..., grid_dim_x_dim, x_dim)
        t = t.reshape((-1, self.x_dim)) # shape (total_gridpoints, x_dim)
        grid_enc = grid_enc.reshape((t.shape[0], -1)) # shape (total_gridpoints, cnn_out)
        dists = torch.cdist(X_t / self.query_l, t / self.query_l) # shape (batch_t, total_gridpoints)
        bases = torch.exp(-0.5 * dists)
        return (grid_enc.unsqueeze(0) * bases.unsqueeze(-1)).sum(1) # shape (batch_t, cnn_out)
    

    def loss(self, X_c, y_c, X_t, y_t, **redundant_kwargs):
            """Predictive log likelihood of targets given contexts"""
            predictive = self(X_c, y_c, X_t)
            ll = predictive.log_prob(y_t.squeeze(-1)).sum()

            metrics = {
                "ll": ll.detach().item(),
            }
                
            return - ll, metrics
    
class CNP(nn.Module):
    def __init__(self,
                 x_dim: int = None,
                 deepset_dims: List[int]=[32, 32],
                 decoder_dims: List[int]=[32, 32],
                 nonlinearity: nn.Module = nn.ReLU(),
                 classification: bool = False,
                ):
        super().__init__()
        self.x_dim = x_dim
        if deepset_dims[-1] != decoder_dims[0]:
            raise ValueError("Final DeepSet layer dimension and decoding MLP first layer dimension must match.")

        deepset_dims = [x_dim + 1] + deepset_dims # y_dim assumed to be 1
        self.encoder = DeepSet(mlp_dims=deepset_dims,
                                nonlinearity=nonlinearity,
                                )
        
        decoder_dims[0] = decoder_dims[0] + x_dim
        if classification:
            decoder_dims += [1]
        else:
            decoder_dims += [2]
        self.decoder = MLP(dims=decoder_dims, nonlinearity=nonlinearity)

        self.classification = classification

    def forward(self, X_c: torch.Tensor, y_c: torch.Tensor, X_t: torch.Tensor):
        if len(y_c.shape) < 2:
            y_c = y_c.unsqueeze(-1)
        r = self.encoder(X_c, y_c, flat_representation=True) # shape (latent_dim,)

        n_t = X_t.shape[0]
        repeated_r = r.unsqueeze(0).repeat((n_t, 1)) # shape (n_t, latent_dim)
        decoder_input = torch.cat((repeated_r, X_t), dim=-1) # shape (n_t, latent_dim + x_dim)

        pred_params = self.decoder(decoder_input) # shape (n_t, 2) or (n_t, 1)

        if self.classification:
            logits = pred_params.squeeze()
            return torch.distributions.Bernoulli(logits=logits)
        else:
            means, stds = pred_params[:,0], 0.02+0.98*nn.functional.softplus(pred_params[:,1])
            return torch.distributions.Normal(means, stds)

    def loss(self, X_c, y_c, X_t, y_t, **redundant_kwargs):
            """Predictive log likelihood of targets given contexts"""
            predictive = self(X_c, y_c, X_t)
            ll = predictive.log_prob(y_t.squeeze(-1)).sum()

            metrics = {
                "ll": ll.detach().item(),
            }
                
            return - ll, metrics

class TNP(nn.Module):
    def __init__(self,
                 x_dim: int = None,
                 num_layers: int = 2,
                 r_dim: int = 64,
                 nonlinearity: nn.Module = nn.ReLU(),
                 classification: bool = False,
                 non_diagonal: bool = False,
                 num_heads: int = 8,
                 d_k: int = 16,
                ):
        super().__init__()
        self.x_dim = x_dim
        self.r_dim = r_dim

        self.tokeniser = MLP([x_dim + 1, r_dim, r_dim], nonlinearity=nonlinearity)

        transformer_layer = nn.TransformerEncoderLayer(
            d_model=r_dim,
            nhead=num_heads,
            dim_feedforward=num_heads,
            dropout=0.0,
            activation=nonlinearity,
            batch_first=True,
        )

        self.transformer = nn.TransformerEncoder(
            transformer_layer,
            num_layers=num_layers,
        )

        if classification or non_diagonal:
            out_dim = 1
        else:
            out_dim = 2

        if non_diagonal:
            self.covariance_encoder = nn.TransformerEncoder(
                transformer_layer,
                num_layers=1,
            )
            self.covariance_decoder = MLP([r_dim, r_dim, d_k])

        self.decoder = MLP([r_dim, r_dim, out_dim], nonlinearity=nonlinearity)

        self.classification = classification
        self.non_diagonal = non_diagonal

    def forward(self, X_c: torch.Tensor, y_c: torch.Tensor, X_t: torch.Tensor):
        if len(y_c.shape) < 2:
            y_c = y_c.unsqueeze(-1)
            
        Z_c = torch.cat((X_c, y_c), dim=-1) # shape (n_c, x_dim+y_dim)
        Z_t = torch.cat((X_t, torch.zeros((X_t.shape[0], 1))), dim=-1) # shape (n_t, x_dim+y_dim)
        Z = torch.cat((Z_c, Z_t), dim=0) # shape (n_c+n_t, x_dim+y_dim)
        n_c, n_t, n = Z_c.shape[0], Z_t.shape[0], Z.shape[0]

        mask = torch.zeros((n, n)).fill_(float('-inf'))
        mask[:,:n_c] = 0.0

        Z = self.tokeniser(Z) # shape (n_c+n_t, r_dim)
        Z = self.transformer(Z, mask=mask) # shape (n_c+n_t, r_dim)
        y_t_params = self.decoder(Z[n_c:,:]) # # shape (n_t, 1) or (n_t, 2)

        if self.classification:
            logits = y_t_params.squeeze()
            return torch.distributions.Bernoulli(logits=logits)
        elif self.non_diagonal:
            means = y_t_params.squeeze()
            y_t_cov_params = self.covariance_decoder(self.covariance_encoder(Z[n_c:,:])) # shape (n_t, d_k)
            L = (y_t_cov_params @ y_t_cov_params.T).tril() # lower triangular of shape (n_t, n_t)
            return torch.distributions.MultivariateNormal(means, scale_tril=L)
        else: # i.e. regression with diagonal Gaussian likelihood
            means, stds = y_t_params[:,0], 0.02+0.98*nn.functional.softplus(y_t_params[:,1])
            return torch.distributions.Normal(means, stds)

    def loss(self, X_c, y_c, X_t, y_t, **redundant_kwargs):
            """Predictive log likelihood of targets given contexts"""
            predictive = self(X_c, y_c, X_t)
            if self.classification or not self.non_diagonal:
                ll = predictive.log_prob(y_t.squeeze(-1)).sum()
            else: # i.e. if self.non_diagonal
                ll = predictive.log_prob(y_t.squeeze(-1))

            metrics = {
                "ll": ll.detach().item(),
            }
                
            return - ll, metrics