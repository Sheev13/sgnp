import torch
from torch import nn
import torch.nn.functional as F
from networks.set_architectures import DeepSet, TrAgDeepSet, ConvDeepSet, Transformer, TrAgTransformer
from networks.base_architectures import MLP
from .likelihoods import GaussianLikelihood
from .covariance_functions import Periodic, Tetouan

from typing import List, Optional

class BaseSVGP(nn.Module):
    def __init__(self, x_dim: int = None, num_inducing: int = 50, likelihood: nn.Module = None, prior: nn.Module = None, use_titsias: bool = False):
        super().__init__()
        if x_dim is None:
            raise ValueError("'x_dim' note specified for BaseMetaSVGP.")
        self.x_dim = x_dim
        self.y_dim = 1
        self.num_inducing = num_inducing
        # num_inducing * num_inputs for inputs Z, num_inducing for mean of U
        # num_inducing * (num_inducing + 1) / 2 for cholesky of U's covariance matrix, L
        if likelihood is None:
            raise ValueError("'likelihood' object not specified.")
        if prior is None:
            raise ValueError("'prior' object not specified.")
        
        self.prior = prior
        self.likelihood = likelihood
        self.use_titsias = use_titsias
        self.meta = False
    
    def _get_variational_parameters(self, X_c: torch.Tensor, y_c: torch.Tensor):
        raise NotImplementedError("No inference networks implemented for base meta-SVGP class.")
    
    def optimal_inducing_variables(self, Z, X_c, y_c, sigma_y_tilde=None):
        if sigma_y_tilde is None:
            sigma_y = self.likelihood.sigma_y
        else:
            sigma_y = sigma_y_tilde

        # computes optimal parameters of q(u) via Titsias 2009.
        Kmm = self.prior.covariance_function(Z)
        Knm = self.prior.covariance_function(X_c, Z)
        L_inv = torch.linalg.cholesky(
            Kmm + sigma_y.pow(-2) * Knm.T @ Knm + torch.eye(Kmm.shape[0]) * 1e-4
        )
        Sigma = torch.cholesky_inverse(L_inv)

        m_star = sigma_y.pow(-2) * Kmm @ Sigma @ Knm.T @ y_c
        # shape (num_inducing, 1)
        S_star = Kmm @ Sigma @ Kmm + torch.eye(Kmm.shape[0]) * 1e-4

        return m_star.squeeze(), S_star


    def q_fn(self, X_t: torch.Tensor,
             Z: torch.Tensor,
             m: torch.Tensor,
             S: torch.Tensor,
             multivariate: bool = False
             ):
        
        # the below is an implementation of Hensman et al. 2015
        Kmm = self.prior.covariance_function(Z)
        Knm = self.prior.covariance_function(X_t, Z)
        assert Knm.shape[1] == self.num_inducing
        Lmm = torch.linalg.cholesky(Kmm)
        A = torch.cholesky_solve(Knm.T, Lmm).T
        f_mu = (A @ m).squeeze() # add prior mean here if decide to use a nonzero one

        if not multivariate:
            Knn_diag = self.prior.covariance_function.diagonal(X_t)
            f_vars = Knn_diag - torch.einsum('ij,jk,ki->i', [A, Kmm - S, A.T])#).clamp(min=1e-8)
            return torch.distributions.Normal(f_mu, f_vars.sqrt())
        else:
            Knn = self.prior.covariance_function(X_t)
            f_covar = Knn - (A @ (Kmm - S) @ A.T)
            return torch.distributions.MultivariateNormal(f_mu, f_covar)
        
    def forward(self, X_t: torch.Tensor, X_c: torch.Tensor, y_c: torch.Tensor, multivariate=False):
        """Computes the posterior predictive in a single forward pass"""
        if self.use_titsias and not isinstance(self.likelihood, GaussianLikelihood) and not self.meta:
            raise ValueError("Cannot use Titsias shortcut for non-Gaussian likelihood in the standard SVGP.")
        Z, m, S = self._get_variational_parameters(X_c, y_c)
        return self.likelihood.posterior_predictive(self.q_fn(X_t, Z, m, S, multivariate=multivariate))
    
    def loss(self, X_c, y_c, X_t, y_t, num_samples=1, use_kl=True):
        """Monte Carlo estimate of ELBO defined on context dataset"""
        Z, m, S = self._get_variational_parameters(X_c, y_c)
        f_xt_samples = self.q_fn(X_t, Z, m, S).rsample((num_samples,))
        preds = self.likelihood(f_xt_samples)
        e_ll = self.likelihood.log_prob(predictions=preds, targets=y_t.squeeze()).mean(0).sum()

        q_u = torch.distributions.MultivariateNormal(m, S)
        kl = torch.tensor(0.0)
        if use_kl:
            kl = torch.distributions.kl.kl_divergence(q_u, self.prior(Z))

        elbo = e_ll - kl
        metrics = {
            "elbo": elbo.detach().item(),
            "e_ll": (e_ll).detach().item(),
            "kl": kl.detach().item()
        }

        for name, param in self.prior.named_parameters():
            if param.requires_grad:
                if (self.prior.covariance_function.ard and name.endswith('log_l')):
                    pass
                elif 'log' in name:
                    if 'log_' in name:
                        name = name.replace("log_", "")
                    elif '_log' in name:
                        name = name.replace("_log", "")
                    else:
                        name = name.replace("log", "")
                    metrics[name] = param.exp().detach().item()
                else:
                    metrics[name] = param.detach().item()

        for name, param in self.likelihood.named_parameters():
            if param.requires_grad:
                if 'log' in name:
                    if 'log_' in name:
                        name = name.replace("log_", "")
                    elif '_log' in name:
                        name = name.replace("_log", "")
                    else:
                        name = name.replace("log", "")
                    metrics[name] = param.exp().detach().item()
                else:
                    metrics[name] = param.detach().item()

        for name, param in self.named_parameters():
            if ('sigma_y' in name) and (param.requires_grad) and not isinstance(self.likelihood, GaussianLikelihood):
                name = 'sigma_y_tilde'
                metrics[name] = param.exp().detach().item()
            
        return - elbo, metrics
    

class SparseVariationalGaussianProcess(BaseSVGP):
    def __init__(self, **svgp_kwargs):
        super().__init__(**svgp_kwargs)

        # inducing points/inputs
        self.Z = nn.Parameter(torch.randn((self.num_inducing, self.x_dim)), requires_grad=True)
        # inducing output means
        self.m = nn.Parameter(torch.randn((self.num_inducing,)), requires_grad=True)
        # inducing output covariance function cholesky decomposition parameterisations
        self.L_log_diag = nn.Parameter(torch.log(torch.ones((self.num_inducing,))*0.1 + torch.randn((self.num_inducing,)) * 0.01), requires_grad=True)
        self.L_off_diag = nn.Parameter(torch.randn((self.num_inducing, self.num_inducing)) * 0.001, requires_grad=True)
        self.meta = False

    @property
    def L(self):
        return torch.diag(self.L_log_diag.exp()+1e-8) + torch.tril(self.L_off_diag, diagonal=-1)
    
    @property
    def S(self):
        return self.L @ self.L.T + torch.eye(self.num_inducing)*1e-5
    
    def _get_variational_parameters(self, X_c, y_c, *args, **kwargs):
        if self.use_titsias:
            m, S = self.optimal_inducing_variables(self.Z, X_c, y_c)
            return self.Z, m, S
        else:
            return self.Z, self.m, self.S
        
    def init_inducing_variables(self, X: torch.Tensor, t: torch.Tensor):
        assert X.shape[0] >= self.num_inducing
        inducing_idx = torch.randperm(X.shape[0])[:self.num_inducing]
        self.Z.data = X[inducing_idx,:]
        if isinstance(self.likelihood, GaussianLikelihood):
            self.m.data = t[inducing_idx]
        else: # Bernoulli likelihood
            # m lives in f space rather than t space, so logistic(m) is roughly 1 or 0 at initialisation
            self.m.data = torch.where(t[inducing_idx] == 1, torch.tensor(2.0), torch.tensor(-2.0))

class SparseGaussianNeuralProcess(BaseSVGP):
    def __init__(self,
                 x_dim: int = None,
                 cnn_hidden_chans: List[int]=[32, 32],
                 cnn_kernel_size: int = 5,
                 nonlinearity: nn.Module = nn.ReLU(),
                 grid_spacing: float = 1e-2,
                 learn_cds_l: bool = True,
                 d_k: int = 16,
                 use_unet: bool = False,
                 use_transformer: bool = False,
                 Z_net_width=128,
                 Z_net_hidden_depth=2,
                 meta_learn_hypers: Optional[List[str]] = None,
                 sigma_y_tilde: Optional[float] = None,
                 const: Optional[float] = None,
                 train_sigma_y_tilde: bool = False,
                 **svgp_kwargs
                ):
        super().__init__(x_dim=x_dim, **svgp_kwargs)
        self.grid_spacing = grid_spacing
        self.log_query_l = nn.Parameter(torch.log(torch.tensor(5 * grid_spacing)),
                                        requires_grad=learn_cds_l)
        
        if use_transformer:
            self.f = TrAgTransformer(x_dim=x_dim,
                                         output_dim=self.num_inducing,
                                         width=Z_net_width,
                                         nonlinearity=nonlinearity,
                                         scale_agnostic=False,
                                         num_layers=Z_net_hidden_depth,
                                        )

        else:
            hidden_dims = [Z_net_width] * Z_net_hidden_depth
            self.f = TrAgDeepSet(mlp_dims=[x_dim+1] + hidden_dims + [self.num_inducing * self.x_dim],
                                     nonlinearity=nonlinearity,
                                     scale_agnostic=False,
                                    )

        chans = [2] + cnn_hidden_chans

        self.g = ConvDeepSet(x_dim=x_dim,
                            grid_spacing=grid_spacing,
                            cnn_chans=chans+[1],
                            cnn_kernel_size=cnn_kernel_size,
                            learn_l=learn_cds_l,
                            nonlinearity=nonlinearity,
                            use_unet=use_unet,
                            )
        self.h_k = ConvDeepSet(x_dim=x_dim,
                            grid_spacing=grid_spacing,
                            cnn_chans=chans+[d_k],
                            cnn_kernel_size=cnn_kernel_size,
                            learn_l=learn_cds_l,
                            nonlinearity=nonlinearity,
                            use_unet=use_unet,
                            )
        self.h_v = ConvDeepSet(x_dim=x_dim,
                            grid_spacing=grid_spacing,
                            cnn_chans=chans+[1],
                            cnn_kernel_size=cnn_kernel_size,
                            learn_l=learn_cds_l,
                            nonlinearity=nonlinearity,
                            use_unet=use_unet,
                            )
        
        self.hypers_net = None

        self.meta_hypers = []
        if meta_learn_hypers is not None and len(meta_learn_hypers) != 0:
            meta_hypers_counter = 0
            if 'l' in [hyper.lower() for hyper in meta_learn_hypers]:
                self.prior.covariance_function.log_l.requires_grad = True
                if self.prior.covariance_function.ard:
                    meta_hypers_counter += x_dim
                    self.meta_hypers.append('ard_l')
                else:
                    meta_hypers_counter += 1
                    self.meta_hypers.append('l')
            if 'sigma_y' in [hyper.lower() for hyper in meta_learn_hypers]:
                if isinstance(self.likelihood, GaussianLikelihood):
                    self.likelihood.sigma_y.requires_grad = True
                    meta_hypers_counter += 1
                    self.meta_hypers.append('sigma_y')
            if 'p' in [hyper.lower() for hyper in meta_learn_hypers]:
                if isinstance(self.prior.covariance_function, Periodic):
                    self.prior.covariance_function.log_p.requires_grad = True
                    meta_hypers_counter += 1
                    self.meta_hypers.append('p')
            for hyper in meta_learn_hypers:
                if hyper.lower() not in ['l', 'sigma_y', 'p']:
                    raise ValueError(f"Hyperparameter '{hyper}' not recognised.")
            
            if use_transformer:
                self.hypers_net = TrAgTransformer(x_dim=x_dim,
                                                  output_dim=meta_hypers_counter,
                                                  width=Z_net_width,
                                                  nonlinearity=nonlinearity,
                                                  scale_agnostic=True,
                                                  num_layers=Z_net_hidden_depth,
                                                 )
            else:
                hidden_dims = [Z_net_width] * Z_net_hidden_depth
                self.hypers_net = TrAgDeepSet(mlp_dims=[x_dim+1] + hidden_dims + [self.num_inducing * self.x_dim],
                                              nonlinearity=nonlinearity,
                                              scale_agnostic=True,
                                             )

        
        self.min_gridpoints = 1
        if use_unet:
            self.min_gridpoints = cnn_kernel_size * 2**self.f.rho.unet_depth

        self.log_kvv_l = nn.Parameter(torch.tensor(1.0).log(), requires_grad=True)

        if self.use_titsias and not isinstance(self.likelihood, GaussianLikelihood): # i.e. s-ConvSGNP since non-Gaussian likelihood
            if sigma_y_tilde is None:
                raise ValueError("User must specify a suitable (pseudo) sigma_y for the s-ConvSGNP.")
            else:
                self.log_sigma_y_tilde = nn.Parameter(torch.tensor(sigma_y_tilde).log(), requires_grad=train_sigma_y_tilde)
            if const is None:
                raise ValueError("User must specify a suitable c for the s-ConvSGNP.")
            else:
                self.const = const
        else:
            self.log_sigma_y_tilde = None
            self.const = None

        self.meta = True

    @property
    def sigma_y_tilde(self):
        if self.log_sigma_y_tilde is None:
            raise ValueError("s-ConvSGNP's pseudo sigma_y is trying to be accessed when s-ConvSGNP is not in use.")
        return self.log_sigma_y_tilde.exp()

    @property
    def query_l(self):
        return self.log_query_l.exp()
    
    @property
    def kvv_l(self):
        return self.log_kvv_l.exp()
    
    def _set_meta_hypers(self, X_c, y_c):

        raw_hypers = self.hypers_net(X_c, y_c).squeeze() # shape (num_meta_hypers,)
        if raw_hypers.dim() == 0:
            raw_hypers = raw_hypers.unsqueeze(0)

        i = 0
        if 'sigma_y' in self.meta_hypers:
            self.likelihood.log_sigma_y.data = raw_hypers[i] - 2.0
            i += 1
        if 'p' in self.meta_hypers:
            self.prior.covariance_function.log_p.data = raw_hypers[i]
            i += 1
        if 'l' in self.meta_hypers:
            self.prior.covariance_function.log_l.data = raw_hypers[i]
        elif 'ard_l' in self.meta_hypers:
            self.prior.covariance_function.log_l.data = raw_hypers[i:]

    def _get_variational_parameters(self, X_c: torch.Tensor, y_c: torch.Tensor):
        if len(X_c.shape) < 2:
            X_c = X_c.unsqueeze(-1)
        if len(y_c.shape) < 2:
            y_c = y_c.unsqueeze(-1)

        if self.hypers_net is not None:
            self._set_meta_hypers(X_c, y_c)

        Z = self.f(X_c, y_c).reshape((self.num_inducing, self.x_dim))

        if self.use_titsias and isinstance(self.likelihood, GaussianLikelihood): # SGNP
            m, S = self.optimal_inducing_variables(Z, X_c, y_c)

        else: # ConvSGNP and s-ConvSGNP
            t = self._construct_grid(X_c, Z)
            m_encoding = self.g(X_c, y_c, t)
            m = self._query_encoding(m_encoding, Z, t).squeeze()
            # kvv covariance parameterisation (see Markou et al 2021)
            S_k_encoding = self.h_k(X_c, y_c, t)
            S_v_encoding = self.h_v(X_c, y_c, t)
            S_k = self._query_encoding(S_k_encoding, Z, t) / self.kvv_l
            S_v = self._query_encoding(S_v_encoding, Z, t)
            K_kk = torch.exp(-0.5 * torch.cdist(S_k, S_k).square())
            vv = torch.outer(S_v.squeeze(), S_v.squeeze())
            S = K_kk * vv + torch.eye(self.num_inducing) * 1e-3

            if self.use_titsias: # i.e. s-ConvSGNP since non-Gaussian likelihood
                if self.sigma_y_tilde is None:
                    raise ValueError("User must specify a suitable (pseudo) sigma_y for the s-ConvSGNP.")
                if self.const is None:
                    raise ValueError("User must specify a suitable c for the s-ConvSGNP.")
                soft_y_c = (2 * y_c - 1) * self.const # assuming y's are 1's or 0's.
                m_est, S_est = self.optimal_inducing_variables(Z, X_c, soft_y_c, sigma_y_tilde=self.sigma_y_tilde)

                m = torch.lerp(m_est, m, weight=0.1)
                S = torch.lerp(S_est, S, weight=0.1)

        return Z, m, S

    def _construct_grid(self, X_c, Z):
        """a function to generate a hypercubic grid of evenly spaced inputs that span the domain of X_c AND Z"""
        # get min and max of union of X_c and Z
        union = torch.cat((X_c, Z), dim=0)
        t_min, t_max = union.min(dim=0)[0], union.max(dim=0)[0]
        # widen the span to avoid edge effects in convolutions later on
        t_min, t_max = t_min - 0.1, t_max + 0.1
        # handle variable shapes
        if len(t_min.shape) == 0: # occurs if each point in X_c and Z is 1 dimensional
            t_min, t_max = t_min.unsqueeze(0), t_max.unsqueeze(0)

        # if x's are d-dimensional, the below generates a d-dimensional image of d-dimensional
        # coordinates, i.e. if x_dim is 2, grid is shape (grid_width, grid_height, 2)
        aranges = [torch.arange(t_min[i].item(), t_max[i].item(), self.grid_spacing) for i in range(len(t_max))]
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

    def _query_encoding(self, grid_enc, Z, t):
        # grid_enc is shape (grid_dim_1, ..., grid_dim_x_dim, cnn_out)
        # Z is shape (num_inducing, x_dim)
        # t is shape (grid_dim_1, ..., grid_dim_x_dim, x_dim)
        t = t.reshape((-1, self.x_dim)) # shape (total_gridpoints, x_dim)
        grid_enc = grid_enc.reshape((t.shape[0], -1)) # shape (total_gridpoints, cnn_out)
        dists = torch.cdist(Z / self.query_l, t / self.query_l) # shape (num_inducing, total_gridpoints)
        bases = torch.exp(-0.5 * dists)
        return (grid_enc.unsqueeze(0) * bases.unsqueeze(-1)).sum(1) # shape (num_inducing, cnn_out)