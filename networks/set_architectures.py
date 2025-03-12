import torch
from torch import nn
import sys
sys.path.append("../")
from .base_architectures import MLP, CNN, UNet

class DeepSet(nn.Module):
    def __init__(self, mlp_dims=None, nonlinearity=nn.ReLU()):
        super().__init__()
        if mlp_dims is None:
            raise ValueError("'mlp_dims' not specified for DeepSet.")
        
        self.mlp_dims = mlp_dims
        
        # pre-aggregation encoder:
        self.phi = MLP(mlp_dims, nonlinearity=nonlinearity)
        # aggregation function:
        self.rho = lambda Phi: Phi.nanmean(dim=0)

    def forward(self, X_c, y_c, flat_representation=False):
        D_c = torch.cat((X_c, y_c), dim=-1) # shape (batch_size, input_dim)
        Phi = self.phi(D_c) # shape (batch_size, output_dim)
        if flat_representation:
            raw_output = self.rho(Phi) # shape (output_dim,)
        else:
            raw_output = self.rho(Phi).reshape((-1, X_c.shape[-1])) # shape (output_dim, x_dim)

        return raw_output
    

class TrAgDeepSet(DeepSet):
    def __init__(self, scale_agnostic: bool = False, **kwargs):
        super().__init__(**kwargs)
        self.scale_agnostic = scale_agnostic

    def forward(self, X_c, y_c):
        X_c_scale = torch.tensor([1.0])
        if self.scale_agnostic:
            if X_c.shape[0] > 1:
                X_c_scale = X_c.std(0)
        X_c_mean = X_c.mean(0)
        X_c_tilde = (X_c - X_c_mean) / X_c_scale
        return super().forward(X_c_tilde, y_c) * X_c_scale + X_c_mean
    

class ConvDeepSet(nn.Module):
    def __init__(self,
                 x_dim=None,
                 grid_spacing_list=[1e-2],
                 cnn_chans=None,
                 cnn_kernel_size=None,
                 nonlinearity=nn.ReLU(),
                 l_multiplier=2,
                 learn_l = True,
                 use_unet: bool = False
                ):
        super().__init__()
        if x_dim is None:
            raise ValueError("'x_dim' not specified for ConvDeepSet.")
        if cnn_chans is None:
            raise ValueError("'cnn_chans' not specified for ConvDeepSet.")
        if cnn_kernel_size is None:
            raise ValueError("'cnn_kernel_size' not specified for ConvDeepSet.")
        assert len(grid_spacing_list) == x_dim
        
        l = l_multiplier * torch.tensor(grid_spacing_list)
        self.log_l = nn.Parameter(torch.log(l),
                                  requires_grad=learn_l)
        
        if use_unet:
            self.rho = UNet(x_dim, cnn_chans[0], cnn_chans[-1], kernel_size=cnn_kernel_size, nonlinearity=nonlinearity)
        else:
            self.rho = CNN(x_dim, cnn_chans, cnn_kernel_size, nonlinearity=nonlinearity)
    
    @property
    def l(self):
        l = self.log_l.exp()
        return l.unsqueeze(-1).repeat((1, 2)).unsqueeze(0) # shape (1, x_dim, 2)
    
    def phi(self, y):
        density = torch.ones_like(y.squeeze()).unsqueeze(-1)
        return torch.cat((density, y.squeeze().unsqueeze(-1)), dim=-1) # shape (batch, 2)
    
    def psi(self, x, t):
        # x is shape (batch, x_dim), t is shape (grid_dim_1, ..., grid_dim_x_dim, x_dim)
        # l is shape (1, 1, 2)
        flat_t = t.reshape((-1, t.shape[-1]))
        scaled_x = x.unsqueeze(-1) / self.l # shape (batch, x_dim, 2)
        scaled_flat_t = flat_t.unsqueeze(-1) / self.l # shape (total_gridpoints, x_dim, 2)
        K = torch.exp(-0.5 * torch.cdist(scaled_x.permute((2, 0, 1)), scaled_flat_t.permute(2, 0, 1))) # shape (2, batch, total_gridpoints)
        return K.permute((1, 2, 0)).reshape(-1, *t.shape[:-1], 2) # shape (batch, grid_dim_1, ..., grid_dim_x_dim, 2)
    
    def forward(self, X_c, y_c, t):
        # obtain grid of evaluations of function space embedding:
        phi = self.phi(y_c) # shape (batch, 2)
        psi = self.psi(X_c, t) # shape (batch, grid_dim_1, ..., grid_dim_x_dim, 2)
        while len(phi.shape) < len(psi.shape): # make phi of shape (batch, 1, ..., 1, 2) of same tensor rank as psi
            phi = phi.unsqueeze(1)
        E_z = (phi * psi).sum(0) # shape (grid_dim_1, ..., grid_dim_x_dim, 2)

        # divide data channel by density channel:
        norm_E_z = torch.zeros_like(E_z)
        norm_E_z[...,1] = E_z[...,1] / E_z[...,0] 
        norm_E_z[...,0] = E_z[...,0]
        # reshape for CNN:
        norm_E_z = norm_E_z.permute((-1, *range(norm_E_z.dim()-1))) # shape (2, grid_dim_1, ..., grid_dim_x_dim)
        # pass through CNN:
        rho = self.rho(norm_E_z)
        # reshape to how it was, and return:
        return rho.permute((*range(1, rho.dim()), 0)) # shape (grid_dim_1, ..., grid_dim_x_dim, 2)
    

class Transformer(nn.Module):
    def __init__(self,
                 x_dim=None,
                 output_dim=None,
                 width=128,
                 nonlinearity=nn.ReLU(),
                 num_layers=2,
                ):
        super().__init__()

        self.tokenizer = MLP([x_dim+1, width])
        num_heads = 8

        if width % num_heads < (num_heads + 1 // 2) + 1:
            width = (width // 8) * 8
        else:
            width = ((width // 8) + 1) * 8

        transformer_layer = nn.TransformerEncoderLayer(
            d_model=width,
            nhead=num_heads,
            dim_feedforward=width,
            dropout=0.0,
            activation=nonlinearity,
            batch_first=True,
        )

        self.transformer = nn.TransformerEncoder(
            transformer_layer,
            num_layers=num_layers,
        )

        self.decoder = MLP([width, output_dim * x_dim])

    def forward(self, X_c, y_c):
        D_c = torch.cat((X_c, y_c), dim=-1) # shape (batch_size, x_dim+1)
        D_c_tokens = self.tokenizer(D_c) # shape (batch_size, width)
        # what we refer to here as batch_size, torch docs for 
        # TransformerEncoderLayer refer to as sequence length. We do not
        # have what they refer to as batch_size, so we unsqueeze instead
        out_tokens = self.transformer(D_c_tokens.unsqueeze(0)).squeeze(0)

        raw_output = self.decoder(out_tokens) # shape (batch_size, output_dim*x_dim)

        return raw_output.mean(0).reshape((-1, D_c.shape[-1]-1)) # shape (output_dim, x_dim)
        # return raw_output[-1].reshape((-1, D_c.shape[-1]-1)) # shape (output_dim, x_dim)
    

class TrAgTransformer(Transformer):
    def __init__(self, scale_agnostic: bool = False, **kwargs):
        super().__init__(**kwargs)
        self.scale_agnostic = scale_agnostic

    def forward(self, X_c, y_c):
        X_c_scale = torch.tensor([1.0])
        if self.scale_agnostic:
            if X_c.shape[0] > 1:
                X_c_scale = X_c.std(0)
        X_c_mean = X_c.mean(0)
        X_c_tilde = (X_c - X_c_mean) / X_c_scale
        return super().forward(X_c_tilde, y_c) * X_c_scale + X_c_mean