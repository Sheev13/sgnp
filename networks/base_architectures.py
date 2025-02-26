from typing import List, Optional

import torch
from torch import nn
import torch.nn.functional as F

class MLP(nn.Module):
    def __init__(
        self,
        dims: List[int],
        nonlinearity: nn.Module = nn.ReLU(),
    ):
        super().__init__()
        net = []
        for i in range(len(dims) - 1):
            net.append(nn.Linear(dims[i], dims[i + 1]))

            if i < len(dims) - 2:
                net.append(nonlinearity)

        self.net = nn.Sequential(*net)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x)
    
class CNN(nn.Module):
    def __init__(
        self,
        dims: int,
        chans: List[int],
        kernel_size: int = 5,
        nonlinearity: nn.Module = nn.ReLU(),
    ):
        super().__init__()
        if dims not in [1, 2, 3]:
            raise ValueError(f"Only 1-, 2-, and 3-D convolutions supported, not {dims}-D.")
        
        conv_layer = {1: nn.Conv1d, 2: nn.Conv2d, 3: nn.Conv3d}[dims]
        net = []
        for i in range(len(chans) - 1):
            net.append(conv_layer(chans[i], chans[i+1], kernel_size=kernel_size, padding='same'))

            if i < len(chans) - 2:
                net.append(nonlinearity)

        self.net = nn.Sequential(*net)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x)
    

class UNet(nn.Module):
    def __init__(
        self,
        dims: int,
        input_chans: int,
        output_chans: int,
        base_chans: int = 16,
        depth: int = 4,
        kernel_size: int = 5,
        nonlinearity: nn.Module = nn.ReLU(),        
    ):
        super().__init__()
        if dims not in [1, 2, 3]:
            raise ValueError(f"Only 1-, 2-, and 3-D convolutions supported, not {dims}-D.")
        conv_layer = {1: nn.Conv1d, 2: nn.Conv2d, 3: nn.Conv3d}[dims]
        conv_trnsps_layer = {1: nn.ConvTranspose1d, 2: nn.ConvTranspose2d, 3: nn.ConvTranspose3d}[dims]

        self.first_conv = conv_layer(input_chans, base_chans, kernel_size=1, padding='same')
        self.nonlinearity = nonlinearity
        self.down_layers = []
        up_layers = []
        self.final_conv = conv_layer(base_chans * 2, output_chans, kernel_size=1, padding='same')

        max_chans = base_chans * 2**(depth//2)
        for i in range(depth):
            a = i // 2
            b = (i + 1) // 2
            self.down_layers.append(
                conv_layer(base_chans * 2**a, base_chans * 2**b, kernel_size=kernel_size, stride=2)
            )
            up_layers.append(
                conv_trnsps_layer(min(base_chans * 2**(b+1), max_chans), base_chans * 2**a, kernel_size=kernel_size, stride=2)
            )

        self.up_layers = up_layers[::-1]
        self.unet_depth = depth

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # pass through first conv layer:
        x = self.first_conv(x)

        # pass through first half of UNet:
        activations = [x]
        for i in range(self.unet_depth):
            x = self.nonlinearity(self.down_layers[i](x))
            if i < self.unet_depth - 1:
                activations.append(x)

        # pass through second half of UNet:
        activations = activations[::-1]
        for i in range(self.unet_depth):
            x = self.nonlinearity(self.up_layers[i](x))
            x = self._pad_to_match(x, reference=activations[i])
            # x = self._slice_to_match(x, reference=activations[i])
            x = torch.cat((activations[i], x), dim=0)

        # pass through final layer to get the desired number of channels:
        return self.final_conv(x)
        
    def _slice_to_match(self, x: torch.Tensor, reference: Optional[torch.Tensor] = None):
        if reference is None:
            raise ValueError("User must specify a refence tensor.")
        
    def _pad_to_match(self, x: torch.Tensor, reference: Optional[torch.Tensor] = None):
        # reference has shape (chans, grid_dim_1, ..., grid_dim_x_dim.)
        # x has same number of chans, but grid dims might be slightly
        # wrong, hence this function. 
        if reference is None:
            raise ValueError("User must specify a refence tensor.")
        
        ref_dims = reference.shape
        pad = []
        for i in range(1, len(ref_dims)):
            if x.shape[i] < ref_dims[i]:
                diff = ref_dims[i] - x.shape[i]
                if diff % 2 == 0: # if difference is even:
                    pad += [diff//2, diff//2]
                else: # if difference is odd:
                    pad += [diff//2 + 1, diff//2]
            else:
                pad += [0, 0]
        pad = tuple(pad[::-1])
        return F.pad(x, pad, mode='reflect')

            
        
        
        
            