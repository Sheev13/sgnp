import torch
from torch import nn
from .covariance_functions import Exponential, Matern1_5, Matern2_5, SquaredExponential, Periodic, Tetouan
from typing import Optional

class GPPrior(nn.Module):
    """Represents a Gaussian Process Prior.
    
    Args:
        num_inputs: 
            an integer denoting the number of input dimensions.
        covariance_function:
            a string denoting the choice of covariance function. Options are:
                'exponential',
                'matern-1.5',
                'matern-2.5',
                'squared-exponential'.
            Default: 'squared-exponential'.
        mean_function:
            a string denoting the choice of prior mean function. Options are:
                'zero',
                'constant',
            Default: 'zero'.
        l: 
            a positive float representing the lengthscale hyperparameter of the 
            covariance function. If ARD is being used, this is the (initial) 
            lengthscale for every dimension, unless any fixed lengthscales are 
            specified via `fixed_ls`.
            Default: 1.0.
        train_l:
            a boolean flag denoting whether or not the lengthscale(s) should 
            be optimised along with any other hyper and/or variational parameters.
            Default: False.
        fixed_ls:
            an optional argument that contains a dictionary of feature index
            (key) lengthscale (value) pairs that are to be held fixed if ARD 
            is being used.
            Default: None.
        ard:
            a boolean flag denoting whether or not to have different lengthscales
            for different feature dimensions. This is only useful if `fixed_ls` 
            True so that different lengthscales can be learned.
            Default: False
    """
    
    def __init__(
        self,
        covariance_function: str = 'squared-exponential',
        **kwargs,
    ):
        super().__init__()
        
        # initialise covariance function object
        covariance_function = covariance_function.lower()
        implemented_covfunc_names = ['exponential', 'matern-1.5', 'matern-2.5', 'squared-exponential', 'periodic', 'tetouan']
        implemented_covfunc_objs = [Exponential, Matern1_5, Matern2_5, SquaredExponential, Periodic, Tetouan]
        if covariance_function not in implemented_covfunc_names:
            raise NotImplementedError(f"{covariance_function} either contains a typo or corresponds to a covariance function not yet implemented")
        for i in range(len(implemented_covfunc_names)):
            if covariance_function == implemented_covfunc_names[i]:
                self.covariance_function = implemented_covfunc_objs[i](**kwargs)
            
    def forward(self, inputs: torch.tensor):            
        mu = torch.zeros_like(inputs[:,0])
        cov = self.covariance_function(inputs)
        return torch.distributions.MultivariateNormal(mu, cov)