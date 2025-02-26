"""
This file contains the following classes:
    GaussianLikelihood
    BernoulliLikelihood
"""
import torch
import torch.nn as nn

class GaussianLikelihood(nn.Module):
    """
    A class representing the Gaussian likelihood function. This should be used
    for regression tasks. It is the probabilistic analogue of a squared error 
    term in a loss function (i.e. log Gaussian likelihood == sum of square residuals)
    
    Args:
        sigma_y:
            a float representing the standard deviation of the observation noise/
            Gaussian likelihood.
            Default: 1e-2.
        
        train_sigma_y:
            a boolean flag denoting whether or not sigma_y should be optimised 
            along with the other hyper and variational parameters.
            Default: False
    """
    def __init__(self, sigma_y: float = 1e-2, train_sigma_y: bool = False):
        super().__init__()
        # parameterise the observation noise in log space since it must be positive.
        # clamp it at some small value to avoid the numerical instability caused by
        # miniscule observation noise values.
        self.log_sigma_y = nn.Parameter(torch.tensor(sigma_y).clamp(min=5e-3).log(), requires_grad=train_sigma_y)

    @property
    def sigma_y(self):
        return self.log_sigma_y.exp()
    
    def forward(self, f: torch.Tensor):
        # represents the transformation applied to f to get to y space.
        # For regression, f and y live in the same space.
        return f
    
    def log_prob(self, predictions: torch.Tensor = None, targets: torch.Tensor = None):
        # computes the log-likelihood. This is needed e.g. in ELBO for Monte Carlo
        # estimate of expected log-likelihood w.r.t. posterior distribution.
        dist = torch.distributions.Normal(predictions, self.sigma_y)
        return dist.log_prob(targets.unsqueeze(0).repeat((predictions.shape[0], 1)))

    def posterior_predictive(self, q_f: torch.distributions.Distribution):
        # p(y_*|x_*, D)        (crucially NOT p(f_*|...)=q_f as this includes observation noise)
        # q_f is the (possibly approximate) posterior over f.
        if isinstance(q_f, torch.distributions.Normal):
            mu = q_f.mean
            var = q_f.variance + self.sigma_y.pow(2)
            return torch.distributions.Normal(mu, var.sqrt())
        elif isinstance(q_f, torch.distributions.MultivariateNormal):
            mu = q_f.mean
            Sigma = q_f.covariance_matrix + torch.eye(mu.shape[0]) * self.sigma_y.pow(2)
            return torch.distributions.MultivariateNormal(mu, Sigma)
    

class BernoulliLikelihood(nn.Module):
    """
    A class representing the Bernoulli likelihood function. This should be used
    for classification tasks. This is the probabilistic analogue of the binary
    cross entropy loss (i.e. log Bernoulli likelihood == binary cross entropy).
    """
    def __init__(self):
        super().__init__()

    def forward(self, f: torch.Tensor):
        # represents the transformation applied to f to get to target space
        # for classification, the outputs must be probabilities between 0 and 1,
        # so the function must be squashed accordingly.
        return torch.sigmoid(f).clamp(min=1e-5, max=1.0-1e-5)
    
    def log_prob(self, predictions: torch.Tensor = None, targets: torch.Tensor = None):
        # computes the log-likelihood. This is needed e.g. in ELBO for Monte Carlo
        # estimate of expected log-likelihood w.r.t. posterior distribution.
        targets_stack = targets.unsqueeze(0).repeat((predictions.shape[0], 1))
        return (targets_stack * torch.log(predictions) + (1 - targets_stack) * torch.log(1 - predictions))

    def posterior_predictive(self, q_f: torch.distributions.Distribution):
        # p(t_*|x_*, D)
        if isinstance(q_f, torch.distributions.Normal):
            fn_means = q_f.mean
            fn_vars = q_f.variance
        elif isinstance(q_f, torch.distributions.MultivariateNormal):
            fn_means = q_f.mean
            fn_vars = q_f.covariance_matrix.diagonal()
        # following Pattern Recognition and Machine Learning, Chris Bishop, pp.218-220
        # (our fn[i] is their a).
        # This is known as the probit approximation to the logistic function.
        return torch.distributions.Bernoulli(probs=torch.sigmoid(fn_means / (1 + torch.pi*fn_vars/8).pow(0.5)))