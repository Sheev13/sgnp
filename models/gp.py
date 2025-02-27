import torch
from torch import nn
from .likelihoods import GaussianLikelihood

from typing import Optional

class GaussianProcess(nn.Module):
    """Represents an exact Gaussian Process.
    
    The implementation closely follows 
        'Gaussian Processes for Machine Learning'
        Rasmussen and Williams (2006).
    """
    def __init__(
        self,
        num_inputs: int = None,
        prior: nn.Module = None,
        sigma_y: float = 0.01,
        train_sigma_y: bool = False,
    ):
        super().__init__()
        self.num_inputs = num_inputs
        self.prior = prior
        self.likelihood = GaussianLikelihood(sigma_y=sigma_y, train_sigma_y=train_sigma_y)

    def p_fn(self, X_test: torch.Tensor, X: torch.Tensor, y: torch.Tensor, multivariate: bool = False):
        """returns the posterior distribution over functions evaluated at X_test
        i.e. p(f(X_test)|D). This implements standard GP posterior distribution
        equations that can be found in e.g. Rasmussen & Williams 2006.
        """
        K_nn = self.prior.covariance_function(X) + self.likelihood.sigma_y.pow(2) * torch.eye(X.shape[0])
        L_nn = torch.linalg.cholesky(K_nn)
        K_nn_inv = torch.cholesky_inverse(L_nn)

        K_tn = self.prior.covariance_function(X_test, X)
        K_tt = self.prior.covariance_function(X_test)

        mu = K_tn @ K_nn_inv @ y
        covar = K_tt - (K_tn @ K_nn_inv @ K_tn.T)
        if multivariate:
            return torch.distributions.MultivariateNormal(loc=mu.squeeze(), covariance_matrix=covar)
        else:
            return torch.distributions.Normal(mu.squeeze(), covar.diagonal())

    def forward(self, X_test: torch.Tensor = None, X: Optional[torch.Tensor] = None, y: Optional[torch.Tensor] = None, multivariate: bool = False):
        """the primary prediction function of the GP for users. X_test specifies the
        inputs at which to obtain predictions. X and y are the inputs and outputs 
        respectively in the dataset. If they are not specified, e.g. left as None,
        this function returns the GP prior distribution over the test points. Otherwise,
        it returns the posterior predictive distribution over the test points.
        """
        if X is None:
            return self.prior(X_test) # useful if the user wants to sample from GP prior
        assert y is not None
        p_fn_test = self.p_fn(X_test, X, y, multivariate=multivariate)
        return self.likelihood.posterior_predictive(p_fn_test)

    def loss(self, X, y):
        """Computes the log marginal likelihood via standard equations that can be
        found in e.g. Rasmussen & Williams 2006. Since torch optimisers do gradient
        *descent*, this returns the *negative* log marginal likelihood. It also
        returns a dictionary of useful metrics including the log marginal likelihood
        and any trainable hyperparameters.
        """
        # objective function is the marginal likelihood
        K_nn = self.prior.covariance_function(X)
        chol = torch.linalg.cholesky(K_nn + self.likelihood.sigma_y**2*torch.eye(X.shape[0]))
        inv = torch.cholesky_inverse(chol)

        a = -0.5 * y.T@inv@y
        b = -0.5 * torch.linalg.det(inv).pow(-1).log()
        c = - X.shape[0]/2 * (torch.tensor(2) * torch.pi).log()
        lml = (a + b + c).squeeze()

        metrics = {
            "lml": lml.detach().item(),
        }
        
        # obtain trainable hyperparams and their values.
        # Many of these are implemented in log-space, so
        # we need to transform them into regular space.
        for name, param in self.prior.named_parameters():
            if param.requires_grad:
                if (self.prior.covariance_function.ard and name == 'covariance_function.log_l'):
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
            
        return - lml, metrics