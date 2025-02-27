import torch

def ackley(X: torch.Tensor):
    # expect X to be shape (batch, dim)
    while len(X.shape) < 2:
        X = X.unsqueeze(0)

    X = X * 5 # just to work with nicer domain

    cos_term = torch.cos(2 * torch.pi * X).mean(-1)
    sqrt_term = -0.2 * X.pow(2).mean(-1).sqrt()

    return 2 * sqrt_term.exp() + 0.1 * cos_term.exp() - 0.1 * torch.exp(torch.tensor(1.0)) 

def sphere(X: torch.Tensor):
    # expect X to be shape (batch, dim)
    while len(X.shape) < 2:
        X = X.unsqueeze(0)

    X = X / 2 # just to work with nicer domain

    return 3 - X.pow(2).mean(-1)

def noisy_function(X: torch.Tensor, std: float = 0.05, func=None):
    if func is None:
        raise ValueError("User must specify a function.")
    elif func.lower() == 'ackley':
        f = ackley
    elif func.lower() == 'sphere':
        f = sphere
    else:
        raise NotImplementedError("User-specified function not recognised.")
    return f(X) + torch.randn((X.shape[0],)) * std

def generate_noisy_dataset(dim=1, n=20, limit=10, std=0.0, func=None):
    X = torch.rand((n, dim)) * 2 * limit - limit
    y = noisy_function(X, std=std, func=func)
    return X, y
