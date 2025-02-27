import torch

def UCB(model, X, beta=0.2, X_c=None, y_c=None):
    with torch.no_grad():
        posterior = model(X, X_c, y_c, multivariate=False, q_f=True)
    mu, sigma = posterior.mean, posterior.std
    return mu + torch.tensor(beta).sqrt() * sigma
