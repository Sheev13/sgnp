import torch

def obtain_me_a_nice_gp_dataset_please(x_range=[-3.0, 3.0], n_range=[5, 100], noise=0.05, l=1.0, noise_range=None, l_range=None, kernel='se', p=1.0, p_range=None, binary_2d=False):
    n = torch.randint(low=min(n_range), high=max(n_range), size=(1,))
    if binary_2d:
        X = torch.rand((n, 2)) * (max(x_range) - min(x_range)) + min(x_range)
    else:
        X = torch.rand((n, 1)) * (max(x_range) - min(x_range)) + min(x_range)

    if noise_range is not None:
        noise = torch.rand((1,)) * (max(noise_range) - min(noise_range)) + min(noise_range)
    if l_range is not None:
        l = torch.rand((1,)) * (max(l_range) - min(l_range)) + min(l_range)
    if p_range is not None:
        p = torch.rand((1,)) * (max(p_range) - min(p_range)) + min(p_range)
    
    if kernel == 'se':
        Sigma = torch.exp(-0.5 * torch.cdist(X/l, X/l, p=2).square())
    elif kernel == 'per':
        Sigma = torch.exp(-2.0 * torch.sin(torch.pi * torch.cdist(X, X) / p).square() / l**2) 
    Sigma += + torch.eye(n) * 1e-5
    z = torch.randn((n, 1))
    f_x = torch.linalg.cholesky(Sigma) @ z
    if binary_2d:
        y = torch.distributions.Bernoulli(logits=f_x * 2.5).sample()
    else:
        y = f_x + torch.randn_like(f_x) * noise
    return X, y