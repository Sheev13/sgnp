import torch
from torch import nn

def ordering_constraint(x: torch.Tensor):
    # x has shape (batch, x_dim)

    batch = x.shape[0]
    ref_i = (batch // 2) + 1
    ref = x[ref_i, :].unsqueeze(0) * 0.1
    output = torch.zeros_like(x)

    lower_raw = x[:ref_i, :]
    if ref_i < 2: # if only a single element is selected
        lower_raw = lower_raw.unsqueeze(0) # ensure still rank 2 tensor
    lower = (lower_raw.exp()/batch).cumsum(dim=0).flip(0)
    if ref_i+1 < batch:
        upper_raw = x[ref_i+1:, :]
        if batch - (ref_i+1) == 1: # if only a single element is selected
            upper_raw = upper_raw.unsqueeze(0) # ensure still rank 2 tensor
        upper = (upper_raw.exp()/batch).cumsum(dim=0)
    

    if ref_i > 0:
        output[:ref_i, :] = -lower + ref
    output[ref_i, :] = ref
    if ref_i+1 < batch:
        output[ref_i+1:, :] = upper + ref

    # print(output)

    return output
