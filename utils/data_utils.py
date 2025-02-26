from typing import List, Any, Tuple

import torch
from torch.utils.data import Dataset

class MetaDataset(Dataset):
    def __init__(self, datasets: List[Any]):
        self.datasets = datasets

    def __len__(self):
        return len(self.datasets)

    def __getitem__(self, idx: int):
        return self.datasets[idx]


def ctxt_trgt_split(
    X: torch.Tensor, y: torch.Tensor, ctxt_proportion_range: Tuple[float]):
    if ctxt_proportion_range[1] < ctxt_proportion_range[0]:
        ctxt_proportion_range = ctxt_proportion_range[::-1]
    if ctxt_proportion_range[0] < 0.0:
        raise ValueError("Cannot have a negative proportion of context points.")
    if ctxt_proportion_range[1] > 1.0:
        raise ValueError("Cannot have a proportion of context points that is greater than 1.")

    
    proportion = torch.rand((1,)) * (ctxt_proportion_range[1] - ctxt_proportion_range[0]) + ctxt_proportion_range[0]

    num_ctxt = int(X.shape[0] * proportion)
    inds = torch.randperm(X.shape[0])
    ctxt_i = inds[:num_ctxt]
    trgt_i = inds[num_ctxt:]

    X_c, y_c = X[ctxt_i], y[ctxt_i]
    X_t, y_t = X[trgt_i], y[trgt_i]

    if X_c.shape[0] == 0:
        X_c, y_c = X_t[0], y_t[0]
    elif X_t.shape[0] == 0:
        X_t, y_t = X_c[0], y_c[0]

    return (X_c, y_c, X_t, y_t)