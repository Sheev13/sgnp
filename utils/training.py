import torch
from torch import nn
from tqdm import tqdm
from typing import Optional, Dict, Tuple, List
from collections import defaultdict
import warnings

from .data_utils import MetaDataset, ctxt_trgt_split

def train_meta_model(
    model,
    md: List[torch.Tensor],
    epochs: Optional[int] = None,
    training_steps: Optional[int] = None,
    learning_rate: float = 1e-2,
    num_samples: int = 1,
    batch_size: Optional[int] = None,
    final_learning_rate: Optional[float] = None,
    max_gradient: Optional[float] = None,
    unfreeze_trainable_hypers_at_step: Optional[int] = None,
    use_gpu: bool = False,
    loss_function: str = 'vi', # one of 'vi', 'npvi', 'npml'
    ctxt_proportion_range: Tuple[float] = (0.4, 0.8),
    es_thresh: Optional[float] = None,
    include_ctxt_in_trgt: bool = False,
    task_subsample_fraction: Optional[float] = None,
) -> Dict:
    
    if loss_function not in ['vi', 'npvi', 'npml']:
        raise ValueError(f"Loss function: {loss_function} not recognised.")
    
    # set device to gpu if user wants to and one is available.
    if use_gpu:
        if torch.cuda.is_available():
            device = torch.device('cuda')
        elif torch.backends.mps.is_available():
            device = torch.device('mps')
        else:
            print("No GPU found, falling back to CPU")
            device = torch.device('cpu')
        torch.set_default_device(device)
        torch.set_default_dtype(torch.float32)
        model.to(device, dtype=torch.float32)
        print("Moving dataset to device...")
        md = [(X.to(device=device, dtype=torch.float32), y.to(device=device, dtype=torch.float32)) for (X, y) in md]
        print("Done.")
    else:
        device = torch.device('cpu')
        
    meta_dataset = MetaDataset(md)

    # move dataset into torch dataloader
    generator = torch.Generator(device=device)
    sampler = torch.utils.data.RandomSampler(meta_dataset, generator=generator)
    dataloader = torch.utils.data.DataLoader(meta_dataset, sampler=sampler)
    dataset_iterator = iter(dataloader)

    # handle batch size, set number of training steps based on user's epochs or training steps input
    num_datasets = len(meta_dataset)
    if batch_size is None:
        batch_size = num_datasets # i.e. no minibatching of the metadataset
    if batch_size > num_datasets:
        warnings.warn("Batch_size is larger than the number of datasets available. Defaulting to 'batch_size = num_datasets'.")
        batch_size = num_datasets
    if (epochs is None) == (training_steps is None):
        if training_steps is None:
            training_steps = 10_000
        warnings.warn(f"Exactly one of `epochs` or `training_steps` must be set. Defaulting to training_steps={training_steps}")
    elif training_steps is None:
        training_steps = int(epochs * num_datasets / batch_size)

    # initialise optimiser and learning rate scheduler.
    optimiser = torch.optim.Adam(model.parameters(), lr=learning_rate)
    if final_learning_rate is not None:
        end_factor = final_learning_rate / learning_rate
    else:
        end_factor = 1.0
    lr_sched = torch.optim.lr_scheduler.LinearLR(
        optimiser, start_factor=1.0, end_factor=end_factor, total_iters=training_steps
    )

    if es_thresh is not None:
        loss_window = - torch.ones((100,)) * 1e-4

    # if hypers are to be trained later, freeze the unfrozen
    # ones and store their names in order to know what to
    # unfreeze later on.
    if unfreeze_trainable_hypers_at_step is not None:
        trainable_params = []
        for name, param in model.likelihood.named_parameters():
            if param.requires_grad:
                trainable_params.append((name, param))
                param.requires_grad = False
        for name, param in model.prior.named_parameters():
            if param.requires_grad:
                trainable_params.append((name, param))
                param.requires_grad = False

    # initialise metrics tracker.
    tracker = defaultdict(list)
    pbar = tqdm(range(training_steps))
    
    # main training loop here.
    for training_step in pbar:

        if unfreeze_trainable_hypers_at_step is not None:
            if training_step == unfreeze_trainable_hypers_at_step:
                for name, param in trainable_params:
                    param.requires_grad = True
                
        optimiser.zero_grad()
        batch_loss = torch.tensor(0.0)
        batch_metrics = defaultdict(float)
        for _ in range(batch_size):
            try:
                (X, y) = next(dataset_iterator)
            except StopIteration:
                dataset_iterator = iter(dataloader)
                (X, y) = next(dataset_iterator)

            X, y = X.squeeze(0), y.squeeze(0)
            if task_subsample_fraction is not None:
                b = int(task_subsample_fraction * X.shape[0])
                b_inds = torch.randperm(X.shape[0])[:b]
                X, y = X[b_inds], y[b_inds]

            if loss_function == 'vi':
                X_c, y_c, X_t, y_t = X, y, X, y
            else:
                X_c, y_c, X_t, y_t = ctxt_trgt_split(X, y, ctxt_proportion_range=ctxt_proportion_range)
                if include_ctxt_in_trgt:
                    X_t, y_t = X, y
            if loss_function == 'npml':
                loss, metrics = model.loss(X_c, y_c, X_t, y_t, num_samples=num_samples, use_kl=False)
            else:
                try:
                    loss, metrics = model.loss(X_c, y_c, X_t, y_t, num_samples=num_samples)
                except ValueError:
                    print("Handled Value Error")
                    loss, metrics = model.loss(X_c, y_c, X_t, y_t, num_samples=num_samples)

            batch_loss += loss / batch_size
            for key, value in metrics.items():
                batch_metrics[key] += value / batch_size
        
        batch_loss.backward()

        # clip gradients if desired.
        if max_gradient is not None:
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_gradient) 
        # handle any NaN gradients

        for p in model.parameters():
            if p.grad is not None:
                if p.grad.data.isnan().any():
                    p.grad.data = torch.nan_to_num(p.grad.data)
                    warnings.warn("Warning: NaN gradients encountered. Proceeded by setting them to zero.")

        # perform gradient update step.
        optimiser.step()
        lr_sched.step()        

        # store metrics.
        for key, value in batch_metrics.items():
            tracker[key].append(float(value))

        batch_metrics["Epochs"] = (training_step * batch_size) / num_datasets

        pbar.set_postfix(batch_metrics)
    
        # handle early stopping
        if es_thresh is not None:
            loss_window = torch.roll(loss_window, 1)
            loss_window[0] = - batch_loss.detach().item()
            if loss_window.mean() > es_thresh:
                break
        
    # return model to cpu if relevant.
    if use_gpu and (torch.cuda.is_available() or torch.backends.mps.is_available()):
        print("Returning model and data to CPU.")
        cpu_device = torch.device('cpu')
        torch.set_default_device(cpu_device)
        torch.set_default_dtype(torch.float64)
        model.to(cpu_device)
        model.to(dtype=torch.float64)

    return tracker







def train_gp(
    model,
    X: torch.Tensor,
    y: torch.Tensor,
    epochs: int = 100,
    learning_rate: float = 1e-2,
    final_learning_rate: Optional[float] = None,
    max_gradient: Optional[float] = None,
    use_gpu: bool = False,
    silent: bool = False,
    svgp: bool = False,
    num_samples: int = 1,
) -> torch.Tensor:
    

    # set device to gpu if user wants to and one is available.
    if use_gpu:
        if torch.cuda.is_available():
            device = torch.device('cuda')
        elif torch.backends.mps.is_available():
            device = torch.device('mps')
        else:
            print("No GPU found, falling back to CPU")
            device = torch.device('cpu')
        torch.set_default_device(device)
        model.to(device)
        print("Moving dataset to device...")
        X = X.to(device)
        t = t.to(device)
        print("Done.")
    else:
        device = torch.device('cpu')

    # initialise optimiser and learning rate scheduler.
    optimiser = torch.optim.Adam(model.parameters(), lr=learning_rate)
    if final_learning_rate is not None:
        end_factor = final_learning_rate / learning_rate
    else:
        end_factor = 1.0
    lr_sched = torch.optim.lr_scheduler.LinearLR(
        optimiser, start_factor=1.0, end_factor=end_factor, total_iters=epochs
    )

    # initialise metrics tracker.
    tracker = defaultdict(list)
    pbar = tqdm(range(epochs), disable=silent)
    
    # main training loop here.
    for _ in pbar:
    
        # compute loss and gradients.
        optimiser.zero_grad()
        if svgp:
            try:
                loss, metrics = model.loss(X, y, X, y, num_samples=num_samples)
            except ValueError:
                # print("Handled Value Error")
                loss, metrics = model.loss(X, y, X, y, num_samples=num_samples)
            # loss, metrics = model.loss(
            #     X, y, X, y, num_samples=num_samples
            # )
        else:
            loss, metrics = model.loss(
                X,
                y,
            )

        loss.backward()
        
        # clip gradients if desired.
        if max_gradient is not None:
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_gradient) 
        for p in model.parameters():
            if p.grad is not None:
                if p.grad.data.isnan().any():
                    p.grad.data = torch.nan_to_num(p.grad.data)
                    warnings.warn("Warning: NaN gradients encountered. Proceeded by setting them to zero.")
        
        # perform gradient update step.
        optimiser.step()
        lr_sched.step()
        
        # store metrics.
        for key, value in metrics.items():
            tracker[key].append(float(value))

        pbar.set_postfix(metrics)
        
    # return model to cpu if relevant.
    if use_gpu and torch.cuda.is_available():
        print("Returning model and tensors to CPU.")
        cpu_device = torch.device('cpu')
        torch.set_default_device(cpu_device)
        model.to(cpu_device)
        X = X.to(device)
        t = t.to(device)

    return tracker