import torch
import numpy as np
import random
from scipy.stats import spearmanr
from copy import deepcopy

def set_random_seed(seed, deterministic=True):
    """Set random seed for reproducibility"""
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    if deterministic:
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False

def pearson_r(x, y):
    """Calculate Pearson correlation coefficient"""
    vx = x - x.mean()
    vy = y - y.mean()
    corr = torch.sum(vx * vy) / (torch.sqrt(torch.sum(vx ** 2)) * torch.sqrt(torch.sum(vy ** 2)) + 1e-8)
    return corr.item()


def spearman_rho(x, y):
    """Calculate Spearman rank correlation coefficient"""
    x_np = x.cpu().numpy()
    y_np = y.cpu().numpy()
    rho, _ = spearmanr(x_np, y_np)
    return rho


def compute_metrics(predictions, targets):
    """
    Compute all evaluation metrics
    
    Args: 
        predictions: torch.Tensor of predictions
        targets: torch.Tensor of ground truth values
        
    Returns:
        tuple: (mse, rmse, mae, pearson_r, spearman_rho)
    """
    predictions = predictions.view(-1)
    targets = targets.view(-1)
    
    mse = torch.mean((predictions - targets) ** 2).item()
    rmse = torch.sqrt(torch.mean((predictions - targets) ** 2)).item()
    mae = torch.mean(torch.abs(predictions - targets)).item()
    r = pearson_r(predictions, targets)
    rho = spearman_rho(predictions, targets)
    
    return mse, rmse, mae, r, rho


def pad_pairformer_embeddings_to_device(single_embeddings, pair_embeddings, device):
    """
    Pad PairFormer embeddings on GPU for efficient batch processing.
    
    This function takes variable-length embeddings on CPU and pads them
    directly on GPU to avoid CPU memory overhead.
    
    Args:
        single_embeddings: List[Tensor(L_i, D_s)] on CPU
            List of single (node) embeddings with varying sequence lengths
        pair_embeddings:  List[Tensor(L_i, L_i, D_p)] on CPU
            List of pair (edge) embeddings with varying sequence lengths
        device: torch.device
            Target device (typically CUDA)
    
    Returns:
        tuple: (padded_singles, padded_pairs)
            - padded_singles:  Tensor(B, L_max, D_s) on GPU
            - padded_pairs:  Tensor(B, L_max, L_max, D_p) on GPU
    
    Example:
        >>> single_emb, pair_emb = pad_pairformer_embeddings_to_device(
        ...    single_list, pair_list, device='cuda: 0'
        ...)
        >>> predictions = model(bg, bg3, single_emb, pair_emb, ...)
    """
    max_len = max(s.size(0) for s in single_embeddings)
    batch_size = len(single_embeddings)
    single_dim = single_embeddings[0].size(1)
    pair_dim = pair_embeddings[0].size(2)

    # Initialize padded tensors directly on GPU (zero-padded)
    final_singles = torch.zeros(batch_size, max_len, single_dim, device=device)
    final_pairs = torch.zeros(batch_size, max_len, max_len, pair_dim, device=device)

    # Copy individual embeddings to GPU and fill padded tensors
    for i, (single, pair) in enumerate(zip(single_embeddings, pair_embeddings)):
        seq_len = single.size(0)
        single_gpu = single.to(device=device, non_blocking=True)
        pair_gpu = pair.to(device=device, non_blocking=True)

        final_singles[i, :seq_len] = single_gpu
        final_pairs[i, :seq_len, :seq_len] = pair_gpu

    return final_singles, final_pairs


class WarmupScheduler: 
    """
    Learning rate warmup scheduler.
    
    Gradually increases learning rate from 0 to base_lr over warmup_epochs,
    then maintains base_lr for remaining training.
    
    Args:
        optimizer: torch.optim.Optimizer
            The optimizer to schedule
        warmup_epochs: int
            Number of warmup epochs
        base_lr: float
            Target learning rate after warmup
        updates_per_epoch: int
            Number of optimizer steps per epoch (for gradient accumulation)
    
    Example:
        >>> optimizer = torch.optim.AdamW(model.parameters(), lr=1e-4)
        >>> scheduler = WarmupScheduler(optimizer, warmup_epochs=3, base_lr=1e-4)
        >>> 
        >>> for epoch in range(max_epochs):
        >>>     for batch in dataloader:
        >>>         loss.backward()
        >>>         optimizer.step()
        >>>         scheduler.step()  # Call after each optimizer step
    """
    def __init__(self, optimizer, warmup_epochs, base_lr, updates_per_epoch=1):
        self.optimizer = optimizer
        self.warmup_steps = warmup_epochs * updates_per_epoch
        self.base_lr = base_lr
        self.current_step = 0
    
    def step(self):
        """Update learning rate (call after each optimizer.step())"""
        self.current_step += 1
        
        if self.current_step <= self.warmup_steps:
            # Linear warmup
            lr = self.base_lr * (self.current_step / self.warmup_steps)
        else:
            # Constant LR after warmup
            lr = self.base_lr
        
        # Update optimizer learning rate
        for param_group in self.optimizer.param_groups:
            param_group['lr'] = lr
        
        return lr
    
    def get_last_lr(self):
        """Get current learning rate"""
        return [param_group['lr'] for param_group in self.optimizer.param_groups]

