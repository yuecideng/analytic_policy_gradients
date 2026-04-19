import random
import torch
import numpy as np
import warp as wp


def set_seed(seed: int, torch_deterministic: bool = True):
    """Set a common random seed for reproducibility across random, numpy, torch, and warp."""
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.backends.cudnn.deterministic = torch_deterministic
    torch.backends.cudnn.benchmark = not torch_deterministic
    wp.set_seed(seed)
