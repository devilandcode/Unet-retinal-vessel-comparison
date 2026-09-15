"""Reproducible RNG for each CV fold."""

import os
import random

import numpy as np


def set_seed(seed: int) -> None:
    """Seed Python, NumPy and (if available) PyTorch. Called once per fold."""
    random.seed(seed)
    np.random.seed(seed)
    os.environ['PYTHONHASHSEED'] = str(seed)
    try:
        import torch
        torch.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False
    except ImportError:
        pass


def fold_train_seed(split_seed: int, fold: int) -> int:
    """Distinct training seed per fold (not the split seed itself)."""
    return int(split_seed) + 1000 + int(fold)
