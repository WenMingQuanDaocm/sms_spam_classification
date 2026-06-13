"""Shared utility functions for reproducible experiments."""

from __future__ import annotations

import os
import random

import numpy as np
import torch

from src.config import RANDOM_STATE


def set_random_seed(seed: int = RANDOM_STATE) -> None:
    """Set Python, NumPy, and PyTorch random seeds."""
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    os.environ["PYTHONHASHSEED"] = str(seed)


def get_training_device() -> torch.device:
    """Return the training device used by this project."""
    return torch.device("cuda" if torch.cuda.is_available() else "cpu")
