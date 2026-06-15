"""用于复现实验的通用工具函数。"""

from __future__ import annotations

import os
import random

import numpy as np
import torch

from src.config import RANDOM_STATE


def set_random_seed(seed: int = RANDOM_STATE) -> None:
    """设置 Python、NumPy 和 PyTorch 的随机种子。"""
    # 同时固定 Python、NumPy 和 PyTorch，减少不同运行之间的随机波动。
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    os.environ["PYTHONHASHSEED"] = str(seed)


def get_training_device() -> torch.device:
    """返回本项目使用的训练设备。"""
    return torch.device("cuda" if torch.cuda.is_available() else "cpu")
