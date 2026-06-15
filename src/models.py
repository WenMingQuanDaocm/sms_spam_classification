"""短信垃圾分类模型定义。"""

from __future__ import annotations

import torch
from torch import nn


class TextMLP(nn.Module):
    """用于 TF-IDF 短信分类的基础 MLP 模型。"""

    def __init__(
        self,
        input_dim: int,
        hidden_layers: tuple[int, ...] = (128, 64),
        dropout: float = 0.3,
        output_dim: int = 2,
    ) -> None:
        super().__init__()
        if input_dim <= 0:
            raise ValueError("input_dim must be positive.")
        if not hidden_layers:
            raise ValueError("hidden_layers must contain at least one layer.")
        if not 0.0 <= dropout < 1.0:
            raise ValueError("dropout must be in [0.0, 1.0).")

        # 根据 hidden_layers 动态搭建全连接层，便于后续扩展结构实验。
        layers: list[nn.Module] = []
        previous_dim = input_dim
        for hidden_dim in hidden_layers:
            if hidden_dim <= 0:
                raise ValueError("hidden layer sizes must be positive.")
            layers.extend(
                [
                    nn.Linear(previous_dim, hidden_dim),
                    nn.ReLU(),
                    nn.Dropout(dropout),
                ]
            )
            previous_dim = hidden_dim
        layers.append(nn.Linear(previous_dim, output_dim))
        self.network = nn.Sequential(*layers)

    def forward(self, features: torch.Tensor) -> torch.Tensor:
        """返回未归一化的类别 logits。"""
        return self.network(features)
