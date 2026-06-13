"""Model definitions for SMS spam classification."""

from __future__ import annotations

import torch
from torch import nn


class TextMLP(nn.Module):
    """A basic MLP for TF-IDF SMS classification."""

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
        """Return unnormalized class logits."""
        return self.network(features)
