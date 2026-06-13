"""Unit tests for PyTorch model definitions."""

from __future__ import annotations

import unittest

import torch

from src.models import TextMLP
from src.utils import set_random_seed


class ModelTests(unittest.TestCase):
    """Test MLP construction and forward output shape."""

    def test_text_mlp_forward_shape(self) -> None:
        set_random_seed(42)
        model = TextMLP(input_dim=10, hidden_layers=(8, 4), dropout=0.1)
        features = torch.ones((3, 10), dtype=torch.float32)

        logits = model(features)

        self.assertEqual(tuple(logits.shape), (3, 2))

    def test_text_mlp_rejects_invalid_dropout(self) -> None:
        with self.assertRaises(ValueError):
            TextMLP(input_dim=10, hidden_layers=(8,), dropout=1.0)


if __name__ == "__main__":
    unittest.main()
