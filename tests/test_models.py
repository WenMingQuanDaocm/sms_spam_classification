"""PyTorch 模型定义的单元测试。"""

from __future__ import annotations

import unittest

import torch

from src.models import TextMLP
from src.utils import set_random_seed


class ModelTests(unittest.TestCase):
    """测试 MLP 构建和前向输出形状。"""

    def test_text_mlp_forward_shape(self) -> None:
        # MLP 输出应为每个样本两个类别的 logits。
        set_random_seed(42)
        model = TextMLP(input_dim=10, hidden_layers=(8, 4), dropout=0.1)
        features = torch.ones((3, 10), dtype=torch.float32)

        logits = model(features)

        self.assertEqual(tuple(logits.shape), (3, 2))

    def test_text_mlp_rejects_invalid_dropout(self) -> None:
        # Dropout=1.0 会丢弃全部隐藏单元，因此配置校验应拒绝。
        with self.assertRaises(ValueError):
            TextMLP(input_dim=10, hidden_layers=(8,), dropout=1.0)


if __name__ == "__main__":
    unittest.main()
