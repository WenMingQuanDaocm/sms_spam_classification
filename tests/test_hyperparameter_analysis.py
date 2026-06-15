"""超参数实验辅助函数的单元测试。"""

from __future__ import annotations

import unittest

import pandas as pd

from src.hyperparameter_analysis import (
    build_experiment_config,
    has_overfitting_signal,
    has_underfitting_signal,
    select_best_learning_rate,
    training_was_stable,
)


class HyperparameterAnalysisTests(unittest.TestCase):
    """测试第六阶段摘要辅助函数的确定性行为。"""

    def test_build_experiment_config_applies_overrides(self) -> None:
        # 覆盖项只改变指定超参数，其余默认配置应保持不变。
        config = build_experiment_config(learning_rate=0.01, dropout=0.5)

        self.assertEqual(config["learning_rate"], 0.01)
        self.assertEqual(config["dropout"], 0.5)
        self.assertEqual(config["hidden_layers"], (128, 64))

    def test_select_best_learning_rate_uses_macro_f1_then_accuracy(self) -> None:
        # 学习率选择先看 Macro-F1，Macro-F1 相同时再看 Accuracy。
        results = pd.DataFrame(
            [
                {"learning_rate": 0.01, "validation_macro_f1": 0.9, "validation_accuracy": 0.95},
                {"learning_rate": 0.001, "validation_macro_f1": 0.9, "validation_accuracy": 0.96},
                {"learning_rate": 0.0001, "validation_macro_f1": 0.8, "validation_accuracy": 0.99},
            ]
        )

        self.assertEqual(select_best_learning_rate(results), 0.001)

    def test_training_signal_helpers(self) -> None:
        # 这组历史记录模拟训练集继续变好、验证集变差的过拟合迹象。
        history = pd.DataFrame(
            [
                {
                    "train_loss": 0.5,
                    "validation_loss": 0.4,
                    "train_accuracy": 0.9,
                    "validation_accuracy": 0.89,
                    "validation_macro_f1": 0.8,
                },
                {
                    "train_loss": 0.01,
                    "validation_loss": 0.6,
                    "train_accuracy": 1.0,
                    "validation_accuracy": 0.9,
                    "validation_macro_f1": 0.92,
                },
                {
                    "train_loss": 0.005,
                    "validation_loss": 0.8,
                    "train_accuracy": 1.0,
                    "validation_accuracy": 0.88,
                    "validation_macro_f1": 0.9,
                },
            ]
        )

        self.assertTrue(training_was_stable(history))
        self.assertTrue(has_overfitting_signal(history))
        self.assertFalse(has_underfitting_signal(history))


if __name__ == "__main__":
    unittest.main()
