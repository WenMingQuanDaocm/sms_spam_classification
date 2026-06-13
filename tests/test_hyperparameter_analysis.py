"""Unit tests for hyperparameter experiment helpers."""

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
    """Test deterministic helper behavior for phase-six summaries."""

    def test_build_experiment_config_applies_overrides(self) -> None:
        config = build_experiment_config(learning_rate=0.01, dropout=0.5)

        self.assertEqual(config["learning_rate"], 0.01)
        self.assertEqual(config["dropout"], 0.5)
        self.assertEqual(config["hidden_layers"], (128, 64))

    def test_select_best_learning_rate_uses_macro_f1_then_accuracy(self) -> None:
        results = pd.DataFrame(
            [
                {"learning_rate": 0.01, "validation_macro_f1": 0.9, "validation_accuracy": 0.95},
                {"learning_rate": 0.001, "validation_macro_f1": 0.9, "validation_accuracy": 0.96},
                {"learning_rate": 0.0001, "validation_macro_f1": 0.8, "validation_accuracy": 0.99},
            ]
        )

        self.assertEqual(select_best_learning_rate(results), 0.001)

    def test_training_signal_helpers(self) -> None:
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
