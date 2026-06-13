"""Unit tests for final evaluation helpers."""

from __future__ import annotations

import unittest

import pandas as pd

from src.config import LABEL_COLUMN, MESSAGE_COLUMN, TARGET_COLUMN
from src.final_evaluation import build_prediction_frame


class FinalEvaluationTests(unittest.TestCase):
    """Test final prediction frame construction."""

    def test_build_prediction_frame_adds_labels_and_probabilities(self) -> None:
        test_data = pd.DataFrame(
            [
                {MESSAGE_COLUMN: "hello", LABEL_COLUMN: "ham", TARGET_COLUMN: 0},
                {MESSAGE_COLUMN: "win prize", LABEL_COLUMN: "spam", TARGET_COLUMN: 1},
            ]
        )

        predictions = build_prediction_frame(
            test_data,
            predicted_targets=[0, 1],
            spam_probabilities=[0.2, 0.9],
        )

        self.assertEqual(predictions["predicted_label"].tolist(), ["ham", "spam"])
        self.assertEqual(predictions["spam_probability"].tolist(), [0.2, 0.9])


if __name__ == "__main__":
    unittest.main()
