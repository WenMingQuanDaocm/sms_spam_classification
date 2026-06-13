"""Unit tests for preprocessing, splitting, and TF-IDF preparation."""

from __future__ import annotations

import unittest

import pandas as pd

from src.config import LABEL_COLUMN, MESSAGE_COLUMN, TARGET_COLUMN
from src.preprocess import (
    clean_sms_data,
    find_split_overlaps,
    fit_tfidf_vectorizer,
    get_label_conflicts,
    normalize_raw_columns,
    split_train_validation_test,
    transform_splits,
)


def build_sample_raw_data() -> pd.DataFrame:
    """Build a balanced synthetic dataset for deterministic unit tests."""
    rows: list[dict[str, str]] = []
    for index in range(10):
        rows.append(
            {
                LABEL_COLUMN: "ham",
                MESSAGE_COLUMN: f"safe common meeting message {index}",
            }
        )
        rows.append(
            {
                LABEL_COLUMN: "spam",
                MESSAGE_COLUMN: f"prize common winner message {index}",
            }
        )
    return pd.DataFrame(rows)


class PreprocessTests(unittest.TestCase):
    """Test the core phase-two and phase-three preprocessing behavior."""

    def test_clean_sms_data_removes_exact_duplicates_and_adds_target(self) -> None:
        raw_data = pd.concat(
            [build_sample_raw_data(), build_sample_raw_data().iloc[[0]]],
            ignore_index=True,
        )

        cleaned = clean_sms_data(raw_data)

        self.assertEqual(len(cleaned), 20)
        self.assertIn(TARGET_COLUMN, cleaned.columns)
        self.assertEqual(set(cleaned[TARGET_COLUMN]), {0, 1})

    def test_label_conflicts_are_detected(self) -> None:
        data = pd.DataFrame(
            [
                {LABEL_COLUMN: "ham", MESSAGE_COLUMN: "same text"},
                {LABEL_COLUMN: "spam", MESSAGE_COLUMN: "same text"},
            ]
        )

        conflicts = get_label_conflicts(normalize_raw_columns(data))

        self.assertEqual(conflicts[MESSAGE_COLUMN].nunique(), 1)

    def test_split_is_stratified_and_has_no_overlap(self) -> None:
        cleaned = clean_sms_data(build_sample_raw_data())

        splits = split_train_validation_test(cleaned)
        overlaps = find_split_overlaps(splits)

        self.assertEqual(len(splits["train"]), 12)
        self.assertEqual(len(splits["validation"]), 4)
        self.assertEqual(len(splits["test"]), 4)
        self.assertEqual(overlaps, {"train_validation": 0, "train_test": 0, "validation_test": 0})

        for split_data in splits.values():
            self.assertEqual(set(split_data[LABEL_COLUMN]), {"ham", "spam"})

    def test_tfidf_fits_train_only_and_transforms_all_splits(self) -> None:
        cleaned = clean_sms_data(build_sample_raw_data())
        splits = split_train_validation_test(cleaned)

        vectorizer = fit_tfidf_vectorizer(splits["train"])
        transformed = transform_splits(vectorizer, splits)

        self.assertGreater(len(vectorizer.vocabulary_), 0)
        self.assertEqual(transformed["train"].shape[0], len(splits["train"]))
        self.assertEqual(transformed["validation"].shape[0], len(splits["validation"]))
        self.assertEqual(transformed["test"].shape[0], len(splits["test"]))
        self.assertEqual(
            transformed["train"].shape[1],
            transformed["validation"].shape[1],
        )
        self.assertEqual(transformed["train"].shape[1], transformed["test"].shape[1])


if __name__ == "__main__":
    unittest.main()
