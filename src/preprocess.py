"""Data quality checks and conservative preprocessing for SMS messages."""

from __future__ import annotations

from dataclasses import asdict, dataclass
import json
from pathlib import Path
from typing import Any

import joblib
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.model_selection import train_test_split

from src.config import (
    LABEL_COLUMN,
    LABEL_TO_TARGET,
    MESSAGE_COLUMN,
    PREPROCESSING_SUMMARY_PATH,
    TARGET_COLUMN,
    TEST_DATA_PATH,
    TEST_SIZE,
    TFIDF_CONFIG,
    TFIDF_VECTORIZER_PATH,
    TRAIN_DATA_PATH,
    TRAIN_SIZE,
    VALIDATION_DATA_PATH,
    VALIDATION_SIZE,
    RANDOM_STATE,
    VALID_LABELS,
)
from src.data_loader import validate_required_columns


@dataclass(frozen=True)
class DataQualityReport:
    """Summary of raw data quality checks."""

    total_rows: int
    missing_value_counts: dict[str, int]
    empty_message_count: int
    invalid_label_count: int
    exact_duplicate_row_count: int
    duplicate_rows_to_remove: int
    duplicate_message_count: int
    label_conflict_message_count: int
    class_counts: dict[str, int]

    def to_dict(self) -> dict[str, Any]:
        """Return a JSON-serializable report dictionary."""
        return asdict(self)


@dataclass(frozen=True)
class SplitReport:
    """Summary of the fixed train/validation/test split."""

    split_sizes: dict[str, int]
    class_distribution: dict[str, dict[str, dict[str, float | int]]]
    overlap_counts: dict[str, int]

    def to_dict(self) -> dict[str, Any]:
        """Return a JSON-serializable split report dictionary."""
        return asdict(self)


def normalize_raw_columns(data: pd.DataFrame) -> pd.DataFrame:
    """Trim labels and messages while preserving message content."""
    validate_required_columns(data)
    normalized = data.copy()
    normalized[LABEL_COLUMN] = (
        normalized[LABEL_COLUMN].astype("string").str.strip().str.lower()
    )
    normalized[MESSAGE_COLUMN] = normalized[MESSAGE_COLUMN].astype("string").str.strip()
    return normalized


def get_empty_message_mask(data: pd.DataFrame) -> pd.Series:
    """Return a mask for rows with empty message text after trimming."""
    validate_required_columns(data)
    messages = data[MESSAGE_COLUMN].astype("string")
    return messages.notna() & messages.str.strip().eq("")


def get_invalid_label_mask(data: pd.DataFrame) -> pd.Series:
    """Return a mask for rows with labels outside the allowed label set."""
    validate_required_columns(data)
    labels = data[LABEL_COLUMN].astype("string")
    return labels.notna() & ~labels.isin(VALID_LABELS)


def get_label_conflicts(data: pd.DataFrame) -> pd.DataFrame:
    """Return rows whose message text appears with more than one label."""
    validate_required_columns(data)
    comparable = data.dropna(subset=[LABEL_COLUMN, MESSAGE_COLUMN])
    label_counts = comparable.groupby(MESSAGE_COLUMN)[LABEL_COLUMN].nunique()
    conflict_messages = label_counts[label_counts > 1].index
    if len(conflict_messages) == 0:
        return data.iloc[0:0].copy()
    return (
        data[data[MESSAGE_COLUMN].isin(conflict_messages)]
        .sort_values([MESSAGE_COLUMN, LABEL_COLUMN])
        .reset_index(drop=True)
    )


def build_quality_report(data: pd.DataFrame) -> DataQualityReport:
    """Build the required data quality report without modifying the input."""
    normalized = normalize_raw_columns(data)
    missing_counts = {
        column: int(normalized[column].isna().sum())
        for column in (LABEL_COLUMN, MESSAGE_COLUMN)
    }

    labels_for_counts = normalized[LABEL_COLUMN].astype("object")
    labels_for_counts = labels_for_counts.where(normalized[LABEL_COLUMN].notna(), "<missing>")
    class_counts = {
        str(label): int(count)
        for label, count in labels_for_counts.value_counts(dropna=False).sort_index().items()
    }

    duplicate_messages = normalized.duplicated(subset=[MESSAGE_COLUMN], keep=False)
    conflicts = get_label_conflicts(normalized)

    return DataQualityReport(
        total_rows=int(len(normalized)),
        missing_value_counts=missing_counts,
        empty_message_count=int(get_empty_message_mask(normalized).sum()),
        invalid_label_count=int(get_invalid_label_mask(normalized).sum()),
        exact_duplicate_row_count=int(
            normalized.duplicated(subset=[LABEL_COLUMN, MESSAGE_COLUMN], keep=False).sum()
        ),
        duplicate_rows_to_remove=int(
            normalized.duplicated(subset=[LABEL_COLUMN, MESSAGE_COLUMN], keep="first").sum()
        ),
        duplicate_message_count=int(duplicate_messages.sum()),
        label_conflict_message_count=int(conflicts[MESSAGE_COLUMN].nunique()),
        class_counts=class_counts,
    )


def clean_sms_data(data: pd.DataFrame) -> pd.DataFrame:
    """Clean raw SMS data for later splitting.

    This function only performs approved preprocessing: trim whitespace, remove
    empty messages, remove exact duplicates, validate labels, and add the
    numeric target column. Label conflicts fail fast instead of being resolved
    silently.
    """
    normalized = normalize_raw_columns(data)
    report = build_quality_report(normalized)

    missing_total = sum(report.missing_value_counts.values())
    if missing_total > 0:
        raise ValueError(f"Missing values found: {report.missing_value_counts}")
    if report.invalid_label_count > 0:
        invalid_labels = (
            normalized.loc[get_invalid_label_mask(normalized), LABEL_COLUMN]
            .dropna()
            .unique()
            .tolist()
        )
        raise ValueError(f"Invalid label(s) found: {invalid_labels}")

    non_empty = normalized.loc[~get_empty_message_mask(normalized)].copy()
    conflicts = get_label_conflicts(non_empty)
    if not conflicts.empty:
        raise ValueError(
            "Label conflicts found for identical message text. "
            "Review conflicts before continuing."
        )

    cleaned = (
        non_empty.drop_duplicates(subset=[LABEL_COLUMN, MESSAGE_COLUMN], keep="first")
        .reset_index(drop=True)
        .copy()
    )
    cleaned[TARGET_COLUMN] = cleaned[LABEL_COLUMN].map(LABEL_TO_TARGET).astype(int)
    return cleaned


def save_cleaned_preview(data: pd.DataFrame, path: str | Path, rows: int = 20) -> Path:
    """Save a small preview CSV for manual review, not a processed split."""
    output_path = Path(path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    data.head(rows).to_csv(output_path, index=False, encoding="utf-8")
    return output_path


def validate_split_input(data: pd.DataFrame) -> None:
    """Validate cleaned data before train/validation/test splitting."""
    validate_required_columns(data, (LABEL_COLUMN, MESSAGE_COLUMN, TARGET_COLUMN))
    expected_total = TRAIN_SIZE + VALIDATION_SIZE + TEST_SIZE
    if abs(expected_total - 1.0) > 1e-9:
        raise ValueError(
            "TRAIN_SIZE + VALIDATION_SIZE + TEST_SIZE must equal 1.0, "
            f"got {expected_total}."
        )
    if data[TARGET_COLUMN].nunique() < 2:
        raise ValueError("Stratified splitting requires at least two classes.")


def split_train_validation_test(data: pd.DataFrame) -> dict[str, pd.DataFrame]:
    """Create fixed 60/20/20 stratified train, validation, and test splits."""
    validate_split_input(data)

    train_data, temp_data = train_test_split(
        data,
        train_size=TRAIN_SIZE,
        stratify=data[TARGET_COLUMN],
        random_state=RANDOM_STATE,
    )
    relative_test_size = TEST_SIZE / (VALIDATION_SIZE + TEST_SIZE)
    validation_data, test_data = train_test_split(
        temp_data,
        test_size=relative_test_size,
        stratify=temp_data[TARGET_COLUMN],
        random_state=RANDOM_STATE,
    )

    return {
        "train": train_data.reset_index(drop=True),
        "validation": validation_data.reset_index(drop=True),
        "test": test_data.reset_index(drop=True),
    }


def save_splits(
    splits: dict[str, pd.DataFrame],
    train_path: str | Path = TRAIN_DATA_PATH,
    validation_path: str | Path = VALIDATION_DATA_PATH,
    test_path: str | Path = TEST_DATA_PATH,
) -> dict[str, Path]:
    """Save train, validation, and test splits as UTF-8 CSV files."""
    output_paths = {
        "train": Path(train_path),
        "validation": Path(validation_path),
        "test": Path(test_path),
    }
    for split_name, output_path in output_paths.items():
        output_path.parent.mkdir(parents=True, exist_ok=True)
        splits[split_name].to_csv(output_path, index=False, encoding="utf-8")
    return output_paths


def get_class_distribution(data: pd.DataFrame) -> dict[str, dict[str, float | int]]:
    """Return class counts and proportions for one split."""
    counts = data[LABEL_COLUMN].value_counts().reindex(VALID_LABELS, fill_value=0)
    total = len(data)
    return {
        label: {
            "count": int(count),
            "proportion": float(count / total) if total else 0.0,
        }
        for label, count in counts.items()
    }


def find_split_overlaps(splits: dict[str, pd.DataFrame]) -> dict[str, int]:
    """Count message text overlaps between split pairs."""
    message_sets = {
        split_name: set(split_data[MESSAGE_COLUMN].astype(str))
        for split_name, split_data in splits.items()
    }
    return {
        "train_validation": len(message_sets["train"] & message_sets["validation"]),
        "train_test": len(message_sets["train"] & message_sets["test"]),
        "validation_test": len(message_sets["validation"] & message_sets["test"]),
    }


def validate_no_split_overlap(splits: dict[str, pd.DataFrame]) -> None:
    """Fail if any message text appears in more than one split."""
    overlap_counts = find_split_overlaps(splits)
    if any(count > 0 for count in overlap_counts.values()):
        raise ValueError(f"Split overlap detected: {overlap_counts}")


def build_split_report(splits: dict[str, pd.DataFrame]) -> SplitReport:
    """Build a report for split sizes, class balance, and overlaps."""
    return SplitReport(
        split_sizes={name: int(len(split_data)) for name, split_data in splits.items()},
        class_distribution={
            name: get_class_distribution(split_data)
            for name, split_data in splits.items()
        },
        overlap_counts=find_split_overlaps(splits),
    )


def fit_tfidf_vectorizer(train_data: pd.DataFrame) -> TfidfVectorizer:
    """Fit TF-IDF on the training split only."""
    validate_required_columns(train_data, (MESSAGE_COLUMN,))
    vectorizer = TfidfVectorizer(**TFIDF_CONFIG)
    vectorizer.fit(train_data[MESSAGE_COLUMN])
    return vectorizer


def transform_splits(
    vectorizer: TfidfVectorizer,
    splits: dict[str, pd.DataFrame],
) -> dict[str, Any]:
    """Transform train, validation, and test messages using a fitted vectorizer."""
    return {
        split_name: vectorizer.transform(split_data[MESSAGE_COLUMN])
        for split_name, split_data in splits.items()
    }


def save_tfidf_vectorizer(
    vectorizer: TfidfVectorizer,
    path: str | Path = TFIDF_VECTORIZER_PATH,
) -> Path:
    """Save the fitted TF-IDF vectorizer."""
    output_path = Path(path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    joblib.dump(vectorizer, output_path)
    return output_path


def build_preprocessing_summary(
    cleaned_data: pd.DataFrame,
    splits: dict[str, pd.DataFrame],
    vectorizer: TfidfVectorizer,
    transformed_splits: dict[str, Any],
) -> dict[str, Any]:
    """Build a reproducibility summary for phase-three preprocessing."""
    split_report = build_split_report(splits)
    feature_shapes = {
        split_name: [int(matrix.shape[0]), int(matrix.shape[1])]
        for split_name, matrix in transformed_splits.items()
    }
    return {
        "cleaned_rows": int(len(cleaned_data)),
        "split_report": split_report.to_dict(),
        "tfidf_config": TFIDF_CONFIG,
        "tfidf_vocabulary_size": int(len(vectorizer.vocabulary_)),
        "feature_shapes": feature_shapes,
        "notes": [
            "TF-IDF vectorizer was fit on the training split only.",
            "Validation and test splits were transformed only.",
        ],
    }


def save_preprocessing_summary(
    summary: dict[str, Any],
    path: str | Path = PREPROCESSING_SUMMARY_PATH,
) -> Path:
    """Save the phase-three preprocessing summary as UTF-8 JSON."""
    output_path = Path(path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(
        json.dumps(summary, indent=2, ensure_ascii=False),
        encoding="utf-8",
    )
    return output_path
