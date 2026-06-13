"""Data quality checks and conservative preprocessing for SMS messages."""

from __future__ import annotations

from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any

import pandas as pd

from src.config import (
    LABEL_COLUMN,
    LABEL_TO_TARGET,
    MESSAGE_COLUMN,
    TARGET_COLUMN,
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
