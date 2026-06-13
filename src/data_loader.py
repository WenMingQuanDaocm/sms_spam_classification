"""Data loading utilities for the raw SMS Spam Collection file."""

from __future__ import annotations

from pathlib import Path

import pandas as pd

from src.config import LABEL_COLUMN, MESSAGE_COLUMN, RAW_DATA_PATH


def validate_raw_data_path(path: str | Path = RAW_DATA_PATH) -> Path:
    """Validate that the raw SMS dataset path exists and is a non-empty file."""
    raw_path = Path(path)
    if not raw_path.exists():
        raise FileNotFoundError(
            f"Raw dataset file not found: {raw_path}. "
            "Place the UCI file at data/raw/SMSSpamCollection."
        )
    if not raw_path.is_file():
        raise ValueError(f"Raw dataset path is not a file: {raw_path}")
    if raw_path.stat().st_size == 0:
        raise ValueError(f"Raw dataset file is empty: {raw_path}")
    return raw_path


def load_raw_sms_data(
    path: str | Path = RAW_DATA_PATH,
    encoding: str = "utf-8",
) -> pd.DataFrame:
    """Load the raw label-tab-message file into a DataFrame.

    The parser splits each line on the first tab only, preserving tabs that may
    appear inside the message text.
    """
    raw_path = validate_raw_data_path(path)
    records: list[dict[str, str]] = []
    malformed_lines: list[int] = []

    with raw_path.open("r", encoding=encoding, errors="strict", newline="") as handle:
        for line_number, line in enumerate(handle, start=1):
            line = line.rstrip("\r\n")
            if "\t" not in line:
                malformed_lines.append(line_number)
                continue
            label, message = line.split("\t", 1)
            records.append({LABEL_COLUMN: label, MESSAGE_COLUMN: message})

    if malformed_lines:
        preview = ", ".join(str(line) for line in malformed_lines[:10])
        raise ValueError(
            "Malformed raw data rows without a tab separator at line(s): "
            f"{preview}"
        )
    if not records:
        raise ValueError(f"No rows were loaded from raw dataset file: {raw_path}")

    return pd.DataFrame.from_records(
        records,
        columns=[LABEL_COLUMN, MESSAGE_COLUMN],
    )


def validate_required_columns(
    data: pd.DataFrame,
    required_columns: tuple[str, ...] = (LABEL_COLUMN, MESSAGE_COLUMN),
) -> None:
    """Validate that a DataFrame contains the columns required for this project."""
    missing_columns = [column for column in required_columns if column not in data.columns]
    if missing_columns:
        raise ValueError(f"Missing required column(s): {missing_columns}")
