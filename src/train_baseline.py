"""Train and validate the majority-class baseline."""

from __future__ import annotations

import time
from pathlib import Path
from typing import Any

import pandas as pd
from sklearn.dummy import DummyClassifier

from src.config import (
    BASELINE_METRICS_PATH,
    MESSAGE_COLUMN,
    TARGET_COLUMN,
    TRAIN_DATA_PATH,
    VALIDATION_DATA_PATH,
)
from src.evaluate import evaluate_predictions, save_metrics


def load_train_validation_splits(
    train_path: str | Path = TRAIN_DATA_PATH,
    validation_path: str | Path = VALIDATION_DATA_PATH,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Load the fixed train and validation splits."""
    train_data = pd.read_csv(train_path, encoding="utf-8")
    validation_data = pd.read_csv(validation_path, encoding="utf-8")
    return train_data, validation_data


def train_majority_baseline(train_data: pd.DataFrame) -> DummyClassifier:
    """Train a majority-class baseline on the training split only."""
    model = DummyClassifier(strategy="most_frequent")
    model.fit(train_data[[MESSAGE_COLUMN]], train_data[TARGET_COLUMN])
    return model


def run_baseline_validation(
    train_path: str | Path = TRAIN_DATA_PATH,
    validation_path: str | Path = VALIDATION_DATA_PATH,
    metrics_path: str | Path = BASELINE_METRICS_PATH,
) -> dict[str, Any]:
    """Train the baseline and evaluate it on the validation split."""
    train_data, validation_data = load_train_validation_splits(
        train_path,
        validation_path,
    )
    start_time = time.perf_counter()
    model = train_majority_baseline(train_data)
    training_time_seconds = time.perf_counter() - start_time

    predictions = model.predict(validation_data[[MESSAGE_COLUMN]])
    metrics = evaluate_predictions(
        validation_data[TARGET_COLUMN],
        predictions,
        model_name="majority_baseline",
        split_name="validation",
    )
    metrics["training_time_seconds"] = float(training_time_seconds)
    metrics["model_config"] = {"strategy": "most_frequent"}
    save_metrics(metrics, metrics_path)
    return metrics


if __name__ == "__main__":
    import json

    print(json.dumps(run_baseline_validation(), indent=2, ensure_ascii=False))
