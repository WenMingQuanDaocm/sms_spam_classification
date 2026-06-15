"""训练并验证多数类基线模型。"""

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
    """加载固定训练集和验证集切分。"""
    train_data = pd.read_csv(train_path, encoding="utf-8")
    validation_data = pd.read_csv(validation_path, encoding="utf-8")
    return train_data, validation_data


def train_majority_baseline(train_data: pd.DataFrame) -> DummyClassifier:
    """只在训练集上训练多数类基线模型。"""
    # 多数类基线不看短信内容，用来衡量类别不均衡带来的“虚高准确率”。
    model = DummyClassifier(strategy="most_frequent")
    model.fit(train_data[[MESSAGE_COLUMN]], train_data[TARGET_COLUMN])
    return model


def run_baseline_validation(
    train_path: str | Path = TRAIN_DATA_PATH,
    validation_path: str | Path = VALIDATION_DATA_PATH,
    metrics_path: str | Path = BASELINE_METRICS_PATH,
) -> dict[str, Any]:
    """训练基线模型并在验证集上评估。"""
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
