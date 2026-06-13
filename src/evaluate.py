"""Evaluation utilities for SMS spam classification models."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
from sklearn.metrics import (
    accuracy_score,
    confusion_matrix,
    f1_score,
    precision_recall_fscore_support,
)

from src.config import LABEL_TO_TARGET, TARGET_TO_LABEL


def evaluate_predictions(
    y_true: pd.Series | np.ndarray,
    y_pred: pd.Series | np.ndarray,
    model_name: str,
    split_name: str,
) -> dict[str, Any]:
    """Compute project-required classification metrics."""
    y_true_array = np.asarray(y_true)
    y_pred_array = np.asarray(y_pred)
    label_ids = [LABEL_TO_TARGET["ham"], LABEL_TO_TARGET["spam"]]

    precision, recall, f1, support = precision_recall_fscore_support(
        y_true_array,
        y_pred_array,
        labels=label_ids,
        zero_division=0,
    )
    matrix = confusion_matrix(y_true_array, y_pred_array, labels=label_ids)

    per_class = {
        TARGET_TO_LABEL[label_id]: {
            "precision": float(precision[index]),
            "recall": float(recall[index]),
            "f1": float(f1[index]),
            "support": int(support[index]),
        }
        for index, label_id in enumerate(label_ids)
    }

    return {
        "model_name": model_name,
        "split_name": split_name,
        "accuracy": float(accuracy_score(y_true_array, y_pred_array)),
        "macro_f1": float(f1_score(y_true_array, y_pred_array, average="macro")),
        "spam_precision": per_class["spam"]["precision"],
        "spam_recall": per_class["spam"]["recall"],
        "spam_f1": per_class["spam"]["f1"],
        "per_class": per_class,
        "confusion_matrix": {
            "labels": ["ham", "spam"],
            "matrix": matrix.astype(int).tolist(),
        },
    }


def save_metrics(metrics: dict[str, Any], path: str | Path) -> Path:
    """Save metrics as UTF-8 JSON."""
    output_path = Path(path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(
        json.dumps(metrics, indent=2, ensure_ascii=False),
        encoding="utf-8",
    )
    return output_path
