"""Evaluation utilities for SMS spam classification models."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
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


def plot_confusion_matrix(
    metrics: dict[str, Any],
    output_path: str | Path,
    title: str,
) -> Path:
    """Plot a confusion matrix from a metrics dictionary."""
    output = Path(output_path)
    output.parent.mkdir(parents=True, exist_ok=True)
    labels = metrics["confusion_matrix"]["labels"]
    matrix = np.asarray(metrics["confusion_matrix"]["matrix"])

    fig, ax = plt.subplots(figsize=(5, 4))
    image = ax.imshow(matrix, cmap="Blues")
    ax.set_title(title)
    ax.set_xlabel("Predicted label")
    ax.set_ylabel("True label")
    ax.set_xticks(np.arange(len(labels)), labels=labels)
    ax.set_yticks(np.arange(len(labels)), labels=labels)

    threshold = matrix.max() / 2 if matrix.max() else 0
    for row_index in range(matrix.shape[0]):
        for column_index in range(matrix.shape[1]):
            color = "white" if matrix[row_index, column_index] > threshold else "black"
            ax.text(
                column_index,
                row_index,
                str(int(matrix[row_index, column_index])),
                ha="center",
                va="center",
                color=color,
            )

    fig.colorbar(image, ax=ax, fraction=0.046, pad=0.04)
    fig.tight_layout()
    fig.savefig(output, dpi=150)
    plt.close(fig)
    return output


def plot_model_comparison(
    comparison_frame: pd.DataFrame,
    output_path: str | Path,
    title: str = "Test Set Model Performance Comparison",
) -> Path:
    """Plot required test-set metrics for all compared models."""
    metric_columns = [
        "accuracy",
        "macro_f1",
        "spam_precision",
        "spam_recall",
        "spam_f1",
    ]
    required_columns = ["model_name", *metric_columns]
    missing_columns = [
        column for column in required_columns if column not in comparison_frame.columns
    ]
    if missing_columns:
        raise ValueError(
            "Model comparison data is missing required columns: "
            + ", ".join(missing_columns)
        )

    model_name_map = {
        "majority_baseline": "Majority\nBaseline",
        "logistic_regression": "Logistic\nRegression",
        "mlp": "MLP",
    }
    metric_label_map = {
        "accuracy": "Accuracy",
        "macro_f1": "Macro-F1",
        "spam_precision": "Spam Precision",
        "spam_recall": "Spam Recall",
        "spam_f1": "Spam F1",
    }

    frame = comparison_frame.copy()
    model_labels = [
        model_name_map.get(str(model_name), str(model_name).replace("_", "\n"))
        for model_name in frame["model_name"]
    ]

    x_positions = np.arange(len(frame))
    bar_width = 0.14
    colors = ["#4C78A8", "#F58518", "#54A24B", "#E45756", "#72B7B2"]

    output = Path(output_path)
    output.parent.mkdir(parents=True, exist_ok=True)

    fig, ax = plt.subplots(figsize=(9, 5))
    for metric_index, metric_column in enumerate(metric_columns):
        offsets = (metric_index - (len(metric_columns) - 1) / 2) * bar_width
        values = frame[metric_column].astype(float).to_numpy()
        bars = ax.bar(
            x_positions + offsets,
            values,
            bar_width,
            label=metric_label_map[metric_column],
            color=colors[metric_index],
        )
        ax.bar_label(bars, fmt="%.3f", padding=2, fontsize=7, rotation=90)

    ax.set_title(title)
    ax.set_ylabel("Score")
    ax.set_xticks(x_positions, model_labels)
    ax.set_ylim(0, 1.15)
    ax.grid(axis="y", linestyle="--", alpha=0.35)
    ax.legend(ncol=3, fontsize=8, loc="upper center", bbox_to_anchor=(0.5, -0.12))
    fig.tight_layout()
    fig.savefig(output, dpi=150)
    plt.close(fig)
    return output
