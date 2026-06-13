"""Exploratory data analysis helpers for SMS spam classification."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import pandas as pd

from src.config import EDA_FIGURES_DIR, LABEL_COLUMN, MESSAGE_COLUMN, METRICS_DIR, VALID_LABELS
from src.preprocess import build_quality_report


TEXT_STAT_COLUMNS = (
    "char_count",
    "word_count",
    "digit_count",
    "exclamation_count",
    "uppercase_count",
)


def add_text_statistics(data: pd.DataFrame) -> pd.DataFrame:
    """Add EDA-only text statistics without changing model input text."""
    enriched = data.copy()
    messages = enriched[MESSAGE_COLUMN].fillna("").astype(str)

    enriched["char_count"] = messages.str.len()
    enriched["word_count"] = messages.str.split().map(len)
    enriched["digit_count"] = messages.str.count(r"\d")
    enriched["exclamation_count"] = messages.str.count("!")
    enriched["uppercase_count"] = messages.map(
        lambda message: sum(1 for character in message if character.isupper())
    )
    return enriched


def build_eda_summary(data: pd.DataFrame) -> dict[str, Any]:
    """Build JSON-serializable EDA statistics from the provided data."""
    enriched = add_text_statistics(data)
    quality_report = build_quality_report(data).to_dict()

    class_counts = enriched[LABEL_COLUMN].value_counts().reindex(VALID_LABELS, fill_value=0)
    class_distribution = {
        label: {
            "count": int(count),
            "proportion": float(count / len(enriched)) if len(enriched) else 0.0,
        }
        for label, count in class_counts.items()
    }

    grouped_stats = (
        enriched.groupby(LABEL_COLUMN)[list(TEXT_STAT_COLUMNS)]
        .agg(["mean", "median", "min", "max"])
        .round(3)
    )
    grouped_stats.columns = [
        f"{column}_{statistic}" for column, statistic in grouped_stats.columns
    ]
    text_statistics_by_label = {
        str(label): {
            str(metric): float(value)
            for metric, value in row.dropna().items()
        }
        for label, row in grouped_stats.iterrows()
    }

    return {
        "shape": {"rows": int(enriched.shape[0]), "columns": int(enriched.shape[1])},
        "columns": list(enriched.columns),
        "dtypes": {column: str(dtype) for column, dtype in enriched.dtypes.items()},
        "quality_report": quality_report,
        "class_distribution": class_distribution,
        "text_statistics_by_label": text_statistics_by_label,
        "preview": enriched.head(5).to_dict(orient="records"),
    }


def save_json(data: dict[str, Any], path: str | Path) -> Path:
    """Save a dictionary as UTF-8 JSON."""
    output_path = Path(path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(
        json.dumps(data, indent=2, ensure_ascii=False),
        encoding="utf-8",
    )
    return output_path


def plot_class_distribution(data: pd.DataFrame, output_path: str | Path) -> Path:
    """Plot ham/spam class counts."""
    output = Path(output_path)
    output.parent.mkdir(parents=True, exist_ok=True)

    counts = data[LABEL_COLUMN].value_counts().reindex(VALID_LABELS, fill_value=0)
    fig, ax = plt.subplots(figsize=(6, 4))
    ax.bar(counts.index, counts.values, color=["#4C78A8", "#F58518"])
    ax.set_title("Class Distribution")
    ax.set_xlabel("Label")
    ax.set_ylabel("Message Count")
    for index, value in enumerate(counts.values):
        ax.text(index, value, str(int(value)), ha="center", va="bottom")
    fig.tight_layout()
    fig.savefig(output, dpi=150)
    plt.close(fig)
    return output


def plot_distribution_by_label(
    data: pd.DataFrame,
    column: str,
    title: str,
    output_path: str | Path,
) -> Path:
    """Plot a histogram for an EDA statistic grouped by label."""
    output = Path(output_path)
    output.parent.mkdir(parents=True, exist_ok=True)

    fig, ax = plt.subplots(figsize=(7, 4))
    for label in VALID_LABELS:
        values = data.loc[data[LABEL_COLUMN] == label, column]
        ax.hist(values, bins=30, alpha=0.55, label=label)
    ax.set_title(title)
    ax.set_xlabel(column)
    ax.set_ylabel("Message Count")
    ax.legend()
    fig.tight_layout()
    fig.savefig(output, dpi=150)
    plt.close(fig)
    return output


def plot_digit_and_exclamation_distributions(
    data: pd.DataFrame,
    output_path: str | Path,
) -> Path:
    """Plot digit and exclamation-count distributions by label."""
    output = Path(output_path)
    output.parent.mkdir(parents=True, exist_ok=True)

    fig, axes = plt.subplots(1, 2, figsize=(10, 4))
    for label in VALID_LABELS:
        label_data = data[data[LABEL_COLUMN] == label]
        axes[0].hist(label_data["digit_count"], bins=20, alpha=0.55, label=label)
        axes[1].hist(label_data["exclamation_count"], bins=20, alpha=0.55, label=label)
    axes[0].set_title("Digit Count")
    axes[0].set_xlabel("digit_count")
    axes[0].set_ylabel("Message Count")
    axes[1].set_title("Exclamation Count")
    axes[1].set_xlabel("exclamation_count")
    for axis in axes:
        axis.legend()
    fig.tight_layout()
    fig.savefig(output, dpi=150)
    plt.close(fig)
    return output


def generate_eda_outputs(
    data: pd.DataFrame,
    figures_dir: str | Path = EDA_FIGURES_DIR,
    summary_path: str | Path = METRICS_DIR / "eda_summary.json",
) -> dict[str, Path]:
    """Generate the required phase-two EDA summary and figures."""
    figures = Path(figures_dir)
    enriched = add_text_statistics(data)

    outputs = {
        "summary": save_json(build_eda_summary(data), summary_path),
        "class_distribution": plot_class_distribution(
            enriched,
            figures / "class_distribution.png",
        ),
        "char_count_distribution": plot_distribution_by_label(
            enriched,
            "char_count",
            "Message Length Distribution",
            figures / "message_char_count_distribution.png",
        ),
        "word_count_distribution": plot_distribution_by_label(
            enriched,
            "word_count",
            "Word Count Distribution",
            figures / "message_word_count_distribution.png",
        ),
        "digit_exclamation_distribution": plot_digit_and_exclamation_distributions(
            enriched,
            figures / "digit_exclamation_distribution.png",
        ),
    }
    return outputs
