"""短信垃圾分类项目的探索性数据分析工具。"""

from __future__ import annotations

import json
import os
from pathlib import Path
from typing import Any

from src.config import (
    EDA_FIGURES_DIR,
    LABEL_COLUMN,
    MATPLOTLIB_CACHE_DIR,
    MESSAGE_COLUMN,
    METRICS_DIR,
    VALID_LABELS,
)

os.environ.setdefault("MPLCONFIGDIR", str(MATPLOTLIB_CACHE_DIR))

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import pandas as pd

from src.plotting import configure_plot_style, display_class_label
from src.preprocess import build_quality_report


configure_plot_style()

TEXT_STAT_COLUMNS = (
    "char_count",
    "word_count",
    "digit_count",
    "exclamation_count",
    "uppercase_count",
)


def add_text_statistics(data: pd.DataFrame) -> pd.DataFrame:
    """添加仅用于 EDA 的文本统计特征，不改变模型输入文本。"""
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
    """根据给定数据构建可保存为 JSON 的 EDA 统计摘要。"""
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
    """将字典保存为 UTF-8 编码的 JSON 文件。"""
    output_path = Path(path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(
        json.dumps(data, indent=2, ensure_ascii=False),
        encoding="utf-8",
    )
    return output_path


def plot_class_distribution(data: pd.DataFrame, output_path: str | Path) -> Path:
    """绘制正常短信和垃圾短信的类别数量图。"""
    output = Path(output_path)
    output.parent.mkdir(parents=True, exist_ok=True)

    counts = data[LABEL_COLUMN].value_counts().reindex(VALID_LABELS, fill_value=0)
    fig, ax = plt.subplots(figsize=(6, 4))
    x_labels = [display_class_label(label) for label in counts.index]
    ax.bar(x_labels, counts.values, color=["#4C78A8", "#F58518"])
    ax.set_xlabel("类别")
    ax.set_ylabel("样本数量")
    ax.set_ylim(0, max(counts.values) * 1.12 if len(counts.values) else 1)
    total_count = int(counts.sum())
    for index, value in enumerate(counts.values):
        # 同时显示数量和比例，让类别不均衡在一张图里就能看清楚。
        proportion = value / total_count if total_count else 0.0
        ax.text(
            index,
            value,
            f"{int(value)}（{proportion:.2%}）",
            ha="center",
            va="bottom",
        )
    fig.tight_layout()
    fig.savefig(output, dpi=150)
    plt.close(fig)
    return output


def plot_distribution_by_label(
    data: pd.DataFrame,
    column: str,
    x_label: str,
    output_path: str | Path,
    density: bool = False,
    x_limits: tuple[float, float] | None = None,
    y_label: str | None = None,
) -> Path:
    """按类别绘制某个 EDA 统计量的直方图。"""
    output = Path(output_path)
    output.parent.mkdir(parents=True, exist_ok=True)

    fig, ax = plt.subplots(figsize=(7, 4))
    for label in VALID_LABELS:
        values = data.loc[data[LABEL_COLUMN] == label, column]
        # density=True 比较的是分布形状，而不是两类样本的绝对数量。
        ax.hist(
            values,
            bins=30,
            density=density,
            range=x_limits,
            alpha=0.55,
            label=display_class_label(label),
        )
    if x_limits is not None:
        ax.set_xlim(*x_limits)
    ax.set_xlabel(x_label)
    ax.set_ylabel(y_label or ("密度" if density else "频数"))
    ax.legend()
    fig.tight_layout()
    fig.savefig(output, dpi=150)
    plt.close(fig)
    return output


def plot_digit_and_exclamation_distributions(
    data: pd.DataFrame,
    output_path: str | Path,
) -> Path:
    """按类别绘制数字数量和感叹号数量分布图。"""
    output = Path(output_path)
    output.parent.mkdir(parents=True, exist_ok=True)

    fig, axes = plt.subplots(1, 2, figsize=(10, 4))
    for label in VALID_LABELS:
        label_data = data[data[LABEL_COLUMN] == label]
        display_label = display_class_label(label)
        axes[0].hist(label_data["digit_count"], bins=20, alpha=0.55, label=display_label)
        axes[1].hist(
            label_data["exclamation_count"],
            bins=20,
            alpha=0.55,
            label=display_label,
        )
    axes[0].set_xlabel("数字数量")
    axes[0].set_ylabel("频数")
    axes[1].set_xlabel("感叹号数量")
    axes[1].set_ylabel("频数")
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
    """生成第二阶段所需的 EDA 摘要和图像。"""
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
            "短信字符数",
            figures / "message_char_count_distribution.png",
            density=True,
            x_limits=(0, 400),
            y_label="密度",
        ),
        "word_count_distribution": plot_distribution_by_label(
            enriched,
            "word_count",
            "短信词数",
            figures / "message_word_count_distribution.png",
        ),
        "digit_exclamation_distribution": plot_digit_and_exclamation_distributions(
            enriched,
            figures / "digit_exclamation_distribution.png",
        ),
    }
    return outputs
