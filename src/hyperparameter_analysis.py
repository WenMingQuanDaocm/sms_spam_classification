"""受控 MLP 超参数敏感性实验。"""

from __future__ import annotations

import json
import math
import os
from pathlib import Path
from typing import Any

import numpy as np

from src.config import MATPLOTLIB_CACHE_DIR

os.environ.setdefault("MPLCONFIGDIR", str(MATPLOTLIB_CACHE_DIR))

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import pandas as pd

from src.config import (
    DROPOUT_EXPERIMENTS_PATH,
    HYPERPARAMETER_FIGURES_DIR,
    LEARNING_RATE_EXPERIMENTS_PATH,
    METRICS_DIR,
    MLP_CONFIG,
    MLP_MODEL_DIR,
)
from src.plotting import configure_plot_style
from src.train_mlp import train_mlp_validation


configure_plot_style()

LEARNING_RATE_VALUES = (0.01, 0.001, 0.0001)
DROPOUT_VALUES = (0.0, 0.3, 0.5)


def format_float_for_path(value: float) -> str:
    """返回适合文件路径使用的浮点数表示。"""
    return str(value).replace(".", "_")


def build_experiment_config(**overrides: Any) -> dict[str, Any]:
    """返回应用受控覆盖项后的默认 MLP 配置。"""
    config = dict(MLP_CONFIG)
    config.update(overrides)
    return config


def load_history(path: str | Path) -> pd.DataFrame:
    """加载单次 MLP 训练历史 CSV。"""
    return pd.read_csv(path, encoding="utf-8")


def has_overfitting_signal(history: pd.DataFrame) -> bool:
    """判断最终训练曲线和验证曲线是否显示明显过拟合迹象。"""
    final_row = history.iloc[-1]
    best_row = history.loc[history["validation_macro_f1"].idxmax()]
    # 这个判断只用于描述训练现象，不参与最终模型选择。
    accuracy_gap = final_row["train_accuracy"] - final_row["validation_accuracy"]
    validation_loss_increased = final_row["validation_loss"] > best_row["validation_loss"]
    return bool(accuracy_gap > 0.03 and validation_loss_increased)


def has_underfitting_signal(history: pd.DataFrame) -> bool:
    """判断本次运行是否存在训练数据拟合不足迹象。"""
    best_row = history.loc[history["validation_macro_f1"].idxmax()]
    return bool(best_row["train_accuracy"] < 0.95 or best_row["validation_macro_f1"] < 0.85)


def training_was_stable(history: pd.DataFrame) -> bool:
    """判断损失是否有限且没有明显发散。"""
    loss_columns = ["train_loss", "validation_loss"]
    finite_losses = history[loss_columns].map(math.isfinite).all().all()
    first_validation_loss = float(history.iloc[0]["validation_loss"])
    max_validation_loss = float(history["validation_loss"].max())
    no_large_loss_spike = max_validation_loss <= max(1.0, first_validation_loss * 10.0)
    return bool(finite_losses and no_large_loss_spike)


def summarize_experiment(
    metrics: dict[str, Any],
    experiment_group: str,
    variable_name: str,
    variable_value: float,
) -> dict[str, Any]:
    """为一次受控实验构建一行 CSV 摘要。"""
    history = load_history(metrics["history_path"])
    best_row = history.loc[history["validation_macro_f1"].idxmax()]
    final_row = history.iloc[-1]
    config = metrics["model_config"]

    return {
        "experiment_group": experiment_group,
        "variable_name": variable_name,
        "variable_value": variable_value,
        "hidden_layers": str(tuple(config["hidden_layers"])),
        "dropout": float(config["dropout"]),
        "learning_rate": float(config["learning_rate"]),
        "weight_decay": float(config["weight_decay"]),
        "batch_size": int(config["batch_size"]),
        "best_epoch": int(metrics["best_epoch"]),
        "epochs_ran": int(metrics["epochs_ran"]),
        "validation_accuracy": float(metrics["accuracy"]),
        "validation_macro_f1": float(metrics["macro_f1"]),
        "spam_precision": float(metrics["spam_precision"]),
        "spam_recall": float(metrics["spam_recall"]),
        "spam_f1": float(metrics["spam_f1"]),
        "training_time_seconds": float(metrics["training_time_seconds"]),
        "best_train_accuracy": float(best_row["train_accuracy"]),
        "best_validation_loss": float(best_row["validation_loss"]),
        "final_train_accuracy": float(final_row["train_accuracy"]),
        "final_validation_accuracy": float(final_row["validation_accuracy"]),
        "training_stable": training_was_stable(history),
        "overfitting_signal": has_overfitting_signal(history),
        "underfitting_signal": has_underfitting_signal(history),
        "checkpoint_path": metrics["checkpoint_path"],
        "history_path": metrics["history_path"],
        "metrics_path": metrics["metrics_path"],
        "training_curves_path": metrics["training_curves_path"],
    }


def run_single_experiment(
    experiment_group: str,
    variable_name: str,
    variable_value: float,
    config: dict[str, Any],
) -> dict[str, Any]:
    """运行一次 MLP 实验，并写入独立输出文件。"""
    value_name = format_float_for_path(variable_value)
    run_name = f"{experiment_group}_{variable_name}_{value_name}"
    metrics_path = METRICS_DIR / "hyperparameter_runs" / f"{run_name}.json"

    # 每组实验写入独立文件，避免敏感性实验覆盖默认模型结果。
    metrics = train_mlp_validation(
        checkpoint_path=MLP_MODEL_DIR / "hyperparameters" / f"{run_name}.pt",
        history_path=METRICS_DIR / "hyperparameter_runs" / f"{run_name}_history.csv",
        metrics_path=metrics_path,
        training_curves_path=HYPERPARAMETER_FIGURES_DIR / f"{run_name}_curves.png",
        config=config,
    )
    metrics["metrics_path"] = str(metrics_path)
    Path(metrics_path).write_text(
        json.dumps(metrics, indent=2, ensure_ascii=False),
        encoding="utf-8",
    )
    return summarize_experiment(metrics, experiment_group, variable_name, variable_value)


def save_experiment_table(rows: list[dict[str, Any]], path: str | Path) -> Path:
    """将实验摘要表保存为 UTF-8 编码的 CSV 文件。"""
    output_path = Path(path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    pd.DataFrame(rows).to_csv(output_path, index=False, encoding="utf-8")
    return output_path


def plot_sensitivity(
    results: pd.DataFrame,
    x_column: str,
    output_path: str | Path,
    log_x: bool = False,
    y_limits: tuple[float, float] = (0.96, 0.99),
) -> Path:
    """绘制某个超参数下的验证集 Macro-F1 和 Accuracy。"""
    output = Path(output_path)
    output.parent.mkdir(parents=True, exist_ok=True)
    ordered = results.sort_values(x_column)

    fig, ax = plt.subplots(figsize=(7, 4))
    macro_f1_values = ordered["validation_macro_f1"].astype(float)
    accuracy_values = ordered["validation_accuracy"].astype(float)
    macro_line = ax.plot(
        ordered[x_column],
        macro_f1_values,
        marker="o",
        label="Macro-F1",
    )
    accuracy_line = ax.plot(
        ordered[x_column],
        accuracy_values,
        marker="o",
        label="Accuracy",
    )

    best_macro_f1 = float(macro_f1_values.max())
    # 并列最优都标出，例如 dropout=0.0 和 0.3 可能得到相同 Macro-F1。
    best_rows = ordered[np.isclose(macro_f1_values, best_macro_f1)]
    ax.scatter(
        best_rows[x_column],
        best_rows["validation_macro_f1"],
        marker="*",
        s=180,
        color="#D62728",
        edgecolor="black",
        linewidth=0.5,
        zorder=5,
        label="最优Macro-F1",
    )

    for x_value, y_value in zip(ordered[x_column], macro_f1_values):
        ax.annotate(
            f"{y_value:.4f}",
            (x_value, y_value),
            textcoords="offset points",
            xytext=(0, 7),
            ha="center",
            fontsize=8,
            color=macro_line[0].get_color(),
        )
    for x_value, y_value in zip(ordered[x_column], accuracy_values):
        ax.annotate(
            f"{y_value:.4f}",
            (x_value, y_value),
            textcoords="offset points",
            xytext=(0, -13),
            ha="center",
            fontsize=8,
            color=accuracy_line[0].get_color(),
        )

    if log_x:
        ax.set_xscale("log")
    x_label = {"learning_rate": "学习率", "dropout": "Dropout比例"}.get(
        x_column,
        x_column,
    )
    ax.set_xlabel(x_label)
    ax.set_ylabel("验证集分数")
    # 缩窄纵轴只是为了论文图中看清细微差异，不改变原始指标。
    ax.set_ylim(*y_limits)
    ax.grid(axis="y", linestyle="--", alpha=0.35)
    ax.legend()
    fig.tight_layout()
    fig.savefig(output, dpi=150)
    plt.close(fig)
    return output


def run_learning_rate_experiments() -> tuple[pd.DataFrame, Path]:
    """在 Dropout 固定为 0.3 时运行受控学习率实验。"""
    rows = []
    for learning_rate in LEARNING_RATE_VALUES:
        config = build_experiment_config(learning_rate=learning_rate, dropout=0.3)
        rows.append(
            run_single_experiment(
                experiment_group="learning_rate",
                variable_name="learning_rate",
                variable_value=learning_rate,
                config=config,
            )
        )
    output_path = save_experiment_table(rows, LEARNING_RATE_EXPERIMENTS_PATH)
    return pd.DataFrame(rows), output_path


def select_best_learning_rate(results: pd.DataFrame) -> float:
    """先按验证集 Macro-F1、再按 Accuracy 选择学习率。"""
    ranked = results.sort_values(
        ["validation_macro_f1", "validation_accuracy"],
        ascending=[False, False],
    )
    return float(ranked.iloc[0]["learning_rate"])


def run_dropout_experiments(best_learning_rate: float) -> tuple[pd.DataFrame, Path]:
    """使用选定学习率运行受控 Dropout 实验。"""
    rows = []
    for dropout in DROPOUT_VALUES:
        config = build_experiment_config(
            learning_rate=best_learning_rate,
            dropout=dropout,
        )
        rows.append(
            run_single_experiment(
                experiment_group="dropout",
                variable_name="dropout",
                variable_value=dropout,
                config=config,
            )
        )
    output_path = save_experiment_table(rows, DROPOUT_EXPERIMENTS_PATH)
    return pd.DataFrame(rows), output_path


def run_hyperparameter_analysis() -> dict[str, Any]:
    """运行第六阶段完整的受控超参数分析。"""
    # 先选学习率，再固定该学习率做 Dropout 实验，保证一次只改变一个变量。
    learning_rate_results, learning_rate_path = run_learning_rate_experiments()
    best_learning_rate = select_best_learning_rate(learning_rate_results)
    dropout_results, dropout_path = run_dropout_experiments(best_learning_rate)

    learning_rate_plot = plot_sensitivity(
        learning_rate_results,
        x_column="learning_rate",
        output_path=HYPERPARAMETER_FIGURES_DIR / "learning_rate_sensitivity.png",
        log_x=True,
    )
    dropout_plot = plot_sensitivity(
        dropout_results,
        x_column="dropout",
        output_path=HYPERPARAMETER_FIGURES_DIR / "dropout_sensitivity.png",
    )

    return {
        "best_learning_rate": best_learning_rate,
        "learning_rate_results": str(learning_rate_path),
        "dropout_results": str(dropout_path),
        "learning_rate_plot": str(learning_rate_plot),
        "dropout_plot": str(dropout_plot),
        "note": "All metrics are validation-only. The test split is not used.",
    }
