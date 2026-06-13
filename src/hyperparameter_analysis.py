"""Controlled MLP hyperparameter sensitivity experiments."""

from __future__ import annotations

import json
import math
from pathlib import Path
from typing import Any

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
from src.train_mlp import train_mlp_validation


LEARNING_RATE_VALUES = (0.01, 0.001, 0.0001)
DROPOUT_VALUES = (0.0, 0.3, 0.5)


def format_float_for_path(value: float) -> str:
    """Return a filesystem-friendly representation of a float."""
    return str(value).replace(".", "_")


def build_experiment_config(**overrides: Any) -> dict[str, Any]:
    """Return the default MLP config with controlled overrides applied."""
    config = dict(MLP_CONFIG)
    config.update(overrides)
    return config


def load_history(path: str | Path) -> pd.DataFrame:
    """Load one MLP training history CSV."""
    return pd.read_csv(path, encoding="utf-8")


def has_overfitting_signal(history: pd.DataFrame) -> bool:
    """Return whether final train/validation curves show a clear gap."""
    final_row = history.iloc[-1]
    best_row = history.loc[history["validation_macro_f1"].idxmax()]
    accuracy_gap = final_row["train_accuracy"] - final_row["validation_accuracy"]
    validation_loss_increased = final_row["validation_loss"] > best_row["validation_loss"]
    return bool(accuracy_gap > 0.03 and validation_loss_increased)


def has_underfitting_signal(history: pd.DataFrame) -> bool:
    """Return whether the run appears not to have fit the training data well."""
    best_row = history.loc[history["validation_macro_f1"].idxmax()]
    return bool(best_row["train_accuracy"] < 0.95 or best_row["validation_macro_f1"] < 0.85)


def training_was_stable(history: pd.DataFrame) -> bool:
    """Return whether losses are finite and no severe divergence is visible."""
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
    """Build one CSV row for a controlled experiment."""
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
    """Run one MLP experiment with isolated output files."""
    value_name = format_float_for_path(variable_value)
    run_name = f"{experiment_group}_{variable_name}_{value_name}"
    metrics_path = METRICS_DIR / "hyperparameter_runs" / f"{run_name}.json"

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
    """Save an experiment summary table as UTF-8 CSV."""
    output_path = Path(path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    pd.DataFrame(rows).to_csv(output_path, index=False, encoding="utf-8")
    return output_path


def plot_sensitivity(
    results: pd.DataFrame,
    x_column: str,
    title: str,
    output_path: str | Path,
    log_x: bool = False,
) -> Path:
    """Plot validation Macro-F1 and Accuracy for one hyperparameter."""
    output = Path(output_path)
    output.parent.mkdir(parents=True, exist_ok=True)
    ordered = results.sort_values(x_column)

    fig, ax = plt.subplots(figsize=(7, 4))
    ax.plot(ordered[x_column], ordered["validation_macro_f1"], marker="o", label="Macro-F1")
    ax.plot(ordered[x_column], ordered["validation_accuracy"], marker="o", label="Accuracy")
    if log_x:
        ax.set_xscale("log")
    ax.set_title(title)
    ax.set_xlabel(x_column)
    ax.set_ylabel("Validation Score")
    ax.set_ylim(0.0, 1.02)
    ax.legend()
    fig.tight_layout()
    fig.savefig(output, dpi=150)
    plt.close(fig)
    return output


def run_learning_rate_experiments() -> tuple[pd.DataFrame, Path]:
    """Run controlled learning-rate experiments with dropout fixed at 0.3."""
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
    """Select the learning rate by validation Macro-F1, then accuracy."""
    ranked = results.sort_values(
        ["validation_macro_f1", "validation_accuracy"],
        ascending=[False, False],
    )
    return float(ranked.iloc[0]["learning_rate"])


def run_dropout_experiments(best_learning_rate: float) -> tuple[pd.DataFrame, Path]:
    """Run controlled dropout experiments using the selected learning rate."""
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
    """Run the full phase-six controlled hyperparameter analysis."""
    learning_rate_results, learning_rate_path = run_learning_rate_experiments()
    best_learning_rate = select_best_learning_rate(learning_rate_results)
    dropout_results, dropout_path = run_dropout_experiments(best_learning_rate)

    learning_rate_plot = plot_sensitivity(
        learning_rate_results,
        x_column="learning_rate",
        title="Learning Rate Sensitivity",
        output_path=HYPERPARAMETER_FIGURES_DIR / "learning_rate_sensitivity.png",
        log_x=True,
    )
    dropout_plot = plot_sensitivity(
        dropout_results,
        x_column="dropout",
        title="Dropout Sensitivity",
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
