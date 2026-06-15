"""Regenerate all project figures from saved data and experiment outputs."""

from __future__ import annotations

import json
import sys
from pathlib import Path

import pandas as pd


PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.config import (
    CONFUSION_MATRIX_DIR,
    DROPOUT_EXPERIMENTS_PATH,
    EDA_FIGURES_DIR,
    HYPERPARAMETER_FIGURES_DIR,
    LEARNING_RATE_EXPERIMENTS_PATH,
    LOGISTIC_TEST_METRICS_PATH,
    METRICS_DIR,
    MLP_TEST_METRICS_PATH,
    MLP_TRAINING_HISTORY_PATH,
    MODEL_COMPARISON_FIGURE_PATH,
    MODEL_COMPARISON_PATH,
    RAW_DATA_PATH,
    TRAINING_FIGURES_DIR,
)
from src.data_loader import load_raw_sms_data
from src.eda import generate_eda_outputs
from src.evaluate import plot_confusion_matrix, plot_model_comparison
from src.hyperparameter_analysis import plot_sensitivity
from src.preprocess import clean_sms_data
from src.train_mlp import plot_training_curves


def _load_json(path: str | Path) -> dict:
    """Load a UTF-8 JSON file."""
    return json.loads(Path(path).read_text(encoding="utf-8"))


def _load_history_records(path: str | Path) -> list[dict]:
    """Load a training-history CSV as plotting records."""
    return pd.read_csv(path, encoding="utf-8").to_dict(orient="records")


def regenerate_eda_figures() -> dict[str, str]:
    """Regenerate EDA figures from the raw dataset."""
    raw_data = load_raw_sms_data(RAW_DATA_PATH)
    cleaned_data = clean_sms_data(raw_data)
    outputs = generate_eda_outputs(cleaned_data, figures_dir=EDA_FIGURES_DIR)
    return {name: str(path) for name, path in outputs.items() if name != "summary"}


def regenerate_training_figures() -> dict[str, str]:
    """Regenerate the default MLP training-curve figure."""
    output_path = plot_training_curves(
        _load_history_records(MLP_TRAINING_HISTORY_PATH),
        TRAINING_FIGURES_DIR / "mlp_training_curves.png",
        best_epoch=5,
        show_titles=True,
    )
    return {"mlp_training_curves": str(output_path)}


def regenerate_hyperparameter_figures() -> dict[str, str]:
    """Regenerate hyperparameter sensitivity plots and per-run curves."""
    outputs: dict[str, str] = {}
    learning_rate_results = pd.read_csv(LEARNING_RATE_EXPERIMENTS_PATH, encoding="utf-8")
    dropout_results = pd.read_csv(DROPOUT_EXPERIMENTS_PATH, encoding="utf-8")

    outputs["learning_rate_sensitivity"] = str(
        plot_sensitivity(
            learning_rate_results,
            x_column="learning_rate",
            output_path=HYPERPARAMETER_FIGURES_DIR / "learning_rate_sensitivity.png",
            log_x=True,
        )
    )
    outputs["dropout_sensitivity"] = str(
        plot_sensitivity(
            dropout_results,
            x_column="dropout",
            output_path=HYPERPARAMETER_FIGURES_DIR / "dropout_sensitivity.png",
        )
    )

    runs_dir = METRICS_DIR / "hyperparameter_runs"
    for history_path in sorted(runs_dir.glob("*_history.csv")):
        run_name = history_path.name.removesuffix("_history.csv")
        output_path = plot_training_curves(
            _load_history_records(history_path),
            HYPERPARAMETER_FIGURES_DIR / f"{run_name}_curves.png",
            show_titles=False,
        )
        outputs[f"{run_name}_curves"] = str(output_path)
    return outputs


def regenerate_confusion_matrix_figures() -> dict[str, str]:
    """Regenerate final test-set confusion matrices from saved metrics."""
    outputs = {
        "logistic_confusion_matrix": plot_confusion_matrix(
            _load_json(LOGISTIC_TEST_METRICS_PATH),
            CONFUSION_MATRIX_DIR / "logistic_confusion_matrix.png",
        ),
        "mlp_confusion_matrix": plot_confusion_matrix(
            _load_json(MLP_TEST_METRICS_PATH),
            CONFUSION_MATRIX_DIR / "mlp_confusion_matrix.png",
        ),
    }
    return {name: str(path) for name, path in outputs.items()}


def regenerate_model_comparison_figure() -> dict[str, str]:
    """Regenerate the final model-comparison figure."""
    comparison = pd.read_csv(MODEL_COMPARISON_PATH, encoding="utf-8")
    output_path = plot_model_comparison(comparison, MODEL_COMPARISON_FIGURE_PATH)
    return {"model_comparison": str(output_path)}


def main() -> None:
    """Regenerate all project figures without retraining models."""
    outputs = {
        "eda": regenerate_eda_figures(),
        "training": regenerate_training_figures(),
        "hyperparameters": regenerate_hyperparameter_figures(),
        "confusion_matrix": regenerate_confusion_matrix_figures(),
        "model_comparison": regenerate_model_comparison_figure(),
        "note": "Figures were regenerated from saved data and metrics; models were not retrained.",
    }
    print(json.dumps(outputs, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
