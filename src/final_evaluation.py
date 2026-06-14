"""Final test-set evaluation for selected SMS spam classifiers."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import joblib
import pandas as pd
import torch
from sklearn.dummy import DummyClassifier

from src.config import (
    BASELINE_TEST_METRICS_PATH,
    CONFUSION_MATRIX_DIR,
    FALSE_NEGATIVES_PATH,
    FALSE_POSITIVES_PATH,
    LABEL_COLUMN,
    LOGISTIC_MODEL_PATH,
    LOGISTIC_TEST_METRICS_PATH,
    LOGISTIC_TEST_PREDICTIONS_PATH,
    MESSAGE_COLUMN,
    MLP_CHECKPOINT_PATH,
    MLP_CONFIG,
    MLP_TEST_METRICS_PATH,
    MLP_TEST_PREDICTIONS_PATH,
    MODEL_COMPARISON_PATH,
    TARGET_COLUMN,
    TEST_DATA_PATH,
    TFIDF_VECTORIZER_PATH,
    TRAIN_DATA_PATH,
)
from src.evaluate import evaluate_predictions, plot_confusion_matrix, save_metrics
from src.models import TextMLP
from src.train_baseline import train_majority_baseline
from src.train_logistic import load_tfidf_vectorizer
from src.train_mlp import sparse_to_float_tensor
from src.utils import get_training_device


def load_test_frame(path: str | Path = TEST_DATA_PATH) -> pd.DataFrame:
    """Load the fixed test split for final evaluation only."""
    return pd.read_csv(path, encoding="utf-8")


def load_train_frame(path: str | Path = TRAIN_DATA_PATH) -> pd.DataFrame:
    """Load the fixed training split for fitting the majority baseline."""
    return pd.read_csv(path, encoding="utf-8")


def load_logistic_model(path: str | Path = LOGISTIC_MODEL_PATH) -> Any:
    """Load the saved logistic regression model."""
    model_path = Path(path)
    if not model_path.exists():
        raise FileNotFoundError(
            f"Logistic regression model not found: {model_path}. "
            "Run scripts/run_classical_models.py first."
        )
    return joblib.load(model_path)


def load_mlp_checkpoint(path: str | Path = MLP_CHECKPOINT_PATH) -> dict[str, Any]:
    """Load the selected MLP checkpoint."""
    checkpoint_path = Path(path)
    if not checkpoint_path.exists():
        raise FileNotFoundError(
            f"MLP checkpoint not found: {checkpoint_path}. "
            "Run scripts/train_mlp.py first."
        )
    return torch.load(checkpoint_path, map_location=get_training_device(), weights_only=False)


def build_mlp_from_checkpoint(checkpoint: dict[str, Any]) -> TextMLP:
    """Reconstruct the MLP model from a saved checkpoint."""
    config = checkpoint.get("config", MLP_CONFIG)
    model = TextMLP(
        input_dim=int(checkpoint["input_dim"]),
        hidden_layers=tuple(config["hidden_layers"]),
        dropout=float(config["dropout"]),
    )
    model.load_state_dict(checkpoint["model_state_dict"])
    model.eval()
    return model


def predict_mlp_probabilities(
    model: TextMLP,
    test_features: Any,
) -> tuple[list[int], list[float]]:
    """Return MLP predicted labels and spam probabilities."""
    device = get_training_device()
    model = model.to(device)
    features = sparse_to_float_tensor(test_features).to(device)

    with torch.no_grad():
        logits = model(features)
        probabilities = torch.softmax(logits, dim=1)
        predictions = probabilities.argmax(dim=1)
    return predictions.cpu().tolist(), probabilities[:, 1].cpu().tolist()


def build_prediction_frame(
    test_data: pd.DataFrame,
    predicted_targets: list[int],
    spam_probabilities: list[float],
) -> pd.DataFrame:
    """Build a prediction CSV frame for test-set outputs."""
    predictions = test_data[[MESSAGE_COLUMN, LABEL_COLUMN, TARGET_COLUMN]].copy()
    predictions["predicted_target"] = predicted_targets
    predictions["predicted_label"] = predictions["predicted_target"].map({0: "ham", 1: "spam"})
    predictions["spam_probability"] = spam_probabilities
    return predictions


def save_prediction_frame(frame: pd.DataFrame, path: str | Path) -> Path:
    """Save test-set predictions as UTF-8 CSV."""
    output = Path(path)
    output.parent.mkdir(parents=True, exist_ok=True)
    frame.to_csv(output, index=False, encoding="utf-8")
    return output


def save_error_analysis(
    prediction_frame: pd.DataFrame,
    false_positives_path: str | Path = FALSE_POSITIVES_PATH,
    false_negatives_path: str | Path = FALSE_NEGATIVES_PATH,
) -> dict[str, Path]:
    """Save false-positive and false-negative cases from MLP predictions."""
    false_positives = prediction_frame[
        (prediction_frame[TARGET_COLUMN] == 0)
        & (prediction_frame["predicted_target"] == 1)
    ].copy()
    false_negatives = prediction_frame[
        (prediction_frame[TARGET_COLUMN] == 1)
        & (prediction_frame["predicted_target"] == 0)
    ].copy()

    false_positives = false_positives.sort_values(
        "spam_probability",
        ascending=False,
    )
    false_negatives = false_negatives.sort_values(
        "spam_probability",
        ascending=True,
    )

    fp_path = Path(false_positives_path)
    fn_path = Path(false_negatives_path)
    fp_path.parent.mkdir(parents=True, exist_ok=True)
    fn_path.parent.mkdir(parents=True, exist_ok=True)
    false_positives.to_csv(fp_path, index=False, encoding="utf-8")
    false_negatives.to_csv(fn_path, index=False, encoding="utf-8")
    return {"false_positives": fp_path, "false_negatives": fn_path}


def save_model_comparison(metrics_by_model: dict[str, dict[str, Any]]) -> Path:
    """Save a final model comparison table."""
    rows = []
    for model_name, metrics in metrics_by_model.items():
        rows.append(
            {
                "model_name": model_name,
                "split_name": metrics["split_name"],
                "accuracy": metrics["accuracy"],
                "macro_f1": metrics["macro_f1"],
                "spam_precision": metrics["spam_precision"],
                "spam_recall": metrics["spam_recall"],
                "spam_f1": metrics["spam_f1"],
            }
        )
    output = Path(MODEL_COMPARISON_PATH)
    output.parent.mkdir(parents=True, exist_ok=True)
    pd.DataFrame(rows).to_csv(output, index=False, encoding="utf-8")
    return output


def evaluate_baseline_on_test(
    train_data: pd.DataFrame,
    test_data: pd.DataFrame,
) -> dict[str, Any]:
    """Fit the majority baseline on training data and evaluate on test data."""
    model: DummyClassifier = train_majority_baseline(train_data)
    predictions = model.predict(test_data[[MESSAGE_COLUMN]])
    metrics = evaluate_predictions(
        test_data[TARGET_COLUMN],
        predictions,
        model_name="majority_baseline",
        split_name="test",
    )
    metrics["model_config"] = {"strategy": "most_frequent"}
    save_metrics(metrics, BASELINE_TEST_METRICS_PATH)
    return metrics


def evaluate_logistic_on_test(
    test_data: pd.DataFrame,
    vectorizer: Any,
) -> dict[str, Any]:
    """Load saved logistic regression and evaluate it on the test split."""
    model = load_logistic_model()
    test_features = vectorizer.transform(test_data[MESSAGE_COLUMN])
    predictions = model.predict(test_features).tolist()
    probabilities = model.predict_proba(test_features)[:, 1].tolist()
    prediction_frame = build_prediction_frame(test_data, predictions, probabilities)
    prediction_path = save_prediction_frame(
        prediction_frame,
        LOGISTIC_TEST_PREDICTIONS_PATH,
    )

    metrics = evaluate_predictions(
        test_data[TARGET_COLUMN],
        predictions,
        model_name="logistic_regression",
        split_name="test",
    )
    metrics["model_path"] = str(LOGISTIC_MODEL_PATH)
    metrics["prediction_path"] = str(prediction_path)
    save_metrics(metrics, LOGISTIC_TEST_METRICS_PATH)
    plot_confusion_matrix(
        metrics,
        CONFUSION_MATRIX_DIR / "logistic_confusion_matrix.png",
        "逻辑回归测试集混淆矩阵",
    )
    return metrics


def evaluate_mlp_on_test(
    test_data: pd.DataFrame,
    vectorizer: Any,
) -> tuple[dict[str, Any], pd.DataFrame]:
    """Load saved MLP checkpoint and evaluate it on the test split."""
    checkpoint = load_mlp_checkpoint()
    model = build_mlp_from_checkpoint(checkpoint)
    test_features = vectorizer.transform(test_data[MESSAGE_COLUMN])
    predictions, probabilities = predict_mlp_probabilities(model, test_features)
    prediction_frame = build_prediction_frame(test_data, predictions, probabilities)
    prediction_path = save_prediction_frame(prediction_frame, MLP_TEST_PREDICTIONS_PATH)

    metrics = evaluate_predictions(
        test_data[TARGET_COLUMN],
        predictions,
        model_name="mlp",
        split_name="test",
    )
    metrics["checkpoint_path"] = str(MLP_CHECKPOINT_PATH)
    metrics["checkpoint_epoch"] = int(checkpoint["epoch"])
    metrics["validation_macro_f1_at_checkpoint"] = float(checkpoint["validation_macro_f1"])
    metrics["model_config"] = checkpoint.get("config", MLP_CONFIG)
    metrics["prediction_path"] = str(prediction_path)
    save_metrics(metrics, MLP_TEST_METRICS_PATH)
    plot_confusion_matrix(
        metrics,
        CONFUSION_MATRIX_DIR / "mlp_confusion_matrix.png",
        "MLP测试集混淆矩阵",
    )
    return metrics, prediction_frame


def run_final_test_evaluation() -> dict[str, Any]:
    """Run final test-set evaluation after validation-based model selection."""
    train_data = load_train_frame()
    test_data = load_test_frame()
    vectorizer = load_tfidf_vectorizer(TFIDF_VECTORIZER_PATH)

    baseline_metrics = evaluate_baseline_on_test(train_data, test_data)
    logistic_metrics = evaluate_logistic_on_test(test_data, vectorizer)
    mlp_metrics, mlp_predictions = evaluate_mlp_on_test(test_data, vectorizer)
    error_paths = save_error_analysis(mlp_predictions)
    comparison_path = save_model_comparison(
        {
            "majority_baseline": baseline_metrics,
            "logistic_regression": logistic_metrics,
            "mlp": mlp_metrics,
        }
    )

    return {
        "baseline": baseline_metrics,
        "logistic_regression": logistic_metrics,
        "mlp": mlp_metrics,
        "model_comparison_path": str(comparison_path),
        "error_analysis_paths": {name: str(path) for name, path in error_paths.items()},
        "note": "This is the first formal test-set evaluation after validation tuning.",
    }


if __name__ == "__main__":
    print(json.dumps(run_final_test_evaluation(), indent=2, ensure_ascii=False))
