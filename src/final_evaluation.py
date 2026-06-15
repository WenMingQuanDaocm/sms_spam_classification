"""对选定短信分类模型进行最终测试集评估。"""

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
    """加载仅用于最终评估的固定测试集。"""
    return pd.read_csv(path, encoding="utf-8")


def load_train_frame(path: str | Path = TRAIN_DATA_PATH) -> pd.DataFrame:
    """加载固定训练集，用于拟合多数类基线模型。"""
    return pd.read_csv(path, encoding="utf-8")


def load_logistic_model(path: str | Path = LOGISTIC_MODEL_PATH) -> Any:
    """加载已保存的逻辑回归模型。"""
    model_path = Path(path)
    if not model_path.exists():
        raise FileNotFoundError(
            f"Logistic regression model not found: {model_path}. "
            "Run scripts/run_classical_models.py first."
        )
    return joblib.load(model_path)


def load_mlp_checkpoint(path: str | Path = MLP_CHECKPOINT_PATH) -> dict[str, Any]:
    """加载选定的 MLP checkpoint。"""
    checkpoint_path = Path(path)
    if not checkpoint_path.exists():
        raise FileNotFoundError(
            f"MLP checkpoint not found: {checkpoint_path}. "
            "Run scripts/train_mlp.py first."
        )
    return torch.load(checkpoint_path, map_location=get_training_device(), weights_only=False)


def build_mlp_from_checkpoint(checkpoint: dict[str, Any]) -> TextMLP:
    """根据保存的 checkpoint 重建 MLP 模型。"""
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
    """返回 MLP 预测标签和 spam 概率。"""
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
    """构建用于保存测试集预测结果的 DataFrame。"""
    predictions = test_data[[MESSAGE_COLUMN, LABEL_COLUMN, TARGET_COLUMN]].copy()
    predictions["predicted_target"] = predicted_targets
    predictions["predicted_label"] = predictions["predicted_target"].map({0: "ham", 1: "spam"})
    predictions["spam_probability"] = spam_probabilities
    return predictions


def save_prediction_frame(frame: pd.DataFrame, path: str | Path) -> Path:
    """将测试集预测结果保存为 UTF-8 编码的 CSV 文件。"""
    output = Path(path)
    output.parent.mkdir(parents=True, exist_ok=True)
    frame.to_csv(output, index=False, encoding="utf-8")
    return output


def save_error_analysis(
    prediction_frame: pd.DataFrame,
    false_positives_path: str | Path = FALSE_POSITIVES_PATH,
    false_negatives_path: str | Path = FALSE_NEGATIVES_PATH,
) -> dict[str, Path]:
    """保存 MLP 预测中的假阳性和假阴性样本。"""
    # 错误分析聚焦最终选中的 MLP，因为它是测试集中表现最强的模型。
    false_positives = prediction_frame[
        (prediction_frame[TARGET_COLUMN] == 0)
        & (prediction_frame["predicted_target"] == 1)
    ].copy()
    false_negatives = prediction_frame[
        (prediction_frame[TARGET_COLUMN] == 1)
        & (prediction_frame["predicted_target"] == 0)
    ].copy()

    # 按置信度排序，让最有代表性的错误样本排在 CSV 前面。
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
    """保存最终模型对比表。"""
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
    """在训练集上拟合多数类基线，并在测试集上评估。"""
    # 基线模型也只用训练集拟合，保证和学习模型遵守同一实验协议。
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
    """加载已保存的逻辑回归模型，并在测试集上评估。"""
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
    )
    return metrics


def evaluate_mlp_on_test(
    test_data: pd.DataFrame,
    vectorizer: Any,
) -> tuple[dict[str, Any], pd.DataFrame]:
    """加载已保存的 MLP checkpoint，并在测试集上评估。"""
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
    )
    return metrics, prediction_frame


def run_final_test_evaluation() -> dict[str, Any]:
    """在基于验证集完成模型选择后运行最终测试集评估。"""
    # 这是第一次使用固定测试集做模型对比，调参阶段不能调用这里。
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
