"""Train and validate logistic regression on TF-IDF features."""

from __future__ import annotations

import time
from pathlib import Path
from typing import Any

import joblib
import pandas as pd
from sklearn.linear_model import LogisticRegression

from src.config import (
    LOGISTIC_FEATURE_WEIGHTS_PATH,
    LOGISTIC_METRICS_PATH,
    LOGISTIC_MODEL_PATH,
    LOGISTIC_REGRESSION_CONFIG,
    MESSAGE_COLUMN,
    TARGET_COLUMN,
    TFIDF_VECTORIZER_PATH,
    TRAIN_DATA_PATH,
    VALIDATION_DATA_PATH,
)
from src.evaluate import evaluate_predictions, save_metrics
from src.train_baseline import load_train_validation_splits


def load_tfidf_vectorizer(path: str | Path = TFIDF_VECTORIZER_PATH) -> Any:
    """Load the fitted training-only TF-IDF vectorizer."""
    vectorizer_path = Path(path)
    if not vectorizer_path.exists():
        raise FileNotFoundError(
            f"TF-IDF vectorizer not found: {vectorizer_path}. "
            "Run scripts/prepare_features.py first."
        )
    return joblib.load(vectorizer_path)


def train_logistic_regression(
    train_features: Any,
    train_targets: pd.Series,
    config: dict[str, Any] | None = None,
) -> LogisticRegression:
    """Train logistic regression using the provided sparse TF-IDF features."""
    model_config = LOGISTIC_REGRESSION_CONFIG if config is None else config
    model = LogisticRegression(**model_config)
    model.fit(train_features, train_targets)
    return model


def save_logistic_model(
    model: LogisticRegression,
    path: str | Path = LOGISTIC_MODEL_PATH,
) -> Path:
    """Save the fitted logistic regression model."""
    output_path = Path(path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    joblib.dump(model, output_path)
    return output_path


def save_logistic_feature_weights(
    model: LogisticRegression,
    vectorizer: Any,
    path: str | Path = LOGISTIC_FEATURE_WEIGHTS_PATH,
    top_k: int = 20,
) -> Path:
    """Save the highest-weight spam and ham features for interpretation."""
    feature_names = vectorizer.get_feature_names_out()
    weights = model.coef_[0]

    spam_indices = weights.argsort()[::-1][:top_k]
    ham_indices = weights.argsort()[:top_k]
    rows: list[dict[str, Any]] = []

    for rank, index in enumerate(spam_indices, start=1):
        rows.append(
            {
                "direction": "spam",
                "rank": rank,
                "feature": feature_names[index],
                "weight": float(weights[index]),
            }
        )
    for rank, index in enumerate(ham_indices, start=1):
        rows.append(
            {
                "direction": "ham",
                "rank": rank,
                "feature": feature_names[index],
                "weight": float(weights[index]),
            }
        )

    output_path = Path(path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    pd.DataFrame(rows).to_csv(output_path, index=False, encoding="utf-8")
    return output_path


def run_logistic_validation(
    train_path: str | Path = TRAIN_DATA_PATH,
    validation_path: str | Path = VALIDATION_DATA_PATH,
    vectorizer_path: str | Path = TFIDF_VECTORIZER_PATH,
    model_path: str | Path = LOGISTIC_MODEL_PATH,
    metrics_path: str | Path = LOGISTIC_METRICS_PATH,
    feature_weights_path: str | Path = LOGISTIC_FEATURE_WEIGHTS_PATH,
) -> dict[str, Any]:
    """Train logistic regression and evaluate it on the validation split."""
    train_data, validation_data = load_train_validation_splits(
        train_path,
        validation_path,
    )
    vectorizer = load_tfidf_vectorizer(vectorizer_path)

    train_features = vectorizer.transform(train_data[MESSAGE_COLUMN])
    validation_features = vectorizer.transform(validation_data[MESSAGE_COLUMN])

    start_time = time.perf_counter()
    model = train_logistic_regression(train_features, train_data[TARGET_COLUMN])
    training_time_seconds = time.perf_counter() - start_time

    predictions = model.predict(validation_features)
    metrics = evaluate_predictions(
        validation_data[TARGET_COLUMN],
        predictions,
        model_name="logistic_regression",
        split_name="validation",
    )
    metrics["training_time_seconds"] = float(training_time_seconds)
    metrics["model_config"] = dict(LOGISTIC_REGRESSION_CONFIG)
    metrics["vectorizer_path"] = str(vectorizer_path)
    metrics["model_path"] = str(save_logistic_model(model, model_path))
    metrics["feature_weights_path"] = str(
        save_logistic_feature_weights(model, vectorizer, feature_weights_path)
    )
    save_metrics(metrics, metrics_path)
    return metrics


if __name__ == "__main__":
    import json

    print(json.dumps(run_logistic_validation(), indent=2, ensure_ascii=False))
