"""Central project configuration.

This module stores paths and default experiment constants only. Training,
data loading, and evaluation logic are added in later stages.
"""

from pathlib import Path
from typing import Any


PROJECT_ROOT = Path(__file__).resolve().parents[1]

DATA_DIR = PROJECT_ROOT / "data"
RAW_DATA_DIR = DATA_DIR / "raw"
PROCESSED_DATA_DIR = DATA_DIR / "processed"
RAW_DATA_PATH = RAW_DATA_DIR / "SMSSpamCollection"
TRAIN_DATA_PATH = PROCESSED_DATA_DIR / "train.csv"
VALIDATION_DATA_PATH = PROCESSED_DATA_DIR / "val.csv"
TEST_DATA_PATH = PROCESSED_DATA_DIR / "test.csv"

FIGURES_DIR = PROJECT_ROOT / "figures"
EDA_FIGURES_DIR = FIGURES_DIR / "eda"
TRAINING_FIGURES_DIR = FIGURES_DIR / "training"
CONFUSION_MATRIX_DIR = FIGURES_DIR / "confusion_matrix"
HYPERPARAMETER_FIGURES_DIR = FIGURES_DIR / "hyperparameters"

MODEL_ARTIFACTS_DIR = PROJECT_ROOT / "models"
LOGISTIC_MODEL_DIR = MODEL_ARTIFACTS_DIR / "logistic_regression"
MLP_MODEL_DIR = MODEL_ARTIFACTS_DIR / "mlp"
TFIDF_VECTORIZER_PATH = MODEL_ARTIFACTS_DIR / "tfidf_vectorizer.joblib"
LOGISTIC_MODEL_PATH = LOGISTIC_MODEL_DIR / "logistic_regression.joblib"
MLP_CHECKPOINT_PATH = MLP_MODEL_DIR / "best_model.pt"

RESULTS_DIR = PROJECT_ROOT / "results"
METRICS_DIR = RESULTS_DIR / "metrics"
PREDICTIONS_DIR = RESULTS_DIR / "predictions"
ERROR_ANALYSIS_DIR = RESULTS_DIR / "error_analysis"
PREPROCESSING_SUMMARY_PATH = METRICS_DIR / "preprocessing_summary.json"
BASELINE_METRICS_PATH = METRICS_DIR / "baseline_metrics.json"
LOGISTIC_METRICS_PATH = METRICS_DIR / "logistic_metrics.json"
LOGISTIC_FEATURE_WEIGHTS_PATH = METRICS_DIR / "logistic_feature_weights.csv"
MLP_VALIDATION_METRICS_PATH = METRICS_DIR / "mlp_validation_metrics.json"
MLP_TRAINING_HISTORY_PATH = METRICS_DIR / "mlp_training_history.csv"

RANDOM_STATE = 42
TRAIN_SIZE = 0.60
VALIDATION_SIZE = 0.20
TEST_SIZE = 0.20

LABEL_COLUMN = "label"
MESSAGE_COLUMN = "message"
TARGET_COLUMN = "target"
VALID_LABELS = ("ham", "spam")
LABEL_TO_TARGET = {"ham": 0, "spam": 1}
TARGET_TO_LABEL = {0: "ham", 1: "spam"}

TFIDF_CONFIG: dict[str, Any] = {
    "lowercase": True,
    "ngram_range": (1, 2),
    "min_df": 2,
    "max_df": 0.98,
    "max_features": 5000,
    "sublinear_tf": True,
}

LOGISTIC_REGRESSION_CONFIG: dict[str, Any] = {
    "C": 1.0,
    "max_iter": 2000,
    "random_state": RANDOM_STATE,
}

MLP_CONFIG: dict[str, Any] = {
    "hidden_layers": (128, 64),
    "dropout": 0.3,
    "learning_rate": 0.001,
    "weight_decay": 1e-4,
    "batch_size": 64,
    "max_epochs": 100,
    "early_stopping_patience": 8,
}
