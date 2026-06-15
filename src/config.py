"""项目集中配置。

本模块只保存路径和默认实验常量；训练、数据加载和评估逻辑放在其他模块中。
"""

from pathlib import Path
from typing import Any


# 统一从项目根目录派生路径，避免脚本从不同工作目录运行时找错文件。
PROJECT_ROOT = Path(__file__).resolve().parents[1]

# 数据、图像、模型和结果路径集中配置，便于复现实验和检查输出。
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
MODEL_COMPARISON_FIGURE_PATH = FIGURES_DIR / "model_comparison.png"
MATPLOTLIB_CACHE_DIR = PROJECT_ROOT / ".matplotlib-cache"

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
LEARNING_RATE_EXPERIMENTS_PATH = METRICS_DIR / "learning_rate_experiments.csv"
DROPOUT_EXPERIMENTS_PATH = METRICS_DIR / "dropout_experiments.csv"
BASELINE_TEST_METRICS_PATH = METRICS_DIR / "baseline_test_metrics.json"
LOGISTIC_TEST_METRICS_PATH = METRICS_DIR / "logistic_test_metrics.json"
MLP_TEST_METRICS_PATH = METRICS_DIR / "mlp_test_metrics.json"
MODEL_COMPARISON_PATH = METRICS_DIR / "model_comparison.csv"
LOGISTIC_TEST_PREDICTIONS_PATH = PREDICTIONS_DIR / "logistic_test_predictions.csv"
MLP_TEST_PREDICTIONS_PATH = PREDICTIONS_DIR / "mlp_test_predictions.csv"
FALSE_POSITIVES_PATH = ERROR_ANALYSIS_DIR / "false_positives.csv"
FALSE_NEGATIVES_PATH = ERROR_ANALYSIS_DIR / "false_negatives.csv"

RANDOM_STATE = 42
TRAIN_SIZE = 0.60
VALIDATION_SIZE = 0.20
TEST_SIZE = 0.20

# 标签映射固定为 ham=0、spam=1，后续指标和混淆矩阵都依赖这个约定。
LABEL_COLUMN = "label"
MESSAGE_COLUMN = "message"
TARGET_COLUMN = "target"
VALID_LABELS = ("ham", "spam")
LABEL_TO_TARGET = {"ham": 0, "spam": 1}
TARGET_TO_LABEL = {0: "ham", 1: "spam"}

TFIDF_CONFIG: dict[str, Any] = {
    # 使用 unigram+bigram 和子线性 TF，兼顾短信短文本中的词和短语信号。
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
    # MLP 默认结构与项目方案保持一致，调参实验只改变受控变量。
    "hidden_layers": (128, 64),
    "dropout": 0.3,
    "learning_rate": 0.001,
    "weight_decay": 1e-4,
    "batch_size": 64,
    "max_epochs": 100,
    "early_stopping_patience": 8,
}
