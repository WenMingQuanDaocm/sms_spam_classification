"""运行第三阶段的数据切分和 TF-IDF 准备流程。"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    # 允许从项目根目录以外的位置运行脚本时仍能导入 src 包。
    sys.path.insert(0, str(PROJECT_ROOT))

from src.config import (
    PREPROCESSING_SUMMARY_PATH,
    RAW_DATA_PATH,
    TEST_DATA_PATH,
    TFIDF_VECTORIZER_PATH,
    TRAIN_DATA_PATH,
    VALIDATION_DATA_PATH,
)
from src.data_loader import load_raw_sms_data
from src.preprocess import (
    build_preprocessing_summary,
    build_split_report,
    clean_sms_data,
    fit_tfidf_vectorizer,
    save_preprocessing_summary,
    save_splits,
    save_tfidf_vectorizer,
    split_train_validation_test,
    transform_splits,
    validate_no_split_overlap,
)


def parse_args() -> argparse.Namespace:
    """解析预处理脚本的命令行参数。"""
    parser = argparse.ArgumentParser(description="Prepare data splits and TF-IDF.")
    parser.add_argument(
        "--raw-data",
        type=Path,
        default=RAW_DATA_PATH,
        help="Path to the raw UCI SMSSpamCollection file.",
    )
    parser.add_argument(
        "--train-path",
        type=Path,
        default=TRAIN_DATA_PATH,
        help="Output path for the training split CSV.",
    )
    parser.add_argument(
        "--validation-path",
        type=Path,
        default=VALIDATION_DATA_PATH,
        help="Output path for the validation split CSV.",
    )
    parser.add_argument(
        "--test-path",
        type=Path,
        default=TEST_DATA_PATH,
        help="Output path for the test split CSV.",
    )
    parser.add_argument(
        "--vectorizer-path",
        type=Path,
        default=TFIDF_VECTORIZER_PATH,
        help="Output path for the fitted TF-IDF vectorizer.",
    )
    parser.add_argument(
        "--summary-path",
        type=Path,
        default=PREPROCESSING_SUMMARY_PATH,
        help="Output path for the preprocessing summary JSON.",
    )
    return parser.parse_args()


def main() -> None:
    """运行数据清洗、分层切分和 TF-IDF 拟合。"""
    args = parse_args()

    raw_data = load_raw_sms_data(args.raw_data)
    cleaned_data = clean_sms_data(raw_data)
    splits = split_train_validation_test(cleaned_data)
    # 切分后再次检查重叠，作为防数据泄漏的最后一道保护。
    validate_no_split_overlap(splits)

    split_paths = save_splits(
        splits,
        train_path=args.train_path,
        validation_path=args.validation_path,
        test_path=args.test_path,
    )
    vectorizer = fit_tfidf_vectorizer(splits["train"])
    transformed_splits = transform_splits(vectorizer, splits)
    vectorizer_path = save_tfidf_vectorizer(vectorizer, args.vectorizer_path)

    summary = build_preprocessing_summary(
        cleaned_data,
        splits,
        vectorizer,
        transformed_splits,
    )
    summary_path = save_preprocessing_summary(summary, args.summary_path)

    report = {
        "split_report": build_split_report(splits).to_dict(),
        "split_paths": {name: str(path) for name, path in split_paths.items()},
        "vectorizer_path": str(vectorizer_path),
        "summary_path": str(summary_path),
        "feature_shapes": summary["feature_shapes"],
    }
    print(json.dumps(report, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
