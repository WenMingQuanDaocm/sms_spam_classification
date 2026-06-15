"""短信数据质量检查和保守预处理工具。"""

from __future__ import annotations

from dataclasses import asdict, dataclass
import json
from pathlib import Path
from typing import Any

import joblib
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.model_selection import train_test_split

from src.config import (
    LABEL_COLUMN,
    LABEL_TO_TARGET,
    MESSAGE_COLUMN,
    PREPROCESSING_SUMMARY_PATH,
    TARGET_COLUMN,
    TEST_DATA_PATH,
    TEST_SIZE,
    TFIDF_CONFIG,
    TFIDF_VECTORIZER_PATH,
    TRAIN_DATA_PATH,
    TRAIN_SIZE,
    VALIDATION_DATA_PATH,
    VALIDATION_SIZE,
    RANDOM_STATE,
    VALID_LABELS,
)
from src.data_loader import validate_required_columns


@dataclass(frozen=True)
class DataQualityReport:
    """原始数据质量检查摘要。"""

    total_rows: int
    missing_value_counts: dict[str, int]
    empty_message_count: int
    invalid_label_count: int
    exact_duplicate_row_count: int
    duplicate_rows_to_remove: int
    duplicate_message_count: int
    label_conflict_message_count: int
    class_counts: dict[str, int]

    def to_dict(self) -> dict[str, Any]:
        """返回可保存为 JSON 的报告字典。"""
        return asdict(self)


@dataclass(frozen=True)
class SplitReport:
    """固定训练集、验证集、测试集切分摘要。"""

    split_sizes: dict[str, int]
    class_distribution: dict[str, dict[str, dict[str, float | int]]]
    overlap_counts: dict[str, int]

    def to_dict(self) -> dict[str, Any]:
        """返回可保存为 JSON 的切分报告字典。"""
        return asdict(self)


def normalize_raw_columns(data: pd.DataFrame) -> pd.DataFrame:
    """清理标签和短信两端空白，同时保留短信正文内容。"""
    validate_required_columns(data)
    normalized = data.copy()
    normalized[LABEL_COLUMN] = (
        normalized[LABEL_COLUMN].astype("string").str.strip().str.lower()
    )
    normalized[MESSAGE_COLUMN] = normalized[MESSAGE_COLUMN].astype("string").str.strip()
    return normalized


def get_empty_message_mask(data: pd.DataFrame) -> pd.Series:
    """返回清理空白后短信正文为空的行掩码。"""
    validate_required_columns(data)
    messages = data[MESSAGE_COLUMN].astype("string")
    return messages.notna() & messages.str.strip().eq("")


def get_invalid_label_mask(data: pd.DataFrame) -> pd.Series:
    """返回标签不属于允许集合的行掩码。"""
    validate_required_columns(data)
    labels = data[LABEL_COLUMN].astype("string")
    return labels.notna() & ~labels.isin(VALID_LABELS)


def get_label_conflicts(data: pd.DataFrame) -> pd.DataFrame:
    """返回同一短信文本对应多个标签的冲突行。"""
    validate_required_columns(data)
    comparable = data.dropna(subset=[LABEL_COLUMN, MESSAGE_COLUMN])
    label_counts = comparable.groupby(MESSAGE_COLUMN)[LABEL_COLUMN].nunique()
    conflict_messages = label_counts[label_counts > 1].index
    if len(conflict_messages) == 0:
        return data.iloc[0:0].copy()
    return (
        data[data[MESSAGE_COLUMN].isin(conflict_messages)]
        .sort_values([MESSAGE_COLUMN, LABEL_COLUMN])
        .reset_index(drop=True)
    )


def build_quality_report(data: pd.DataFrame) -> DataQualityReport:
    """在不修改输入数据的前提下构建数据质量报告。"""
    normalized = normalize_raw_columns(data)
    missing_counts = {
        column: int(normalized[column].isna().sum())
        for column in (LABEL_COLUMN, MESSAGE_COLUMN)
    }

    labels_for_counts = normalized[LABEL_COLUMN].astype("object")
    labels_for_counts = labels_for_counts.where(normalized[LABEL_COLUMN].notna(), "<missing>")
    class_counts = {
        str(label): int(count)
        for label, count in labels_for_counts.value_counts(dropna=False).sort_index().items()
    }

    duplicate_messages = normalized.duplicated(subset=[MESSAGE_COLUMN], keep=False)
    conflicts = get_label_conflicts(normalized)

    return DataQualityReport(
        total_rows=int(len(normalized)),
        missing_value_counts=missing_counts,
        empty_message_count=int(get_empty_message_mask(normalized).sum()),
        invalid_label_count=int(get_invalid_label_mask(normalized).sum()),
        exact_duplicate_row_count=int(
            normalized.duplicated(subset=[LABEL_COLUMN, MESSAGE_COLUMN], keep=False).sum()
        ),
        duplicate_rows_to_remove=int(
            normalized.duplicated(subset=[LABEL_COLUMN, MESSAGE_COLUMN], keep="first").sum()
        ),
        duplicate_message_count=int(duplicate_messages.sum()),
        label_conflict_message_count=int(conflicts[MESSAGE_COLUMN].nunique()),
        class_counts=class_counts,
    )


def clean_sms_data(data: pd.DataFrame) -> pd.DataFrame:
    """清洗原始短信数据，为后续切分做准备。

    本函数只执行方案允许的预处理：去除首尾空白、删除空短信、删除精确重复样本、
    校验标签并添加数值目标列。标签冲突会直接报错，不会静默修正。
    """
    normalized = normalize_raw_columns(data)
    report = build_quality_report(normalized)

    # 缺失标签或短信会让后续指标含义不清，因此在切分前直接报错。
    missing_total = sum(report.missing_value_counts.values())
    if missing_total > 0:
        raise ValueError(f"Missing values found: {report.missing_value_counts}")
    if report.invalid_label_count > 0:
        invalid_labels = (
            normalized.loc[get_invalid_label_mask(normalized), LABEL_COLUMN]
            .dropna()
            .unique()
            .tolist()
        )
        raise ValueError(f"Invalid label(s) found: {invalid_labels}")

    non_empty = normalized.loc[~get_empty_message_mask(normalized)].copy()
    conflicts = get_label_conflicts(non_empty)
    if not conflicts.empty:
        # 相同短信对应不同标签会破坏目标定义，不能静默修正。
        raise ValueError(
            "Label conflicts found for identical message text. "
            "Review conflicts before continuing."
        )

    # 切分前删除精确重复样本，防止同一短信泄漏到不同数据集。
    cleaned = (
        non_empty.drop_duplicates(subset=[LABEL_COLUMN, MESSAGE_COLUMN], keep="first")
        .reset_index(drop=True)
        .copy()
    )
    cleaned[TARGET_COLUMN] = cleaned[LABEL_COLUMN].map(LABEL_TO_TARGET).astype(int)
    return cleaned


def save_cleaned_preview(data: pd.DataFrame, path: str | Path, rows: int = 20) -> Path:
    """保存少量清洗后样本供人工查看，不作为正式数据切分。"""
    output_path = Path(path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    data.head(rows).to_csv(output_path, index=False, encoding="utf-8")
    return output_path


def validate_split_input(data: pd.DataFrame) -> None:
    """在训练集、验证集、测试集切分前校验清洗后数据。"""
    validate_required_columns(data, (LABEL_COLUMN, MESSAGE_COLUMN, TARGET_COLUMN))
    expected_total = TRAIN_SIZE + VALIDATION_SIZE + TEST_SIZE
    if abs(expected_total - 1.0) > 1e-9:
        raise ValueError(
            "TRAIN_SIZE + VALIDATION_SIZE + TEST_SIZE must equal 1.0, "
            f"got {expected_total}."
        )
    if data[TARGET_COLUMN].nunique() < 2:
        raise ValueError("Stratified splitting requires at least two classes.")


def split_train_validation_test(data: pd.DataFrame) -> dict[str, pd.DataFrame]:
    """创建固定 60/20/20 的分层训练集、验证集和测试集。"""
    validate_split_input(data)

    # 先切出训练集，再把剩余 40% 切成验证集和测试集。
    train_data, temp_data = train_test_split(
        data,
        train_size=TRAIN_SIZE,
        stratify=data[TARGET_COLUMN],
        random_state=RANDOM_STATE,
    )
    # 第二次切分面对的是剩余 40%，所以要换算比例才能得到最终 60/20/20。
    relative_test_size = TEST_SIZE / (VALIDATION_SIZE + TEST_SIZE)
    validation_data, test_data = train_test_split(
        temp_data,
        test_size=relative_test_size,
        stratify=temp_data[TARGET_COLUMN],
        random_state=RANDOM_STATE,
    )

    return {
        "train": train_data.reset_index(drop=True),
        "validation": validation_data.reset_index(drop=True),
        "test": test_data.reset_index(drop=True),
    }


def save_splits(
    splits: dict[str, pd.DataFrame],
    train_path: str | Path = TRAIN_DATA_PATH,
    validation_path: str | Path = VALIDATION_DATA_PATH,
    test_path: str | Path = TEST_DATA_PATH,
) -> dict[str, Path]:
    """将训练集、验证集和测试集保存为 UTF-8 编码的 CSV 文件。"""
    output_paths = {
        "train": Path(train_path),
        "validation": Path(validation_path),
        "test": Path(test_path),
    }
    for split_name, output_path in output_paths.items():
        output_path.parent.mkdir(parents=True, exist_ok=True)
        splits[split_name].to_csv(output_path, index=False, encoding="utf-8")
    return output_paths


def get_class_distribution(data: pd.DataFrame) -> dict[str, dict[str, float | int]]:
    """返回某个数据切分中的类别数量和比例。"""
    counts = data[LABEL_COLUMN].value_counts().reindex(VALID_LABELS, fill_value=0)
    total = len(data)
    return {
        label: {
            "count": int(count),
            "proportion": float(count / total) if total else 0.0,
        }
        for label, count in counts.items()
    }


def find_split_overlaps(splits: dict[str, pd.DataFrame]) -> dict[str, int]:
    """统计不同数据切分之间的短信文本重叠数量。"""
    message_sets = {
        split_name: set(split_data[MESSAGE_COLUMN].astype(str))
        for split_name, split_data in splits.items()
    }
    return {
        "train_validation": len(message_sets["train"] & message_sets["validation"]),
        "train_test": len(message_sets["train"] & message_sets["test"]),
        "validation_test": len(message_sets["validation"] & message_sets["test"]),
    }


def validate_no_split_overlap(splits: dict[str, pd.DataFrame]) -> None:
    """如果同一短信文本出现在多个切分中，则直接报错。"""
    overlap_counts = find_split_overlaps(splits)
    if any(count > 0 for count in overlap_counts.values()):
        raise ValueError(f"Split overlap detected: {overlap_counts}")


def build_split_report(splits: dict[str, pd.DataFrame]) -> SplitReport:
    """构建切分大小、类别平衡和重叠情况报告。"""
    return SplitReport(
        split_sizes={name: int(len(split_data)) for name, split_data in splits.items()},
        class_distribution={
            name: get_class_distribution(split_data)
            for name, split_data in splits.items()
        },
        overlap_counts=find_split_overlaps(splits),
    )


def fit_tfidf_vectorizer(train_data: pd.DataFrame) -> TfidfVectorizer:
    """只在训练集上拟合 TF-IDF 向量器。"""
    validate_required_columns(train_data, (MESSAGE_COLUMN,))
    # 只在训练集拟合 TF-IDF，避免验证集和测试集的词表统计泄漏。
    vectorizer = TfidfVectorizer(**TFIDF_CONFIG)
    vectorizer.fit(train_data[MESSAGE_COLUMN])
    return vectorizer


def transform_splits(
    vectorizer: TfidfVectorizer,
    splits: dict[str, pd.DataFrame],
) -> dict[str, Any]:
    """使用已拟合的向量器转换训练集、验证集和测试集短信。"""
    # 验证集和测试集只使用冻结后的向量器 transform，不能重新 fit。
    return {
        split_name: vectorizer.transform(split_data[MESSAGE_COLUMN])
        for split_name, split_data in splits.items()
    }


def save_tfidf_vectorizer(
    vectorizer: TfidfVectorizer,
    path: str | Path = TFIDF_VECTORIZER_PATH,
) -> Path:
    """保存已拟合的 TF-IDF 向量器。"""
    output_path = Path(path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    joblib.dump(vectorizer, output_path)
    return output_path


def build_preprocessing_summary(
    cleaned_data: pd.DataFrame,
    splits: dict[str, pd.DataFrame],
    vectorizer: TfidfVectorizer,
    transformed_splits: dict[str, Any],
) -> dict[str, Any]:
    """构建第三阶段预处理的复现性摘要。"""
    split_report = build_split_report(splits)
    feature_shapes = {
        split_name: [int(matrix.shape[0]), int(matrix.shape[1])]
        for split_name, matrix in transformed_splits.items()
    }
    return {
        "cleaned_rows": int(len(cleaned_data)),
        "split_report": split_report.to_dict(),
        "tfidf_config": TFIDF_CONFIG,
        "tfidf_vocabulary_size": int(len(vectorizer.vocabulary_)),
        "feature_shapes": feature_shapes,
        "notes": [
            "TF-IDF vectorizer was fit on the training split only.",
            "Validation and test splits were transformed only.",
        ],
    }


def save_preprocessing_summary(
    summary: dict[str, Any],
    path: str | Path = PREPROCESSING_SUMMARY_PATH,
) -> Path:
    """将第三阶段预处理摘要保存为 UTF-8 编码的 JSON 文件。"""
    output_path = Path(path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(
        json.dumps(summary, indent=2, ensure_ascii=False),
        encoding="utf-8",
    )
    return output_path
