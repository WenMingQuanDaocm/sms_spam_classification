"""运行第二阶段的数据加载、质量检查、清洗和 EDA 输出。"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    # 让脚本无论从哪里启动，都能找到项目内的 src 模块。
    sys.path.insert(0, str(PROJECT_ROOT))

from src.config import EDA_FIGURES_DIR, METRICS_DIR, RAW_DATA_PATH
from src.data_loader import load_raw_sms_data
from src.eda import generate_eda_outputs
from src.preprocess import build_quality_report, clean_sms_data


def parse_args() -> argparse.Namespace:
    """解析 EDA 脚本的命令行参数。"""
    parser = argparse.ArgumentParser(description="Run SMS spam EDA.")
    parser.add_argument(
        "--raw-data",
        type=Path,
        default=RAW_DATA_PATH,
        help="Path to the raw UCI SMSSpamCollection file.",
    )
    parser.add_argument(
        "--figures-dir",
        type=Path,
        default=EDA_FIGURES_DIR,
        help="Directory for EDA figures.",
    )
    parser.add_argument(
        "--summary-path",
        type=Path,
        default=METRICS_DIR / "eda_summary.json",
        help="Path for the EDA summary JSON.",
    )
    return parser.parse_args()


def main() -> None:
    """运行第二阶段 EDA 工作流。"""
    args = parse_args()

    raw_data = load_raw_sms_data(args.raw_data)
    raw_quality_report = build_quality_report(raw_data).to_dict()
    cleaned_data = clean_sms_data(raw_data)
    # 同时保存清洗前后质量报告，便于论文说明数据处理效果。
    cleaned_quality_report = build_quality_report(cleaned_data).to_dict()
    outputs = generate_eda_outputs(
        cleaned_data,
        figures_dir=args.figures_dir,
        summary_path=args.summary_path,
    )

    report = {
        "raw_quality_report": raw_quality_report,
        "cleaned_quality_report": cleaned_quality_report,
        "outputs": {name: str(path) for name, path in outputs.items()},
    }
    print(json.dumps(report, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
