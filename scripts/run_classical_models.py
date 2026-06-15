"""运行第四阶段的多数类基线和逻辑回归验证。"""

from __future__ import annotations

import json
import sys
from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    # 命令行脚本先把项目根目录加入导入路径，避免找不到 src。
    sys.path.insert(0, str(PROJECT_ROOT))

from src.train_baseline import run_baseline_validation
from src.train_logistic import run_logistic_validation


def main() -> None:
    """训练并验证传统模型，不使用测试集。"""
    # 第四阶段只使用验证集评估，测试集留到最终阶段。
    report = {
        "baseline": run_baseline_validation(),
        "logistic_regression": run_logistic_validation(),
        "note": "Metrics are validation-only. The test split is not used here.",
    }
    print(json.dumps(report, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
