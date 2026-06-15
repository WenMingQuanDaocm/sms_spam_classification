"""运行第七阶段最终测试集评估。"""

from __future__ import annotations

import json
import sys
from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    # 最终评估脚本需要导入 src 中保存的评估流程。
    sys.path.insert(0, str(PROJECT_ROOT))

from src.final_evaluation import run_final_test_evaluation


def main() -> None:
    """对选定模型运行最终测试集评估。"""
    # 第七阶段才正式使用测试集，避免调参阶段偷看测试结果。
    print(json.dumps(run_final_test_evaluation(), indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
