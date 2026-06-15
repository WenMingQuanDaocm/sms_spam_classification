"""运行第六阶段 MLP 超参数敏感性实验。"""

from __future__ import annotations

import json
import sys
from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    # 加入项目根目录，支持从任意当前目录调用脚本。
    sys.path.insert(0, str(PROJECT_ROOT))

from src.hyperparameter_analysis import run_hyperparameter_analysis


def main() -> None:
    """运行受控学习率和 Dropout 实验。"""
    # 敏感性实验一次只改变一个超参数，方便解释因果影响。
    print(json.dumps(run_hyperparameter_analysis(), indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
