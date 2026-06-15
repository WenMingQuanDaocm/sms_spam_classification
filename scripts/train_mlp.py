"""运行第五阶段 MLP 训练和验证。"""

from __future__ import annotations

import json
import sys
from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    # 保证直接运行脚本时可以导入 src.train_mlp。
    sys.path.insert(0, str(PROJECT_ROOT))

from src.train_mlp import train_mlp_validation


def main() -> None:
    """只使用训练集和验证集训练默认 MLP。"""
    # 第五阶段训练默认 MLP，仍然不读取测试集。
    report = train_mlp_validation()
    report["note"] = "Metrics are validation-only. The test split is not used here."
    print(json.dumps(report, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
