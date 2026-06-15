"""生成最终模型综合性能对比图。"""

from __future__ import annotations

import json
import sys
from pathlib import Path

import pandas as pd


PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    # 绘图脚本复用 src.evaluate 中的统一画图函数。
    sys.path.insert(0, str(PROJECT_ROOT))

from src.config import MODEL_COMPARISON_FIGURE_PATH, MODEL_COMPARISON_PATH
from src.evaluate import plot_model_comparison


def main() -> None:
    """读取最终测试指标并保存模型对比图。"""
    comparison_path = Path(MODEL_COMPARISON_PATH)
    if not comparison_path.exists():
        # 没有最终评估结果时不能凭空画对比图。
        raise FileNotFoundError(
            f"Model comparison metrics not found: {comparison_path}. "
            "Run scripts/run_final_evaluation.py first."
        )

    comparison_frame = pd.read_csv(comparison_path, encoding="utf-8")
    figure_path = plot_model_comparison(
        comparison_frame,
        MODEL_COMPARISON_FIGURE_PATH,
    )
    print(
        json.dumps(
            {
                "model_comparison_path": str(comparison_path),
                "figure_path": str(figure_path),
            },
            indent=2,
            ensure_ascii=False,
        )
    )


if __name__ == "__main__":
    main()
