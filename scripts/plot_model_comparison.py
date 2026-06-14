"""Generate the final model performance comparison figure."""

from __future__ import annotations

import json
import sys
from pathlib import Path

import pandas as pd


PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.config import MODEL_COMPARISON_FIGURE_PATH, MODEL_COMPARISON_PATH
from src.evaluate import plot_model_comparison


def main() -> None:
    """Read final test metrics and save the comparison plot."""
    comparison_path = Path(MODEL_COMPARISON_PATH)
    if not comparison_path.exists():
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
