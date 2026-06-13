"""Run phase-four majority baseline and logistic regression validation."""

from __future__ import annotations

import json
import sys
from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.train_baseline import run_baseline_validation
from src.train_logistic import run_logistic_validation


def main() -> None:
    """Train and validate classical models without touching the test split."""
    report = {
        "baseline": run_baseline_validation(),
        "logistic_regression": run_logistic_validation(),
        "note": "Metrics are validation-only. The test split is not used here.",
    }
    print(json.dumps(report, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
