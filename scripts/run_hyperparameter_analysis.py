"""Run phase-six MLP hyperparameter sensitivity experiments."""

from __future__ import annotations

import json
import sys
from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.hyperparameter_analysis import run_hyperparameter_analysis


def main() -> None:
    """Run controlled learning-rate and dropout experiments."""
    print(json.dumps(run_hyperparameter_analysis(), indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
