"""Run phase-five MLP training and validation."""

from __future__ import annotations

import json
import sys
from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.train_mlp import train_mlp_validation


def main() -> None:
    """Train the default MLP using train/validation splits only."""
    report = train_mlp_validation()
    report["note"] = "Metrics are validation-only. The test split is not used here."
    print(json.dumps(report, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
