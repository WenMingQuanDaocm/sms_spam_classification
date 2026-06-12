"""Report the local Python environment for reproducibility checks."""

from __future__ import annotations

import json
import platform
import sys
from importlib import metadata
from typing import Any


PACKAGE_DISTRIBUTIONS = {
    "numpy": "numpy",
    "pandas": "pandas",
    "matplotlib": "matplotlib",
    "scikit_learn": "scikit-learn",
    "torch": "torch",
    "tqdm": "tqdm",
    "joblib": "joblib",
}


def get_package_versions() -> dict[str, str | None]:
    """Return installed package versions without importing heavy packages."""
    versions: dict[str, str | None] = {}
    for name, distribution in PACKAGE_DISTRIBUTIONS.items():
        try:
            versions[name] = metadata.version(distribution)
        except metadata.PackageNotFoundError:
            versions[name] = None
    return versions


def get_torch_device_info() -> dict[str, Any]:
    """Return basic PyTorch device information when torch is installed."""
    try:
        import torch
    except ImportError:
        return {
            "torch_importable": False,
            "cuda_available": None,
            "device": "cpu",
        }

    cuda_available = torch.cuda.is_available()
    return {
        "torch_importable": True,
        "cuda_available": cuda_available,
        "device": "cuda" if cuda_available else "cpu",
    }


def main() -> None:
    """Print environment information as formatted JSON."""
    report = {
        "os": platform.platform(),
        "python_version": sys.version,
        "python_executable": sys.executable,
        "packages": get_package_versions(),
        "torch_device": get_torch_device_info(),
    }
    print(json.dumps(report, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
