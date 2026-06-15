"""输出本地 Python 环境信息，用于复现性检查。"""

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
    """在不导入大型依赖的情况下返回已安装包版本。"""
    versions: dict[str, str | None] = {}
    for name, distribution in PACKAGE_DISTRIBUTIONS.items():
        try:
            # 使用 metadata 查询版本，避免导入大型库带来的额外启动开销。
            versions[name] = metadata.version(distribution)
        except metadata.PackageNotFoundError:
            versions[name] = None
    return versions


def get_torch_device_info() -> dict[str, Any]:
    """当 PyTorch 已安装时返回基础设备信息。"""
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
    """以格式化 JSON 形式打印环境信息。"""
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
