"""Shared plotting helpers for project figures."""

from __future__ import annotations

import matplotlib.pyplot as plt


CLASS_LABEL_DISPLAY = {
    "ham": "正常短信",
    "spam": "垃圾短信",
}

MODEL_LABEL_DISPLAY = {
    "majority_baseline": "多数类\n基线",
    "logistic_regression": "逻辑回归",
    "mlp": "MLP",
}


def configure_plot_style() -> None:
    """Use Chinese-capable fonts when available and keep minus signs readable."""
    plt.rcParams["font.family"] = "sans-serif"
    plt.rcParams["font.sans-serif"] = [
        "Microsoft YaHei",
        "SimHei",
        "Noto Sans CJK SC",
        "Arial Unicode MS",
        "DejaVu Sans",
    ]
    plt.rcParams["axes.unicode_minus"] = False


def display_class_label(label: str) -> str:
    """Return the display label used in plots."""
    return CLASS_LABEL_DISPLAY.get(str(label), str(label))


def display_model_label(model_name: str) -> str:
    """Return the display model name used in plots."""
    return MODEL_LABEL_DISPLAY.get(str(model_name), str(model_name).replace("_", "\n"))
