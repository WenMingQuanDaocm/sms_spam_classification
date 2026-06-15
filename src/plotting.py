"""项目图像的通用绘图辅助函数。"""

from __future__ import annotations

import matplotlib.pyplot as plt


CLASS_LABEL_DISPLAY = {
    # 图中使用中文类别名，指标文件中仍保留原始 ham/spam 标签。
    "ham": "正常短信",
    "spam": "垃圾短信",
}

MODEL_LABEL_DISPLAY = {
    "majority_baseline": "多数类\n基线",
    "logistic_regression": "逻辑回归",
    "mlp": "MLP",
}


def configure_plot_style() -> None:
    """尽量使用支持中文的字体，并保证负号正常显示。"""
    # 依次尝试常见中文字体，保证 Windows 和其他环境尽量都能正常显示中文。
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
    """返回图中使用的类别显示名称。"""
    return CLASS_LABEL_DISPLAY.get(str(label), str(label))


def display_model_label(model_name: str) -> str:
    """返回图中使用的模型显示名称。"""
    return MODEL_LABEL_DISPLAY.get(str(model_name), str(model_name).replace("_", "\n"))
