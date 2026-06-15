"""项目入口占位脚本。

完整实验流程由后续阶段的脚本实现，这里不直接运行训练或评估。
"""

from src.config import RAW_DATA_PATH


def main() -> None:
    """打印预期数据路径，但不运行任何实验。"""
    # 主入口只提示项目状态，避免双击或误运行时直接开始训练实验。
    print("SMS spam classification project initialized.")
    print(f"Expected raw dataset path: {RAW_DATA_PATH}")
    print("No data loading, training, or evaluation is run in this stage.")


if __name__ == "__main__":
    main()
