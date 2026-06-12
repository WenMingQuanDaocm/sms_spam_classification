"""Project entry point placeholder.

The full experiment pipeline is intentionally implemented in later stages.
"""

from src.config import RAW_DATA_PATH


def main() -> None:
    """Print the expected dataset path without running experiments."""
    print("SMS spam classification project initialized.")
    print(f"Expected raw dataset path: {RAW_DATA_PATH}")
    print("No data loading, training, or evaluation is run in this stage.")


if __name__ == "__main__":
    main()
