"""Train and validate the PyTorch MLP on TF-IDF features."""

from __future__ import annotations

import os
import time
from pathlib import Path
from typing import Any

import joblib

from src.config import MATPLOTLIB_CACHE_DIR

os.environ.setdefault("MPLCONFIGDIR", str(MATPLOTLIB_CACHE_DIR))

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import pandas as pd
import torch
from sklearn.metrics import f1_score
from torch import nn
from torch.utils.data import DataLoader, TensorDataset

from src.config import (
    MESSAGE_COLUMN,
    MLP_CHECKPOINT_PATH,
    MLP_CONFIG,
    MLP_TRAINING_HISTORY_PATH,
    MLP_VALIDATION_METRICS_PATH,
    RANDOM_STATE,
    TARGET_COLUMN,
    TFIDF_VECTORIZER_PATH,
    TRAIN_DATA_PATH,
    TRAINING_FIGURES_DIR,
    VALIDATION_DATA_PATH,
)
from src.evaluate import evaluate_predictions, save_metrics
from src.models import TextMLP
from src.plotting import configure_plot_style
from src.train_logistic import load_tfidf_vectorizer
from src.utils import get_training_device, set_random_seed


configure_plot_style()


def sparse_to_float_tensor(matrix: Any) -> torch.Tensor:
    """Convert a sparse TF-IDF matrix to a float32 tensor for the MLP."""
    return torch.from_numpy(matrix.toarray()).float()


def build_dataloader(
    features: torch.Tensor,
    targets: pd.Series,
    batch_size: int,
    shuffle: bool,
    seed: int = RANDOM_STATE,
) -> DataLoader:
    """Build a deterministic DataLoader from dense features and labels."""
    labels = torch.tensor(targets.to_numpy(), dtype=torch.long)
    dataset = TensorDataset(features, labels)
    generator = torch.Generator()
    generator.manual_seed(seed)
    return DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=shuffle,
        generator=generator,
    )


def load_mlp_data(
    train_path: str | Path = TRAIN_DATA_PATH,
    validation_path: str | Path = VALIDATION_DATA_PATH,
    vectorizer_path: str | Path = TFIDF_VECTORIZER_PATH,
) -> tuple[pd.DataFrame, pd.DataFrame, Any, torch.Tensor, torch.Tensor]:
    """Load train/validation splits and transform them with the fitted vectorizer."""
    train_data = pd.read_csv(train_path, encoding="utf-8")
    validation_data = pd.read_csv(validation_path, encoding="utf-8")
    vectorizer = load_tfidf_vectorizer(vectorizer_path)

    train_features = sparse_to_float_tensor(vectorizer.transform(train_data[MESSAGE_COLUMN]))
    validation_features = sparse_to_float_tensor(
        vectorizer.transform(validation_data[MESSAGE_COLUMN])
    )
    return train_data, validation_data, vectorizer, train_features, validation_features


def train_one_epoch(
    model: nn.Module,
    dataloader: DataLoader,
    criterion: nn.Module,
    optimizer: torch.optim.Optimizer,
    device: torch.device,
) -> dict[str, float]:
    """Run one training epoch and return loss/accuracy."""
    model.train()
    total_loss = 0.0
    total_correct = 0
    total_samples = 0

    for features, targets in dataloader:
        features = features.to(device)
        targets = targets.to(device)

        optimizer.zero_grad()
        logits = model(features)
        loss = criterion(logits, targets)
        loss.backward()
        optimizer.step()

        batch_size = targets.size(0)
        total_loss += float(loss.item()) * batch_size
        total_correct += int((logits.argmax(dim=1) == targets).sum().item())
        total_samples += batch_size

    return {
        "loss": total_loss / total_samples,
        "accuracy": total_correct / total_samples,
    }


def evaluate_mlp_epoch(
    model: nn.Module,
    dataloader: DataLoader,
    criterion: nn.Module,
    device: torch.device,
) -> dict[str, float | list[int]]:
    """Evaluate the MLP on one split."""
    model.eval()
    total_loss = 0.0
    total_correct = 0
    total_samples = 0
    predictions: list[int] = []
    targets_all: list[int] = []

    with torch.no_grad():
        for features, targets in dataloader:
            features = features.to(device)
            targets = targets.to(device)
            logits = model(features)
            loss = criterion(logits, targets)
            predicted = logits.argmax(dim=1)

            batch_size = targets.size(0)
            total_loss += float(loss.item()) * batch_size
            total_correct += int((predicted == targets).sum().item())
            total_samples += batch_size
            predictions.extend(predicted.cpu().tolist())
            targets_all.extend(targets.cpu().tolist())

    return {
        "loss": total_loss / total_samples,
        "accuracy": total_correct / total_samples,
        "macro_f1": float(f1_score(targets_all, predictions, average="macro")),
        "predictions": predictions,
        "targets": targets_all,
    }


def save_checkpoint(
    model: TextMLP,
    path: str | Path,
    input_dim: int,
    epoch: int,
    validation_macro_f1: float,
    config: dict[str, Any],
) -> Path:
    """Save the best MLP checkpoint."""
    output_path = Path(path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    torch.save(
        {
            "model_state_dict": model.state_dict(),
            "input_dim": input_dim,
            "epoch": epoch,
            "validation_macro_f1": validation_macro_f1,
            "config": config,
        },
        output_path,
    )
    return output_path


def save_training_history(history: list[dict[str, Any]], path: str | Path) -> Path:
    """Save per-epoch MLP training history."""
    output_path = Path(path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    pd.DataFrame(history).to_csv(output_path, index=False, encoding="utf-8")
    return output_path


def plot_training_curves(
    history: list[dict[str, Any]],
    output_path: str | Path = TRAINING_FIGURES_DIR / "mlp_training_curves.png",
    best_epoch: int | None = None,
    show_titles: bool = True,
) -> Path:
    """Plot MLP train/validation loss and accuracy curves."""
    output = Path(output_path)
    output.parent.mkdir(parents=True, exist_ok=True)
    history_frame = pd.DataFrame(history)

    fig, axes = plt.subplots(1, 2, figsize=(10, 4))
    axes[0].plot(history_frame["epoch"], history_frame["train_loss"], label="训练集")
    axes[0].plot(
        history_frame["epoch"],
        history_frame["validation_loss"],
        label="验证集",
    )
    if show_titles:
        axes[0].set_title("MLP训练与验证损失")
    axes[0].set_xlabel("训练轮次（Epoch）")
    axes[0].set_ylabel("损失值（Loss）")

    axes[1].plot(history_frame["epoch"], history_frame["train_accuracy"], label="训练集")
    axes[1].plot(
        history_frame["epoch"],
        history_frame["validation_accuracy"],
        label="验证集",
    )
    if show_titles:
        axes[1].set_title("MLP训练与验证准确率")
    axes[1].set_xlabel("训练轮次（Epoch）")
    axes[1].set_ylabel("准确率（Accuracy）")

    if best_epoch is not None:
        for axis in axes:
            axis.axvline(
                best_epoch,
                color="#D62728",
                linestyle="--",
                linewidth=1.2,
                label=f"Best Epoch={best_epoch}",
            )

    for axis in axes:
        axis.legend()

    fig.tight_layout()
    fig.savefig(output, dpi=150)
    plt.close(fig)
    return output


def train_mlp_validation(
    train_path: str | Path = TRAIN_DATA_PATH,
    validation_path: str | Path = VALIDATION_DATA_PATH,
    vectorizer_path: str | Path = TFIDF_VECTORIZER_PATH,
    checkpoint_path: str | Path = MLP_CHECKPOINT_PATH,
    history_path: str | Path = MLP_TRAINING_HISTORY_PATH,
    metrics_path: str | Path = MLP_VALIDATION_METRICS_PATH,
    training_curves_path: str | Path = TRAINING_FIGURES_DIR / "mlp_training_curves.png",
    config: dict[str, Any] | None = None,
) -> dict[str, Any]:
    """Train the default MLP and evaluate the best checkpoint on validation."""
    run_config = dict(MLP_CONFIG if config is None else config)
    set_random_seed(RANDOM_STATE)
    device = get_training_device()

    train_data, validation_data, vectorizer, train_features, validation_features = load_mlp_data(
        train_path,
        validation_path,
        vectorizer_path,
    )
    input_dim = int(train_features.shape[1])
    train_loader = build_dataloader(
        train_features,
        train_data[TARGET_COLUMN],
        batch_size=int(run_config["batch_size"]),
        shuffle=True,
    )
    validation_loader = build_dataloader(
        validation_features,
        validation_data[TARGET_COLUMN],
        batch_size=int(run_config["batch_size"]),
        shuffle=False,
    )

    model = TextMLP(
        input_dim=input_dim,
        hidden_layers=tuple(run_config["hidden_layers"]),
        dropout=float(run_config["dropout"]),
    ).to(device)
    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=float(run_config["learning_rate"]),
        weight_decay=float(run_config["weight_decay"]),
    )

    best_validation_macro_f1 = -1.0
    best_epoch = 0
    epochs_without_improvement = 0
    history: list[dict[str, Any]] = []
    start_time = time.perf_counter()
    best_predictions: list[int] = []

    for epoch in range(1, int(run_config["max_epochs"]) + 1):
        train_metrics = train_one_epoch(model, train_loader, criterion, optimizer, device)
        validation_metrics = evaluate_mlp_epoch(model, validation_loader, criterion, device)
        validation_macro_f1 = float(validation_metrics["macro_f1"])
        is_best = validation_macro_f1 > best_validation_macro_f1

        if is_best:
            best_validation_macro_f1 = validation_macro_f1
            best_epoch = epoch
            epochs_without_improvement = 0
            best_predictions = list(validation_metrics["predictions"])
            save_checkpoint(
                model,
                checkpoint_path,
                input_dim=input_dim,
                epoch=epoch,
                validation_macro_f1=validation_macro_f1,
                config=run_config,
            )
        else:
            epochs_without_improvement += 1

        history.append(
            {
                "epoch": epoch,
                "train_loss": train_metrics["loss"],
                "validation_loss": validation_metrics["loss"],
                "train_accuracy": train_metrics["accuracy"],
                "validation_accuracy": validation_metrics["accuracy"],
                "validation_macro_f1": validation_macro_f1,
                "learning_rate": float(run_config["learning_rate"]),
                "saved_best_model": bool(is_best),
            }
        )

        if epochs_without_improvement >= int(run_config["early_stopping_patience"]):
            break

    training_time_seconds = time.perf_counter() - start_time
    history_output = save_training_history(history, history_path)
    training_curves_output = Path(training_curves_path)
    is_main_training_curve = (
        training_curves_output.parent == TRAINING_FIGURES_DIR
        and training_curves_output.name == "mlp_training_curves.png"
    )
    curves_output = plot_training_curves(
        history,
        training_curves_path,
        best_epoch=best_epoch if is_main_training_curve else None,
        show_titles=is_main_training_curve,
    )

    metrics = evaluate_predictions(
        validation_data[TARGET_COLUMN],
        best_predictions,
        model_name="mlp",
        split_name="validation",
    )
    metrics["training_time_seconds"] = float(training_time_seconds)
    metrics["best_epoch"] = int(best_epoch)
    metrics["epochs_ran"] = int(len(history))
    metrics["model_config"] = run_config
    metrics["device"] = str(device)
    metrics["checkpoint_path"] = str(checkpoint_path)
    metrics["history_path"] = str(history_output)
    metrics["training_curves_path"] = str(curves_output)
    metrics["vectorizer_path"] = str(vectorizer_path)
    metrics["tfidf_vocabulary_size"] = int(len(vectorizer.vocabulary_))
    save_metrics(metrics, metrics_path)
    return metrics


if __name__ == "__main__":
    import json

    print(json.dumps(train_mlp_validation(), indent=2, ensure_ascii=False))
