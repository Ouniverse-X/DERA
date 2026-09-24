#!/usr/bin/env python
"""Initial FCN program for fixed PhonemeSpectra arrays."""

from __future__ import annotations

import argparse
import json
import random
import time
from pathlib import Path

import numpy as np
import torch
from torch import nn
from torch.utils.data import DataLoader, TensorDataset

SEED = 0
BATCH_SIZE = 128
EPOCHS = 50
LEARNING_RATE = 0.001


class TimeSeriesFCN(nn.Module):
    """Three-block fully convolutional network from Wang et al. (2017)."""

    def __init__(self):
        super().__init__()
        self.features = nn.Sequential(
            nn.Conv1d(11, 128, kernel_size=8, padding="same", bias=False),
            nn.BatchNorm1d(128),
            nn.ReLU(),
            nn.Conv1d(128, 256, kernel_size=5, padding="same", bias=False),
            nn.BatchNorm1d(256),
            nn.ReLU(),
            nn.Conv1d(256, 128, kernel_size=3, padding="same", bias=False),
            nn.BatchNorm1d(128),
            nn.ReLU(),
            nn.AdaptiveAvgPool1d(1),
        )
        self.classifier = nn.Linear(128, 39)

    def forward(self, values: torch.Tensor) -> torch.Tensor:
        return self.classifier(self.features(values).squeeze(-1))


def seed_everything() -> None:
    random.seed(SEED)
    np.random.seed(SEED)
    torch.manual_seed(SEED)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(SEED)
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True


def resolve_device(value: str) -> torch.device:
    if value == "auto":
        return torch.device("cuda" if torch.cuda.is_available() else "cpu")
    return torch.device(value)


def make_loader(values: np.ndarray, labels: np.ndarray, *, shuffle: bool) -> DataLoader:
    dataset = TensorDataset(torch.from_numpy(values), torch.from_numpy(labels))
    generator = torch.Generator().manual_seed(SEED)
    return DataLoader(
        dataset,
        batch_size=BATCH_SIZE,
        shuffle=shuffle,
        generator=generator if shuffle else None,
    )


def evaluate(
    model: nn.Module,
    values: np.ndarray,
    labels: np.ndarray,
    device: torch.device,
) -> tuple[float, float]:
    criterion = nn.CrossEntropyLoss(reduction="sum")
    correct = 0
    loss_sum = 0.0
    model.eval()
    with torch.inference_mode():
        for inputs, targets in make_loader(values, labels, shuffle=False):
            inputs = inputs.to(device)
            targets = targets.to(device)
            logits = model(inputs)
            loss_sum += criterion(logits, targets).item()
            correct += (logits.argmax(dim=1) == targets).sum().item()
    count = len(labels)
    return correct / count, loss_sum / count


def train_model(
    model: nn.Module,
    train_values: np.ndarray,
    train_labels: np.ndarray,
    validation_values: np.ndarray,
    validation_labels: np.ndarray,
    device: torch.device,
) -> None:
    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=LEARNING_RATE)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
        optimizer,
        mode="max",
        factor=0.5,
        patience=5,
    )
    train_loader = make_loader(train_values, train_labels, shuffle=True)
    best_accuracy = -1.0
    best_state: dict[str, torch.Tensor] | None = None
    for _ in range(EPOCHS):
        model.train()
        for inputs, targets in train_loader:
            inputs = inputs.to(device)
            targets = targets.to(device)
            optimizer.zero_grad(set_to_none=True)
            loss = criterion(model(inputs), targets)
            loss.backward()
            optimizer.step()
        validation_accuracy, _ = evaluate(
            model,
            validation_values,
            validation_labels,
            device,
        )
        scheduler.step(validation_accuracy)
        if validation_accuracy > best_accuracy:
            best_accuracy = validation_accuracy
            best_state = {
                name: value.detach().cpu().clone() for name, value in model.state_dict().items()
            }
    if best_state is not None:
        model.load_state_dict(best_state)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--data-dir", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--device", default="auto")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    started = time.monotonic()
    seed_everything()
    device = resolve_device(args.device)
    with np.load(args.data_dir / "data.npz", allow_pickle=False) as data:
        train_values = data["train_values"].astype(np.float32, copy=False)
        train_labels = data["train_labels"].astype(np.int64, copy=False)
        validation_values = data["validation_values"].astype(np.float32, copy=False)
        validation_labels = data["validation_labels"].astype(np.int64, copy=False)
        evaluation_values = data["evaluation_values"].astype(np.float32, copy=False)
        evaluation_labels = data["evaluation_labels"].astype(np.int64, copy=False)

    model = TimeSeriesFCN().to(device)
    train_model(
        model,
        train_values,
        train_labels,
        validation_values,
        validation_labels,
        device,
    )
    accuracy, cross_entropy = evaluate(
        model,
        evaluation_values,
        evaluation_labels,
        device,
    )
    payload = {
        "metrics": {
            "accuracy": accuracy,
            "cross_entropy": cross_entropy,
        },
        "runtime": {
            "elapsed_seconds": time.monotonic() - started,
            "device": str(device),
        },
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(
        json.dumps(payload, indent=2, sort_keys=True, allow_nan=False) + "\n",
        encoding="utf-8",
    )
    print(json.dumps(payload["metrics"], sort_keys=True))


if __name__ == "__main__":
    main()
