#!/usr/bin/env python
"""EmbeddingBag baseline for fixed AG News token arrays."""

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
EMBEDDING_DIMENSION = 64
BATCH_SIZE = 256
EPOCHS = 5
LEARNING_RATE = 5.0
CLASS_COUNT = 4


class TextClassificationModel(nn.Module):
    """EmbeddingBag plus linear classifier from the PyTorch AG News tutorial."""

    def __init__(self, vocabulary_size: int):
        super().__init__()
        self.embedding = nn.EmbeddingBag(
            vocabulary_size,
            EMBEDDING_DIMENSION,
            mode="mean",
            padding_idx=0,
        )
        self.classifier = nn.Linear(EMBEDDING_DIMENSION, CLASS_COUNT)
        self.embedding.weight.data.uniform_(-0.5, 0.5)
        self.classifier.weight.data.uniform_(-0.5, 0.5)
        self.classifier.bias.data.zero_()

    def forward(self, token_ids: torch.Tensor) -> torch.Tensor:
        batch_size, sequence_length = token_ids.shape
        offsets = torch.arange(
            0,
            batch_size * sequence_length,
            sequence_length,
            device=token_ids.device,
        )
        embedded = self.embedding(token_ids.reshape(-1), offsets)
        return self.classifier(embedded)


def resolve_device(value: str) -> torch.device:
    if value == "auto":
        return torch.device("cuda" if torch.cuda.is_available() else "cpu")
    return torch.device(value)


def seed_everything() -> None:
    random.seed(SEED)
    np.random.seed(SEED)
    torch.manual_seed(SEED)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(SEED)


def make_loader(
    tokens: np.ndarray,
    labels: np.ndarray,
    *,
    shuffle: bool,
) -> DataLoader[tuple[torch.Tensor, torch.Tensor]]:
    dataset = TensorDataset(
        torch.from_numpy(np.asarray(tokens, dtype=np.int64)),
        torch.from_numpy(np.asarray(labels, dtype=np.int64)),
    )
    generator = torch.Generator().manual_seed(SEED)
    return DataLoader(
        dataset,
        batch_size=BATCH_SIZE,
        shuffle=shuffle,
        generator=generator if shuffle else None,
    )


def train_model(
    model: nn.Module,
    tokens: np.ndarray,
    labels: np.ndarray,
    device: torch.device,
) -> None:
    loader = make_loader(tokens, labels, shuffle=True)
    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.SGD(model.parameters(), lr=LEARNING_RATE)
    scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=1, gamma=0.5)
    for _ in range(EPOCHS):
        model.train()
        for token_ids, targets in loader:
            token_ids = token_ids.to(device)
            targets = targets.to(device)
            optimizer.zero_grad(set_to_none=True)
            logits = model(token_ids)
            loss = criterion(logits, targets)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 0.1)
            optimizer.step()
        scheduler.step()


def evaluate(
    model: nn.Module,
    tokens: np.ndarray,
    labels: np.ndarray,
    device: torch.device,
) -> tuple[float, float]:
    loader = make_loader(tokens, labels, shuffle=False)
    criterion = nn.CrossEntropyLoss(reduction="sum")
    correct = 0
    total = 0
    loss_sum = 0.0
    model.eval()
    with torch.inference_mode():
        for token_ids, targets in loader:
            token_ids = token_ids.to(device)
            targets = targets.to(device)
            logits = model(token_ids)
            loss_sum += criterion(logits, targets).item()
            correct += (logits.argmax(dim=1) == targets).sum().item()
            total += targets.numel()
    return correct / total, loss_sum / total


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
        train_tokens = data["train_tokens"]
        train_labels = data["train_labels"]
        evaluation_tokens = data["evaluation_tokens"]
        evaluation_labels = data["evaluation_labels"]
        vocabulary_size = int(data["vocabulary_size"])
    model = TextClassificationModel(vocabulary_size).to(device)
    train_model(model, train_tokens, train_labels, device)
    accuracy, cross_entropy = evaluate(
        model,
        evaluation_tokens,
        evaluation_labels,
        device,
    )
    args.output.parent.mkdir(parents=True, exist_ok=True)
    payload = {
        "metrics": {"accuracy": accuracy, "cross_entropy": cross_entropy},
        "runtime": {
            "elapsed_seconds": time.monotonic() - started,
            "device": str(device),
        },
    }
    args.output.write_text(
        json.dumps(payload, indent=2, sort_keys=True, allow_nan=False) + "\n",
        encoding="utf-8",
    )
    print(json.dumps(payload["metrics"], sort_keys=True))


if __name__ == "__main__":
    main()
