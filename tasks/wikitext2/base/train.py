#!/usr/bin/env python
"""Initial causal Transformer program for fixed WikiText-2 token streams."""

from __future__ import annotations

import argparse
import json
import math
import random
import time
from pathlib import Path

import numpy as np
import torch
from torch import nn
from torch.nn import functional as F

SEED = 1111
EMBEDDING_DIMENSION = 64
ATTENTION_HEADS = 4
FEEDFORWARD_DIMENSION = 128
ENCODER_LAYERS = 2
DROPOUT = 0.2
SEQUENCE_LENGTH = 64
TRAIN_BATCH_SIZE = 64
EVALUATION_BATCH_SIZE = 64
EPOCHS = 3
LEARNING_RATE = 0.001
GRADIENT_CLIP = 1.0


def sinusoidal_positions(length: int, dimension: int) -> torch.Tensor:
    positions = torch.arange(length, dtype=torch.float32).unsqueeze(1)
    frequencies = torch.exp(
        torch.arange(0, dimension, 2, dtype=torch.float32) * (-math.log(10_000.0) / dimension)
    )
    encoding = torch.zeros(length, dimension, dtype=torch.float32)
    encoding[:, 0::2] = torch.sin(positions * frequencies)
    encoding[:, 1::2] = torch.cos(positions * frequencies)
    return encoding


class SmallCausalTransformer(nn.Module):
    """Two-layer word-level Transformer derived from the PyTorch example."""

    def __init__(self, vocabulary_size: int):
        super().__init__()
        self.embedding = nn.Embedding(vocabulary_size, EMBEDDING_DIMENSION)
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=EMBEDDING_DIMENSION,
            nhead=ATTENTION_HEADS,
            dim_feedforward=FEEDFORWARD_DIMENSION,
            dropout=DROPOUT,
            activation="relu",
            batch_first=True,
        )
        self.transformer = nn.TransformerEncoder(
            encoder_layer,
            num_layers=ENCODER_LAYERS,
        )
        self.decoder = nn.Linear(EMBEDDING_DIMENSION, vocabulary_size)
        self.decoder.weight = self.embedding.weight
        self.register_buffer(
            "position_encoding",
            sinusoidal_positions(SEQUENCE_LENGTH, EMBEDDING_DIMENSION),
            persistent=False,
        )
        self.register_buffer(
            "causal_mask",
            torch.triu(
                torch.full((SEQUENCE_LENGTH, SEQUENCE_LENGTH), float("-inf")),
                diagonal=1,
            ),
            persistent=False,
        )
        self.embedding.weight.data.uniform_(-0.1, 0.1)
        self.decoder.bias.data.zero_()

    def forward(self, token_ids: torch.Tensor) -> torch.Tensor:
        sequence_length = token_ids.shape[1]
        if sequence_length > SEQUENCE_LENGTH:
            raise ValueError("input exceeds the fixed sequence length")
        hidden = self.embedding(token_ids) * math.sqrt(EMBEDDING_DIMENSION)
        hidden = hidden + self.position_encoding[:sequence_length]
        hidden = self.transformer(
            hidden,
            mask=self.causal_mask[:sequence_length, :sequence_length],
        )
        return self.decoder(hidden)


def resolve_device(value: str) -> torch.device:
    if value == "auto":
        return torch.device("cuda" if torch.cuda.is_available() else "cpu")
    device = torch.device(value)
    if device.type == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("CUDA was requested but is unavailable")
    return device


def seed_everything() -> None:
    random.seed(SEED)
    np.random.seed(SEED)
    torch.manual_seed(SEED)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(SEED)
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True


def validate_token_stream(name: str, values: np.ndarray, vocabulary_size: int) -> None:
    if values.ndim != 1 or len(values) < 2 or values.dtype.kind not in {"i", "u"}:
        raise ValueError(f"{name} must be a non-empty one-dimensional integer array")
    if int(values.min()) < 0 or int(values.max()) >= vocabulary_size:
        raise ValueError(f"{name} contains a token outside the fixed vocabulary")


def batchify(values: np.ndarray, batch_size: int, device: torch.device) -> torch.Tensor:
    token_ids = torch.from_numpy(np.asarray(values, dtype=np.int64))
    usable = (len(token_ids) // batch_size) * batch_size
    if usable <= batch_size:
        raise ValueError("token stream is too short for the fixed batch size")
    return token_ids[:usable].view(batch_size, -1).to(device)


def batches(data: torch.Tensor):
    final_start = data.shape[1] - 1
    for start in range(0, final_start, SEQUENCE_LENGTH):
        length = min(SEQUENCE_LENGTH, final_start - start)
        yield (
            data[:, start : start + length],
            data[:, start + 1 : start + 1 + length],
        )


def evaluate(model: nn.Module, data: torch.Tensor) -> float:
    loss_sum = 0.0
    token_count = 0
    model.eval()
    with torch.inference_mode():
        for inputs, targets in batches(data):
            logits = model(inputs)
            loss_sum += F.cross_entropy(
                logits.reshape(-1, logits.shape[-1]),
                targets.reshape(-1),
                reduction="sum",
            ).item()
            token_count += targets.numel()
    if token_count == 0:
        raise RuntimeError("evaluation produced no target tokens")
    return loss_sum / token_count


def train_model(
    model: nn.Module,
    train_data: torch.Tensor,
    validation_data: torch.Tensor,
) -> tuple[list[float], int]:
    optimizer = torch.optim.AdamW(model.parameters(), lr=LEARNING_RATE)
    best_validation_loss = float("inf")
    best_epoch = 0
    validation_cross_entropy_by_epoch: list[float] = []
    best_state: dict[str, torch.Tensor] | None = None
    for epoch in range(1, EPOCHS + 1):
        model.train()
        for inputs, targets in batches(train_data):
            optimizer.zero_grad(set_to_none=True)
            logits = model(inputs)
            loss = F.cross_entropy(
                logits.reshape(-1, logits.shape[-1]),
                targets.reshape(-1),
            )
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), GRADIENT_CLIP)
            optimizer.step()
        validation_loss = evaluate(model, validation_data)
        validation_cross_entropy_by_epoch.append(validation_loss)
        if validation_loss < best_validation_loss:
            best_validation_loss = validation_loss
            best_epoch = epoch
            best_state = {
                name: value.detach().cpu().clone() for name, value in model.state_dict().items()
            }
    if best_state is None:
        raise RuntimeError("training did not produce a checkpoint")
    model.load_state_dict(best_state)
    return validation_cross_entropy_by_epoch, best_epoch


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
        validation_tokens = data["validation_tokens"]
        evaluation_tokens = data["evaluation_tokens"]
        vocabulary_size = int(data["vocabulary_size"])
    if vocabulary_size <= 1:
        raise ValueError("vocabulary_size must be greater than one")
    validate_token_stream("train_tokens", train_tokens, vocabulary_size)
    validate_token_stream("validation_tokens", validation_tokens, vocabulary_size)
    validate_token_stream("evaluation_tokens", evaluation_tokens, vocabulary_size)
    train_data = batchify(train_tokens, TRAIN_BATCH_SIZE, device)
    validation_data = batchify(validation_tokens, EVALUATION_BATCH_SIZE, device)
    evaluation_data = batchify(evaluation_tokens, EVALUATION_BATCH_SIZE, device)
    model = SmallCausalTransformer(vocabulary_size).to(device)
    validation_curve, best_epoch = train_model(model, train_data, validation_data)
    cross_entropy = evaluate(model, evaluation_data)
    perplexity = math.exp(cross_entropy)
    if not math.isfinite(cross_entropy) or not math.isfinite(perplexity):
        raise RuntimeError("language-model metrics are non-finite")
    args.output.parent.mkdir(parents=True, exist_ok=True)
    payload = {
        "metrics": {
            "cross_entropy": cross_entropy,
            "perplexity": perplexity,
        },
        "training": {
            "epochs_requested": EPOCHS,
            "best_epoch": best_epoch,
            "validation_cross_entropy_by_epoch": validation_curve,
        },
        "runtime": {
            "elapsed_seconds": time.monotonic() - started,
            "device": str(device),
            "parameter_count": sum(parameter.numel() for parameter in model.parameters()),
        },
    }
    args.output.write_text(
        json.dumps(payload, indent=2, sort_keys=True, allow_nan=False) + "\n",
        encoding="utf-8",
    )
    print(
        json.dumps(
            {"metrics": payload["metrics"], "training": payload["training"]},
            sort_keys=True,
        )
    )


if __name__ == "__main__":
    main()
