#!/usr/bin/env python3
"""In-process evaluator for fixed EasyFSL feature episodes."""

from __future__ import annotations

import argparse
import json
import random
import sys
import time
from pathlib import Path

import numpy as np
import torch
from torch import nn


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--program-dir", type=Path, required=True)
    parser.add_argument("--data-dir", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--device", default="auto")
    args = parser.parse_args()
    started = time.monotonic()
    random.seed(0)
    np.random.seed(0)
    torch.manual_seed(0)
    device = torch.device("cuda" if args.device == "auto" and torch.cuda.is_available() else ("cpu" if args.device == "auto" else args.device))
    sys.path.insert(0, str(args.program_dir.resolve()))
    from easyfsl.methods import PrototypicalNetworks

    features = np.load(args.data_dir / "features.npy", mmap_mode="r", allow_pickle=False)
    with np.load(args.data_dir / "episodes.npz", allow_pickle=False) as episodes:
        support_indices = episodes["support_indices"]
        support_labels = episodes["support_labels"]
        query_indices = episodes["query_indices"]
        query_labels = episodes["query_labels"]
    model = PrototypicalNetworks(backbone=nn.Identity(), feature_normalization=2).to(device)
    model.eval()
    accuracies = []
    with torch.inference_mode():
        for index in range(len(support_indices)):
            support = torch.from_numpy(np.asarray(features[support_indices[index]])).to(device)
            labels = torch.from_numpy(support_labels[index]).to(device)
            query = torch.from_numpy(np.asarray(features[query_indices[index]])).to(device)
            model.process_support_set(support, labels)
            prediction = model(query).argmax(dim=1).cpu().numpy()
            accuracies.append(float(np.mean(prediction == query_labels[index])))
    values = np.asarray(accuracies, dtype=np.float64)
    payload = {
        "metrics": {
            "accuracy": float(values.mean()),
            "accuracy_stderr": float(values.std(ddof=1) / np.sqrt(len(values))),
        },
        "runtime": {"elapsed_seconds": time.monotonic() - started, "device": str(device)},
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    print(json.dumps(payload["metrics"], sort_keys=True))


if __name__ == "__main__":
    main()
