#!/usr/bin/env python3
"""Evaluate one WikiText-2 program using a prepared data split."""

from __future__ import annotations

import argparse
import json
import math
import subprocess
import sys
from pathlib import Path


MIN_PLAUSIBLE_CROSS_ENTROPY = 1.0


def validate_program_source(program_dir: Path) -> None:
    """Validate the data interface and causal language-model computation."""
    train_path = program_dir / "train.py"
    source = train_path.read_text(encoding="utf-8")
    compact = "".join(source.split())
    required = ("data.npz", "train_tokens", "validation_tokens", "evaluation_tokens")
    missing = [marker for marker in required if marker not in source]
    if missing:
        raise ValueError(
            "program must consume the prepared data.npz interface; "
            f"missing markers: {missing}"
        )
    forbidden = {
        "os.walk(": "recursive filesystem search",
        ".rglob(": "recursive filesystem search",
        ".parents": "parent-directory traversal",
        "v[:,1:": "future-token value leakage",
        "v[...,1:": "future-token value leakage",
    }
    violations = [reason for pattern, reason in forbidden.items() if pattern in compact]
    if violations:
        raise ValueError("invalid WikiText-2 program: " + ", ".join(sorted(set(violations))))


def validate_result(output: Path) -> None:
    payload = json.loads(output.read_text(encoding="utf-8"))
    value = float(payload["metrics"]["cross_entropy"])
    if not math.isfinite(value) or value < MIN_PLAUSIBLE_CROSS_ENTROPY:
        raise ValueError(f"invalid cross_entropy={value}")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--program-dir", type=Path, required=True)
    parser.add_argument("--data-dir", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--device", default="auto")
    args = parser.parse_args()
    try:
        validate_program_source(args.program_dir.resolve())
    except (OSError, UnicodeError, ValueError) as exc:
        print(f"preflight validation failed: {exc}", file=sys.stderr)
        raise SystemExit(2)
    command = [sys.executable, "train.py", "--data-dir", str(args.data_dir.resolve()), "--output", str(args.output.resolve()), "--device", args.device]
    completed = subprocess.run(command, cwd=args.program_dir.resolve(), check=False)
    if completed.returncode != 0:
        raise SystemExit(completed.returncode)
    try:
        validate_result(args.output.resolve())
    except (OSError, json.JSONDecodeError, KeyError, TypeError, ValueError) as exc:
        print(f"result validation failed: {exc}", file=sys.stderr)
        args.output.unlink(missing_ok=True)
        raise SystemExit(3)


if __name__ == "__main__":
    main()
