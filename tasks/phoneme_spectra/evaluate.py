#!/usr/bin/env python3
"""Evaluate one PhonemeSpectra program using a prepared data split."""

from __future__ import annotations

import argparse
import subprocess
import sys
from pathlib import Path


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--program-dir", type=Path, required=True)
    parser.add_argument("--data-dir", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--device", default="auto")
    args = parser.parse_args()
    command = [sys.executable, "train.py", "--data-dir", str(args.data_dir.resolve()), "--output", str(args.output.resolve()), "--device", args.device]
    raise SystemExit(subprocess.run(command, cwd=args.program_dir.resolve(), check=False).returncode)


if __name__ == "__main__":
    main()
