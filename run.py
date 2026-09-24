"""Run the DERA research loop."""

from __future__ import annotations

import argparse
import os
from pathlib import Path

from llm_client import DeepSeekClient
from search_loop import DERA


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--task", type=Path, required=True)
    parser.add_argument("--data-root", type=Path, required=True)
    parser.add_argument("--run-dir", type=Path, required=True)
    parser.add_argument("--evaluation-budget", type=int, default=50, help="Total number of Code Agent evaluations and diagnostic trials")
    parser.add_argument("--per-iteration-trial-limit", type=int, default=4, help="Maximum number of diagnostic trials after one code iteration")
    parser.add_argument("--seed", type=int, default=4)
    parser.add_argument("--device", default="auto")
    parser.add_argument("--timeout", type=float, default=1200.0)
    parser.add_argument("--model", default=os.environ.get("DEEPSEEK_MODEL", "deepseek-v4-flash"))
    args = parser.parse_args()
    if args.evaluation_budget < 1:
        parser.error("--evaluation-budget must be positive")
    if args.per_iteration_trial_limit < 1:
        parser.error("--per-iteration-trial-limit must be positive")
    client = DeepSeekClient(model=args.model, log_dir=args.run_dir / "llm_calls")
    result = DERA(
        args.task,
        args.data_root,
        args.run_dir,
        client,
        args.device,
        args.timeout,
        args.evaluation_budget,
        args.per_iteration_trial_limit,
    ).run(args.seed)
    print(result["best"])


if __name__ == "__main__":
    main()
