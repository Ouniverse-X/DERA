"""Run the DERA research loop."""

from __future__ import annotations

import argparse
import os
from pathlib import Path

try:
    from .llm_client import DeepSeekClient
    from .search_loop import DERA
except ImportError:
    from llm_client import DeepSeekClient
    from search_loop import DERA


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--task", type=Path, required=True)
    parser.add_argument("--data-root", type=Path, required=True)
    parser.add_argument("--run-dir", type=Path, required=True)
    parser.add_argument("--rounds", type=int, default=3)
    parser.add_argument("--pilot-trials", type=int, default=3, help="Total evaluations per candidate, including the inherited/default anchor")
    parser.add_argument("--hpo-budget", type=int, default=None, help="Global budget for extra TPE trials only; one default evaluation is guaranteed per LLM research round")
    parser.add_argument("--evaluation-budget", type=int, default=None, help="Combined development budget for code evaluations and extra HPO trials; shared initial evaluation excluded")
    parser.add_argument("--max-hpo-trials-per-round", type=int, required=True, help="Hard upper bound on extra TPE trials in one research round; the LLM chooses the actual count from zero to this limit")
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--device", default="auto")
    parser.add_argument("--timeout", type=float, default=1200.0)
    parser.add_argument("--model", default=os.environ.get("DEEPSEEK_MODEL", "deepseek-v4-flash"))
    args = parser.parse_args()
    if args.max_hpo_trials_per_round < 1:
        parser.error("--max-hpo-trials-per-round must be positive")
    if args.evaluation_budget is not None and args.evaluation_budget < 1:
        parser.error("--evaluation-budget must be positive")
    client = DeepSeekClient(model=args.model, log_dir=args.run_dir / "llm_calls")
    result = DERA(args.task, args.data_root, args.run_dir, client, args.device, args.pilot_trials, args.timeout, args.hpo_budget, args.max_hpo_trials_per_round, args.evaluation_budget).run(args.rounds, args.seed)
    print(result["best"])


if __name__ == "__main__":
    main()
