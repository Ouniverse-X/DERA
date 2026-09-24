# DERA

This repository contains the implementation of the **Diagnostically Enhanced Research Agent (DERA)**.

DERA combines iterative code development with diagnostic experiments. At each iteration, the Code Agent modifies and evaluates the current best program. The Diagnostic Agent then decides whether to conduct HPO, designs and analyzes diagnostic trials, and provides its analysis to the next Code Agent. We use Optuna TPE to execute the diagnostic trials.

The repository includes the four tasks used in our experiments:

- WikiText-2 language modeling
- AG News text classification
- EasyFSL few-shot image classification
- PhonemeSpectra time-series classification

Each task includes its initial program and evaluator. Due to repository space constraints, `data/` contains compact development and test subsets for running the code. See [`data/README.md`](data/README.md) for the complete datasets and their original sources.

## Run

Install the dependencies:

```bash
python -m pip install -r requirements.txt
```

Set `DEEPSEEK_API_KEY`, then run:

```bash
python run.py \
  --task tasks/TASK \
  --data-root data \
  --run-dir runs/TASK-seed4 \
  --evaluation-budget 50 \
  --per-iteration-trial-limit 4 \
  --seed 4 \
  --device cuda:0
```

Each Code Agent evaluation and diagnostic trial counts toward the shared evaluation budget. The best program selected on the development set is evaluated once on the test set.
