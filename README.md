# DERA

Official core implementation of the **Diagnostically Enhanced Research Agent
(DERA)**, described in *Enhancing Automated Research Agents with Diagnostic
Experiments*.

DERA combines iterative code development by a language-model agent with
candidate-specific hyperparameter optimization. Diagnostic evaluations help
distinguish limitations of a proposed program from limitations of its current
configuration, and their evidence informs the next code change. Candidate
selection uses development results; the selected program and configuration
are frozen before a final test evaluation.

This is a **method-only release**. It contains no benchmark tasks, datasets,
checkpoints, baselines, test suite, or experimental result artifacts. The
repository alone is therefore not a complete reproduction package for the
paper's reported results.

## Implementation

| Path | Role |
| --- | --- |
| `run.py` | Command-line entry point and run configuration |
| `search_loop.py` | Code-editing loop, feedback, selection, and budget accounting |
| `hpo_runner.py` | Development evaluations, Optuna trials, and configuration freezing |
| `space_validator.py` | Validation of proposed hyperparameter search spaces |
| `llm_client.py` | OpenAI-compatible model client and call records |
| `schema.py` | Search-space and evaluation records |
| `prompts/` | Agent instructions and structured-response contracts |

## Requirements

Use Python 3.10 or newer and install the core dependencies:

```bash
python -m pip install -r requirements.txt
```

Set `DEEPSEEK_API_KEY` or `OPENAI_API_KEY` in the environment. An alternative
OpenAI-compatible endpoint can be supplied with `DEEPSEEK_BASE_URL`. The
`openai` package is used as an API client; installing it does not prescribe
which model to use. Dependencies required by a task evaluator are separate
from this repository's core dependencies.

## Task contract

DERA operates on a caller-supplied task directory. The directory must contain
a `task.json` definition, a `base/` initial program, and the evaluator named
by `task.json`. The definition specifies:

- `task_id` and `data_subdir`;
- `editable_files`, relative to the candidate program;
- `objective.metric` and `objective.direction` (`minimize` or `maximize`);
- `evaluation_command`, with placeholders for the Python executable,
  candidate directory, data directory, output path, and device.

The evaluator must write JSON containing `metrics[objective.metric]`. DERA
reads task data from `<data-root>/<data_subdir>/development` during search
and from `<data-root>/<data_subdir>/test` only for final evaluation. It does
not modify the evaluator or the data.

## Running and outputs

Run `python run.py --help` for the complete command-line interface. A generic
invocation is:

```text
python run.py --task TASK_DIR --data-root DATA_ROOT --run-dir RUN_DIR \
  --rounds N --hpo-budget B --max-hpo-trials-per-round M
```

Replace the uppercase placeholders with paths and budgets appropriate to the
task. The optional `--evaluation-budget` caps the combined number of code
evaluations and additional HPO trials. The initial-program evaluation and
final test evaluation are tracked separately from that development budget.

The run directory stores generated solutions, LLM call records, per-round
development evaluations, `summary.json`, and `final_test/result.json`. A run
can be resumed with the same settings after the previous process has stopped;
completed rounds are recovered from their records.

## Citation

If you use DERA in research, please cite *Enhancing Automated Research Agents
with Diagnostic Experiments*. Full bibliographic information will be added
when the paper is available.
