# DERA

The Diagnostically Enhanced Research Agent (DERA) combines LLM-guided code
development with candidate-specific hyperparameter optimization. This
repository contains the method's core implementation only.

## Core logic

DERA edits a candidate program, evaluates it on development data, and uses
optional HPO trials to diagnose the candidate's behavior. The measured result
and diagnostic feedback inform the next code edit. A controller maintains the
best development-measured program and configuration. Test evaluation is
performed only after selection is complete.

The implementation is organized as follows:

- `run.py`: command-line entry point;
- `search_loop.py`: candidate generation, feedback, selection, and budgets;
- `hpo_runner.py`: evaluation and configuration freezing;
- `space_validator.py`: validation of proposed search spaces;
- `llm_client.py`: OpenAI-compatible model calls and response logging;
- `schema.py`: search-space and trial records;
- `prompts/`: agent instructions and response contracts.

## Interface

The caller supplies a task directory containing `task.json`, a `base/`
initial solution, and the evaluator named by `task.json`. The task definition
specifies editable files, the objective, data subdirectory, and evaluator
command. The evaluator writes a JSON result containing the objective metric.
DERA reads development and test splits from the supplied data root; it does
not package or modify task data.

Install the core dependencies with `pip install -r requirements.txt` and
configure an API key with `DEEPSEEK_API_KEY` or `OPENAI_API_KEY`. Additional
dependencies are determined by the caller's task evaluator. Run
`python run.py --help` for the available inputs and budget controls.

Each run writes candidate code, LLM responses, per-round development records,
the selected result, and one final test result to its run directory.
