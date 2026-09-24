"""DERA research loop."""

from __future__ import annotations

import json
import shutil
from pathlib import Path
from typing import Any

from hpo_runner import DiagnosticRunner
from llm_client import DeepSeekClient
from schema import SearchSpace
from space_validator import validate_space


ROOT = Path(__file__).resolve().parent


def _prompt(name: str) -> str:
    return (ROOT / "prompts" / name).read_text(encoding="utf-8")


def _files(directory: Path, editable: list[str]) -> dict[str, str]:
    return {name: (directory / name).read_text(encoding="utf-8") for name in editable}


def _better(direction: str, left: float | None, right: float | None) -> bool:
    if left is None: return False
    if right is None: return True
    return left < right if direction == "minimize" else left > right


def _normalize_edit(edit: dict[str, Any], editable: list[str]) -> dict[str, str]:
    raw = edit.get("files")
    if not isinstance(raw, dict):
        return {}
    allowed = set(editable)
    changed = {
        name: content for name, content in raw.items()
        if name in allowed and isinstance(content, str) and content
    }
    return changed


def _objective_from_result(path: Path, metric: str) -> float | None:
    if not path.is_file():
        return None
    payload = json.loads(path.read_text(encoding="utf-8"))
    return float(payload["metrics"][metric])


class DERA:
    def __init__(self, task_dir: Path, data_root: Path, run_dir: Path, client: DeepSeekClient, device: str = "auto", timeout: float = 1200.0, evaluation_budget: int = 50, per_iteration_trial_limit: int = 4) -> None:
        self.task_dir = task_dir.resolve()
        self.data_root = data_root.expanduser().resolve()
        self.run_dir = run_dir.resolve()
        self.client = client
        self.device = device
        self.evaluation_budget = evaluation_budget
        self.per_iteration_trial_limit = per_iteration_trial_limit
        self.task = json.loads((self.task_dir / "task.json").read_text(encoding="utf-8"))
        self.runner = DiagnosticRunner(self.task_dir, self.data_root, device, timeout)

    def run(self, seed: int = 4) -> dict[str, Any]:
        allowed_initial = {"llm_calls", "programs", "iterations", "initial_development"}
        existing = set(item.name for item in self.run_dir.iterdir()) if self.run_dir.exists() else set()
        if existing - allowed_initial:
            raise ValueError("run directory must be empty or new")
        self.run_dir.mkdir(parents=True, exist_ok=True)
        programs = self.run_dir / "programs"
        programs.mkdir(exist_ok=True)
        initial = programs / "p0000"
        if not initial.exists():
            shutil.copytree(self.task_dir / "base", initial)
        metric = self.task["objective"]["metric"]
        initial_result = self.run_dir / "initial_development" / "result.json"
        initial_score = _objective_from_result(initial_result, metric)
        if initial_score is None:
            initial_eval = self.runner.evaluate(
                initial, SearchSpace(), {}, "development", self.run_dir / "initial_development"
            )
            initial_score = initial_eval.objective
        best_record: dict[str, Any] | None = None
        if initial_score is not None:
            best_record = {
                "program_id": "p0000", "score": initial_score, "params": {},
                "iteration": 0, "evaluation_id": "initial_development",
            }
        current = initial
        history: list[dict[str, Any]] = []
        existing_iterations = sorted((self.run_dir / "iterations").glob("iteration-*")) if (self.run_dir / "iterations").exists() else []
        if existing_iterations:
            for folder in existing_iterations:
                record_file = folder / "record.json"
                if record_file.is_file():
                    record = json.loads(record_file.read_text(encoding="utf-8"))
                    history.append(record)
            start_iteration = len(history) + 1
        else:
            start_iteration = 1
        direction = self.task["objective"]["direction"]
        for record in history:
            score = record.get("program_best_objective", record.get("initial_objective"))
            if score is not None and (best_record is None or _better(direction, score, best_record["score"])):
                best_record = {
                    "program_id": record["program_id"], "score": score,
                    "params": record.get("program_best_params", record.get("diagnostic_summary", {}).get("best_params", {})),
                    "iteration": record["iteration"],
                    "evaluation_id": record.get("program_best_evaluation_id", f"iteration-{record['iteration']:04d}/initial"),
                }
        if best_record is not None:
            current = programs / best_record["program_id"]
        diagnostic_trials_used = sum(int(record.get("diagnostic_trials_used", 0)) for record in history)

        iteration_id = start_iteration
        while True:
            evaluations_used = len(history) + diagnostic_trials_used
            if evaluations_used >= self.evaluation_budget:
                break
            current_trial_limit = min(
                self.per_iteration_trial_limit,
                self.evaluation_budget - evaluations_used - 1,
            )
            base_program_id = current.name
            before = _files(current, self.task["editable_files"])
            edit = self.client.json(
                _prompt("code_editor_system.md"),
                _prompt("code_editor.md").format(task=json.dumps(self.task, ensure_ascii=False, indent=2), iteration_id=iteration_id, current_files=json.dumps(before, ensure_ascii=False), best=json.dumps(best_record, ensure_ascii=False), analysis=json.dumps(history[-1].get("diagnostic_analysis", {}) if history else {}, ensure_ascii=False), history=json.dumps(history[-3:], ensure_ascii=False)),
                f"iteration-{iteration_id:04d}-code",
            )
            files = _normalize_edit(edit, self.task["editable_files"])
            if not files:
                retry_user = _prompt("code_editor.md").format(task=json.dumps(self.task, ensure_ascii=False, indent=2), iteration_id=iteration_id, current_files=json.dumps(before, ensure_ascii=False), best=json.dumps(best_record, ensure_ascii=False), analysis=json.dumps(history[-1].get("diagnostic_analysis", {}) if history else {}, ensure_ascii=False), history=json.dumps(history[-3:], ensure_ascii=False)) + "\nYour previous response had no valid files. Return complete contents for at least one editable file."
                edit = self.client.json(_prompt("code_editor_system.md"), retry_user, f"iteration-{iteration_id:04d}-code-retry")
                files = _normalize_edit(edit, self.task["editable_files"])
            if not files:
                raise ValueError("code editor returned no editable files")
            modified_program = programs / f"p{iteration_id:04d}"
            shutil.copytree(current, modified_program, dirs_exist_ok=True)
            for relative, content in files.items():
                target = modified_program / relative
                target.parent.mkdir(parents=True, exist_ok=True)
                target.write_text(content, encoding="utf-8")

            iteration_dir = self.run_dir / "iterations" / f"iteration-{iteration_id:04d}"
            iteration_dir.mkdir(parents=True, exist_ok=True)
            initial_eval = self.runner.evaluate(
                modified_program, SearchSpace(), {}, "development", iteration_dir / "initial"
            )
            initial_objective = initial_eval.objective
            initial_result_path = iteration_dir / "initial" / "result.json"
            initial_payload = {}
            if initial_result_path.is_file():
                initial_payload = json.loads(initial_result_path.read_text(encoding="utf-8"))
            initial_evidence = {
                "status": initial_eval.status,
                "objective": initial_objective,
                "elapsed_seconds": initial_eval.elapsed_seconds,
                "error": initial_eval.error,
                "result": initial_payload,
            }

            diagnostic_user = _prompt("diagnostic_planner.md").format(
                task=json.dumps(self.task, ensure_ascii=False, indent=2),
                modified_program_files=json.dumps(_files(modified_program, self.task["editable_files"]), ensure_ascii=False),
                hypothesis=edit.get("hypothesis", ""),
                best=json.dumps(best_record, ensure_ascii=False, indent=2),
                history=json.dumps(history[-5:], ensure_ascii=False, indent=2),
                initial_evaluation=json.dumps(initial_evidence, ensure_ascii=False, indent=2),
                evaluation_budget=self.evaluation_budget,
                evaluations_used=evaluations_used + 1,
                evaluations_remaining=max(0, self.evaluation_budget - evaluations_used - 1),
                current_trial_limit=max(0, current_trial_limit),
            )
            space_value = self.client.json(
                _prompt("diagnostic_planner_system.md"),
                diagnostic_user,
                f"iteration-{iteration_id:04d}-diagnostic-plan",
            )
            space, errors = validate_space(SearchSpace.from_dict(space_value), _files(modified_program, self.task["editable_files"]), self.task["editable_files"])
            use_hpo = bool(space_value.get("use_hpo", False))
            diagnostic_trials = space_value.get("diagnostic_trials", 0) if use_hpo else 0
            plan_errors = list(errors)
            if not isinstance(diagnostic_trials, int) or isinstance(diagnostic_trials, bool) or diagnostic_trials < 0 or diagnostic_trials > max(0, current_trial_limit):
                plan_errors.append(f"diagnostic_trials must be an integer between 0 and {max(0, current_trial_limit)}")
            if use_hpo and (not space.parameters or diagnostic_trials < 1):
                plan_errors.append("use_hpo requires a non-empty valid search space and at least one diagnostic trial")
            if plan_errors:
                retry_user = diagnostic_user + "\nYour previous diagnostic plan was invalid: " + json.dumps(plan_errors, ensure_ascii=False) + ". Return a valid plan within the remaining evaluation budget."
                space_value = self.client.json(_prompt("diagnostic_planner_system.md"), retry_user, f"iteration-{iteration_id:04d}-diagnostic-plan-retry")
                space, errors = validate_space(SearchSpace.from_dict(space_value), _files(modified_program, self.task["editable_files"]), self.task["editable_files"])
                use_hpo = bool(space_value.get("use_hpo", False))
                diagnostic_trials = space_value.get("diagnostic_trials", 0) if use_hpo else 0
                plan_errors = list(errors)
                if not isinstance(diagnostic_trials, int) or isinstance(diagnostic_trials, bool) or diagnostic_trials < 0 or diagnostic_trials > max(0, current_trial_limit):
                    plan_errors.append(f"diagnostic_trials must be an integer between 0 and {max(0, current_trial_limit)}")
                if use_hpo and (not space.parameters or diagnostic_trials < 1):
                    plan_errors.append("use_hpo requires a non-empty valid search space and at least one diagnostic trial")
            if plan_errors:
                raise ValueError(f"Diagnostic Agent returned an invalid plan: {plan_errors}")
            (iteration_dir / "diagnostic_plan.json").write_text(json.dumps({"use_hpo": use_hpo, "diagnostic_trials": diagnostic_trials, "current_trial_limit": current_trial_limit, "tpe_settings": space_value.get("tpe_settings", {}), "space": space.to_dict(), "strategy": space_value.get("strategy", ""), "errors": errors}, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
            if use_hpo:
                diagnostic_result = self.runner.run(
                    modified_program, space, diagnostic_trials, seed + iteration_id,
                    iteration_dir / "diagnostic_trials", anchor=initial_eval,
                    tpe_settings=space_value.get("tpe_settings", {}),
                )
                summary = diagnostic_result.to_dict()
            else:
                anchor_trial = initial_eval.to_dict()
                anchor_trial["trial_id"] = 0
                summary = {
                    "direction": direction, "initial_objective": initial_objective,
                    "best_objective": initial_objective, "best_params": {},
                    "trials": [anchor_trial], "best_so_far": [initial_objective] if initial_objective is not None else [],
                    "failure_rate": 0.0 if initial_objective is not None else 1.0,
                    "space": SearchSpace().to_dict(),
                }
            summary["diagnostic_strategy"] = space_value.get("strategy", "")
            summary["diagnostic_rationale"] = space_value.get("rationale", "")
            (iteration_dir / "diagnostic_summary.json").write_text(json.dumps(summary, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
            diagnostic_trials_used += diagnostic_trials
            program_best = summary.get("best_objective")
            program_params = summary.get("best_params", {})
            best_trial = next(
                (trial for trial in summary.get("trials", []) if trial.get("status") == "success" and trial.get("objective") == program_best),
                None,
            )
            best_trial_id = best_trial.get("trial_id", 0) if best_trial else 0
            if program_best is not None and program_params:
                self.runner.materialize(modified_program, space, program_params)
            evaluation_id = f"iteration-{iteration_id:04d}/initial" if best_trial_id == 0 else f"iteration-{iteration_id:04d}/diagnostic-trials/trial-{best_trial_id:04d}"
            program_state = {
                "program_id": modified_program.name, "config": program_params,
                "objective": program_best,
                "evaluation_id": evaluation_id,
            }
            (modified_program / "selected_config.json").write_text(json.dumps(program_state, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
            analysis = self.client.json(
                _prompt("diagnostic_analyzer_system.md"),
                _prompt("diagnostic_analyzer.md").format(
                    task=json.dumps(self.task, ensure_ascii=False, indent=2),
                    hypothesis=edit.get("hypothesis", ""),
                    best=json.dumps(best_record, ensure_ascii=False, indent=2),
                    initial_objective=initial_objective,
                    program_best=json.dumps(program_state, ensure_ascii=False, indent=2),
                    diagnostic_summary=json.dumps(summary, ensure_ascii=False, indent=2),
                ),
                f"iteration-{iteration_id:04d}-diagnostic-analysis",
            )
            improved = program_best is not None and (
                best_record is None or _better(direction, program_best, best_record.get("score"))
            )
            if improved:
                best_record = {
                    "program_id": modified_program.name, "score": program_best,
                    "params": program_params, "iteration": iteration_id,
                    "evaluation_id": program_state["evaluation_id"],
                }
                current = modified_program
            else:
                current = programs / best_record["program_id"] if best_record is not None else initial
            (iteration_dir / "diagnostic_analysis.json").write_text(json.dumps(analysis, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
            record = {
                "iteration": iteration_id, "program_id": modified_program.name,
                "base_program_id": base_program_id,
                "hypothesis": edit.get("hypothesis", ""),
                "initial_objective": initial_objective, "initial_status": initial_eval.status,
                "program_best_objective": program_best,
                "program_best_params": program_params,
                "program_best_evaluation_id": program_state["evaluation_id"],
                "improved_best": improved,
                "diagnostic_trials_used": diagnostic_trials,
                "per_iteration_trial_limit": current_trial_limit,
                "diagnostic_summary": summary,
                "diagnostic_analysis": analysis,
            }
            (iteration_dir / "record.json").write_text(json.dumps(record, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
            history.append(record)
            iteration_id += 1

        result = {
            "task": self.task["task_id"],
            "iterations_completed": len(history),
            "evaluation_budget": self.evaluation_budget,
            "evaluations_used": len(history) + diagnostic_trials_used,
            "per_iteration_trial_limit": self.per_iteration_trial_limit,
            "code_evaluations": len(history),
            "diagnostic_trials": diagnostic_trials_used,
            "best": best_record,
            "history": history,
        }
        if best_record is not None:
            selected = programs / best_record["program_id"]
            final_evaluation = self.runner.evaluate(selected, SearchSpace(), {}, "test", self.run_dir / "final_test")
            result["final_test"] = final_evaluation.to_dict()
            (self.run_dir / "final_test.json").write_text(json.dumps(final_evaluation.to_dict(), ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
        (self.run_dir / "summary.json").write_text(json.dumps(result, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
        return result
