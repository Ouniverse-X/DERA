"""DERA research loop."""

from __future__ import annotations

import json
import shutil
from pathlib import Path
from typing import Any

try:
    from .llm_client import DeepSeekClient
    from .schema import SearchSpace
    from .space_validator import validate_space
    from .hpo_runner import PilotRunner
except ImportError:
    from llm_client import DeepSeekClient
    from schema import SearchSpace
    from space_validator import validate_space
    from hpo_runner import PilotRunner


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
    for name in ("change_summary", "implemented_mechanism", "planned_vs_actual"):
        value = raw.get(name)
        if name not in edit and isinstance(value, str):
            edit[name] = value
    return changed


def _objective_from_result(path: Path, metric: str) -> float | None:
    if not path.is_file():
        return None
    payload = json.loads(path.read_text(encoding="utf-8"))
    return float(payload["metrics"][metric])


def _resolve_decision(direction: str, candidate_best: float | None, incumbent: dict[str, Any] | None, requested: str) -> str:
    if candidate_best is not None and (incumbent is None or _better(direction, candidate_best, incumbent.get("score"))):
        return "promote"
    if requested == "keep_candidate" and candidate_best is not None:
        return "archive"
    if requested == "debug_candidate":
        return "debug"
    return "discard"


class DERA:
    def __init__(self, task_dir: Path, data_root: Path, run_dir: Path, client: DeepSeekClient, device: str = "auto", pilot_trials: int = 3, timeout: float = 1200.0, hpo_budget: int | None = None, max_hpo_trials_per_round: int | None = None, evaluation_budget: int | None = None) -> None:
        self.task_dir = task_dir.resolve()
        self.data_root = data_root.expanduser().resolve()
        self.run_dir = run_dir.resolve()
        self.client = client
        self.device = device
        self.pilot_trials = pilot_trials
        self.hpo_budget = hpo_budget
        self.evaluation_budget = evaluation_budget
        self.max_hpo_trials_per_round = max_hpo_trials_per_round
        self.task = json.loads((self.task_dir / "task.json").read_text(encoding="utf-8"))
        self.runner = PilotRunner(self.task_dir, self.data_root, device, timeout)

    def run(self, rounds: int, seed: int = 0) -> dict[str, Any]:
        allowed_initial = {"llm_calls", "solutions", "rounds", "initial_development", "archive.json"}
        existing = set(item.name for item in self.run_dir.iterdir()) if self.run_dir.exists() else set()
        if existing - allowed_initial:
            raise ValueError("run directory must be empty or new")
        self.run_dir.mkdir(parents=True, exist_ok=True)
        solutions = self.run_dir / "solutions"
        solutions.mkdir(exist_ok=True)
        initial = solutions / "s0000"
        if not initial.exists():
            shutil.copytree(self.task_dir / "base", initial)
        metric = self.task["objective"]["metric"]
        initial_result = self.run_dir / "initial_development" / "result.json"
        initial_score = _objective_from_result(initial_result, metric)
        if initial_score is None:
            initial_eval = self.runner.evaluate_frozen(
                initial, SearchSpace(), {}, "development", self.run_dir / "initial_development"
            )
            initial_score = initial_eval.objective
        best_record: dict[str, Any] | None = None
        if initial_score is not None:
            best_record = {
                "solution_id": "s0000", "score": initial_score, "params": {},
                "round": 0, "evaluation_id": "initial_development",
            }
        current = initial
        history: list[dict[str, Any]] = []
        existing_rounds = sorted((self.run_dir / "rounds").glob("round-*")) if (self.run_dir / "rounds").exists() else []
        if existing_rounds:
            for folder in existing_rounds:
                record_file = folder / "record.json"
                if record_file.is_file():
                    record = json.loads(record_file.read_text(encoding="utf-8"))
                    history.append(record)
            start_round = len(history) + 1
        else:
            start_round = 1
        direction = self.task["objective"]["direction"]
        for record in history:
            score = record.get("candidate_best_objective", record.get("default_objective"))
            if score is not None and (best_record is None or _better(direction, score, best_record["score"])):
                best_record = {
                    "solution_id": record["solution_id"], "score": score,
                    "params": record.get("candidate_best_params", record.get("hpo", {}).get("best_params", {})),
                    "round": record["round"],
                    "evaluation_id": record.get("candidate_best_evaluation_id", f"round-{record['round']:04d}/default"),
                }
        if best_record is not None:
            current = solutions / best_record["solution_id"]
        if history and history[-1].get("action", {}).get("controller_decision") in {"archive", "debug"}:
            branch = solutions / history[-1]["solution_id"]
            if branch.is_dir():
                current = branch
        archive_path = self.run_dir / "archive.json"
        archive: list[dict[str, Any]] = json.loads(archive_path.read_text(encoding="utf-8")) if archive_path.is_file() else []
        total_hpo_budget = self.hpo_budget if self.hpo_budget is not None else rounds * self.pilot_trials
        hpo_trials_used = sum(int(record.get("hpo_trials_used", 0)) for record in history)
        default_evaluations_used = 1 + len(history)

        for round_id in range(start_round, rounds + 1):
            charged_evaluations_used = len(history) + hpo_trials_used
            if self.evaluation_budget is not None and charged_evaluations_used >= self.evaluation_budget:
                break
            remaining_hpo_budget = total_hpo_budget - hpo_trials_used
            per_round_hpo_limit = remaining_hpo_budget
            if self.max_hpo_trials_per_round is not None:
                per_round_hpo_limit = min(per_round_hpo_limit, self.max_hpo_trials_per_round)
            if self.evaluation_budget is not None:
                per_round_hpo_limit = min(per_round_hpo_limit, self.evaluation_budget - charged_evaluations_used - 1)
            base_solution_id = current.name
            before = _files(current, self.task["editable_files"])
            edit = self.client.json(
                _prompt("code_editor_system.md"),
                _prompt("code_editor.md").format(task=json.dumps(self.task, ensure_ascii=False, indent=2), round_id=round_id, current_files=json.dumps(before, ensure_ascii=False), incumbent=json.dumps(best_record, ensure_ascii=False), archive=json.dumps(archive, ensure_ascii=False), feedback=json.dumps(history[-1] if history else {}, ensure_ascii=False), history=json.dumps(history[-3:], ensure_ascii=False)),
                f"round-{round_id:04d}-editor",
            )
            files = _normalize_edit(edit, self.task["editable_files"])
            if not files:
                retry_user = _prompt("code_editor.md").format(task=json.dumps(self.task, ensure_ascii=False, indent=2), round_id=round_id, current_files=json.dumps(before, ensure_ascii=False), incumbent=json.dumps(best_record, ensure_ascii=False), archive=json.dumps(archive, ensure_ascii=False), feedback=json.dumps(history[-1] if history else {}, ensure_ascii=False), history=json.dumps(history[-3:], ensure_ascii=False)) + "\nYour previous response had no valid files. Return complete contents for at least one editable file."
                edit = self.client.json(_prompt("code_editor_system.md"), retry_user, f"round-{round_id:04d}-editor-retry")
                files = _normalize_edit(edit, self.task["editable_files"])
            if not files:
                raise ValueError("code editor returned no editable files")
            candidate = solutions / f"s{round_id:04d}"
            shutil.copytree(current, candidate, dirs_exist_ok=True)
            for relative, content in files.items():
                target = candidate / relative
                target.parent.mkdir(parents=True, exist_ok=True)
                target.write_text(content, encoding="utf-8")

            round_dir = self.run_dir / "rounds" / f"round-{round_id:04d}"
            round_dir.mkdir(parents=True, exist_ok=True)
            default_eval = self.runner.evaluate_frozen(
                candidate, SearchSpace(), {}, "development", round_dir / "default"
            )
            default_evaluations_used += 1
            default_objective = default_eval.objective
            default_result_path = round_dir / "default" / "result.json"
            default_payload = {}
            if default_result_path.is_file():
                default_payload = json.loads(default_result_path.read_text(encoding="utf-8"))
            default_evidence = {
                "status": default_eval.status,
                "objective": default_objective,
                "elapsed_seconds": default_eval.elapsed_seconds,
                "error": default_eval.error,
                "result": default_payload,
            }

            optimization_user = _prompt("space_designer.md").format(
                task=json.dumps(self.task, ensure_ascii=False, indent=2),
                candidate_files=json.dumps(_files(candidate, self.task["editable_files"]), ensure_ascii=False),
                change_summary=edit.get("change_summary", ""),
                incumbent=json.dumps(best_record, ensure_ascii=False, indent=2),
                archive=json.dumps(archive, ensure_ascii=False, indent=2),
                history=json.dumps(history[-5:], ensure_ascii=False, indent=2),
                default_evaluation=json.dumps(default_evidence, ensure_ascii=False, indent=2),
                total_hpo_budget=total_hpo_budget,
                hpo_trials_used=hpo_trials_used,
                remaining_hpo_budget=max(0, remaining_hpo_budget),
                per_round_hpo_limit=max(0, per_round_hpo_limit),
                rounds_remaining=rounds - round_id + 1,
            )
            space_value = self.client.json(
                _prompt("space_designer_system.md"),
                optimization_user,
                f"round-{round_id:04d}-space",
            )
            space, errors = validate_space(SearchSpace.from_dict(space_value), _files(candidate, self.task["editable_files"]), self.task["editable_files"])
            use_tpe = bool(space_value.get("use_tpe", False))
            tpe_trials = space_value.get("tpe_trials", 0) if use_tpe else 0
            plan_errors = list(errors)
            if not isinstance(tpe_trials, int) or isinstance(tpe_trials, bool) or tpe_trials < 0 or tpe_trials > max(0, per_round_hpo_limit):
                plan_errors.append(f"tpe_trials must be an integer between 0 and {max(0, per_round_hpo_limit)}")
            if use_tpe and (not space.parameters or tpe_trials < 1):
                plan_errors.append("use_tpe requires a non-empty valid search space and at least one trial")
            if plan_errors:
                retry_user = optimization_user + "\nYour previous optimization plan was invalid: " + json.dumps(plan_errors, ensure_ascii=False) + ". Return a valid autonomous TPE decision within the remaining global budget."
                space_value = self.client.json(_prompt("space_designer_system.md"), retry_user, f"round-{round_id:04d}-space-retry")
                space, errors = validate_space(SearchSpace.from_dict(space_value), _files(candidate, self.task["editable_files"]), self.task["editable_files"])
                use_tpe = bool(space_value.get("use_tpe", False))
                tpe_trials = space_value.get("tpe_trials", 0) if use_tpe else 0
                plan_errors = list(errors)
                if not isinstance(tpe_trials, int) or isinstance(tpe_trials, bool) or tpe_trials < 0 or tpe_trials > max(0, per_round_hpo_limit):
                    plan_errors.append(f"tpe_trials must be an integer between 0 and {max(0, per_round_hpo_limit)}")
                if use_tpe and (not space.parameters or tpe_trials < 1):
                    plan_errors.append("use_tpe requires a non-empty valid search space and at least one trial")
            if plan_errors:
                raise ValueError(f"Optimization Agent returned an invalid plan: {plan_errors}")
            (round_dir / "hpo_space.json").write_text(json.dumps({"use_tpe": use_tpe, "tpe_trials": tpe_trials, "per_round_hpo_limit": per_round_hpo_limit, "total_hpo_budget": total_hpo_budget, "tpe_settings": space_value.get("tpe_settings", {}), "space": space.to_dict(), "strategy": space_value.get("strategy", ""), "errors": errors}, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
            if use_tpe:
                hpo = self.runner.run(
                    candidate, space, tpe_trials, seed + round_id,
                    round_dir / "hpo", sampler_name="tpe_dynamic", anchor=default_eval,
                    tpe_settings=space_value.get("tpe_settings", {}),
                )
                summary = hpo.to_dict()
            else:
                anchor_trial = default_eval.to_dict()
                anchor_trial["trial_id"] = 0
                summary = {
                    "direction": direction, "default_objective": default_objective,
                    "best_objective": default_objective, "best_params": {},
                    "trials": [anchor_trial], "best_so_far": [default_objective] if default_objective is not None else [],
                    "failure_rate": 0.0 if default_objective is not None else 1.0,
                    "space": SearchSpace().to_dict(),
                }
            summary["optimization_strategy"] = space_value.get("strategy", "")
            summary["optimization_reason"] = space_value.get("rationale", "")
            (round_dir / "hpo_summary.json").write_text(json.dumps(summary, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
            hpo_trials_used += tpe_trials
            candidate_best = summary.get("best_objective")
            candidate_params = summary.get("best_params", {})
            best_trial = next(
                (trial for trial in summary.get("trials", []) if trial.get("status") == "success" and trial.get("objective") == candidate_best),
                None,
            )
            best_trial_id = best_trial.get("trial_id", 0) if best_trial else 0
            if candidate_best is not None and candidate_params:
                self.runner.materialize(candidate, space, candidate_params)
            evaluation_id = f"round-{round_id:04d}/default" if best_trial_id == 0 else f"round-{round_id:04d}/hpo/trial-{best_trial_id:04d}"
            frozen_state = {
                "solution_id": candidate.name, "config": candidate_params,
                "objective": candidate_best,
                "evaluation_id": evaluation_id,
            }
            (candidate / "frozen_config.json").write_text(json.dumps(frozen_state, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
            action = self.client.json(
                _prompt("feedback_controller_system.md"),
                _prompt("feedback_controller.md").format(
                    task=json.dumps(self.task, ensure_ascii=False, indent=2),
                    change_summary=edit.get("change_summary", ""),
                    incumbent=json.dumps(best_record, ensure_ascii=False, indent=2),
                    default_objective=default_objective,
                    candidate_best=json.dumps(frozen_state, ensure_ascii=False, indent=2),
                    hpo_summary=json.dumps(summary, ensure_ascii=False, indent=2),
                ),
                f"round-{round_id:04d}-feedback",
            )
            requested = str(action.get("decision", "discard_candidate"))
            controller_decision = _resolve_decision(direction, candidate_best, best_record, requested)
            if controller_decision == "promote":
                best_record = {
                    "solution_id": candidate.name, "score": candidate_best,
                    "params": candidate_params, "round": round_id,
                    "evaluation_id": frozen_state["evaluation_id"],
                }
                current = candidate
            elif controller_decision == "archive":
                archive.append(frozen_state)
                archive.sort(key=lambda item: item["objective"], reverse=direction == "maximize")
                archive = archive[:5]
                current = candidate
            elif controller_decision == "debug":
                current = candidate
            else:
                controller_decision = "discard"
                current = solutions / best_record["solution_id"] if best_record is not None else initial
            action["controller_decision"] = controller_decision
            archive_path.write_text(json.dumps(archive, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
            (round_dir / "action.json").write_text(json.dumps(action, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
            record = {
                "round": round_id, "solution_id": candidate.name,
                "base_solution_id": base_solution_id,
                "change_summary": edit.get("change_summary", ""), "action": action,
                "default_objective": default_objective, "default_status": default_eval.status,
                "candidate_best_objective": candidate_best,
                "candidate_best_params": candidate_params,
                "candidate_best_evaluation_id": frozen_state["evaluation_id"],
                "hpo_trials_used": tpe_trials,
                "per_round_hpo_limit": per_round_hpo_limit,
                "hpo_budget_remaining": total_hpo_budget - hpo_trials_used,
                "hpo": summary,
            }
            (round_dir / "record.json").write_text(json.dumps(record, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
            history.append(record)

        result = {"task": self.task["task_id"], "rounds": rounds, "research_rounds_completed": len(history), "evaluation_budget": self.evaluation_budget, "charged_development_evaluations": len(history) + hpo_trials_used, "hpo_budget": total_hpo_budget, "max_hpo_trials_per_round": self.max_hpo_trials_per_round, "hpo_trials_used": hpo_trials_used, "default_evaluations_used": default_evaluations_used, "total_development_evaluations": default_evaluations_used + hpo_trials_used, "best": best_record, "archive": archive, "history": history}
        if best_record is not None:
            selected = solutions / best_record["solution_id"]
            frozen = self.runner.evaluate_frozen(selected, SearchSpace(), {}, "test", self.run_dir / "final_test")
            result["final_test"] = frozen.to_dict()
            (self.run_dir / "final_test.json").write_text(json.dumps(frozen.to_dict(), ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
        (self.run_dir / "summary.json").write_text(json.dumps(result, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
        return result
