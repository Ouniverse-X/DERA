"""Small-budget parameter probes for a DERA code candidate."""

from __future__ import annotations

import ast
import json
import math
import os
import signal
import shutil
import subprocess
import sys
import time
from pathlib import Path
from typing import Any

import optuna

try:
    from .schema import HPOSummary, SearchSpace, TrialRecord
except ImportError:
    from schema import HPOSummary, SearchSpace, TrialRecord


def _patch_constants(path: Path, values: dict[str, Any]) -> None:
    tree = ast.parse(path.read_text(encoding="utf-8"))
    changed: set[str] = set()
    for node in tree.body:
        if not isinstance(node, ast.Assign):
            continue
        if len(node.targets) != 1 or not isinstance(node.targets[0], ast.Name):
            continue
        name = node.targets[0].id
        if name in values:
            node.value = ast.parse(repr(values[name]), mode="eval").body
            changed.add(name)
    missing = set(values) - changed
    if missing: raise ValueError(f"constants not found: {sorted(missing)}")
    path.write_text(ast.unparse(tree) + "\n", encoding="utf-8")


def _literal_constants(path: Path) -> dict[str, Any]:
    tree = ast.parse(path.read_text(encoding="utf-8"))
    result: dict[str, Any] = {}
    for node in tree.body:
        if not isinstance(node, ast.Assign) or len(node.targets) != 1:
            continue
        target = node.targets[0]
        if not isinstance(target, ast.Name):
            continue
        try:
            result[target.id] = ast.literal_eval(node.value)
        except (ValueError, TypeError):
            continue
    return result


class PilotRunner:
    def __init__(self, task_dir: Path, data_root: Path, device: str, timeout: float = 1200.0) -> None:
        self.task_dir = task_dir.resolve()
        self.data_root = data_root.expanduser().resolve()
        self.device = device
        self.timeout = timeout
        self.task = json.loads((self.task_dir / "task.json").read_text(encoding="utf-8"))

    def _sample(self, trial: optuna.Trial, space: SearchSpace) -> dict[str, Any]:
        result: dict[str, Any] = {}
        for p in space.parameters:
            if p.kind == "float": result[p.name] = trial.suggest_float(p.name, float(p.low), float(p.high), log=p.log)
            elif p.kind == "int": result[p.name] = trial.suggest_int(p.name, int(p.low), int(p.high), log=p.log)
            elif p.kind == "categorical": result[p.name] = trial.suggest_categorical(p.name, list(p.choices))
            else: result[p.name] = trial.suggest_categorical(p.name, [False, True])
        return result

    def _distributions(self, space: SearchSpace) -> dict[str, optuna.distributions.BaseDistribution]:
        result: dict[str, optuna.distributions.BaseDistribution] = {}
        for parameter in space.parameters:
            if parameter.kind == "float":
                result[parameter.name] = optuna.distributions.FloatDistribution(float(parameter.low), float(parameter.high), log=parameter.log)
            elif parameter.kind == "int":
                result[parameter.name] = optuna.distributions.IntDistribution(int(parameter.low), int(parameter.high), log=parameter.log)
            elif parameter.kind == "categorical":
                result[parameter.name] = optuna.distributions.CategoricalDistribution(list(parameter.choices))
            else:
                result[parameter.name] = optuna.distributions.CategoricalDistribution([False, True])
        return result

    def config(self, candidate_dir: Path, space: SearchSpace) -> dict[str, Any]:
        """Read the candidate's inherited values for the active search space."""
        result: dict[str, Any] = {}
        constants_by_file: dict[str, dict[str, Any]] = {}
        for parameter in space.parameters:
            if parameter.file not in constants_by_file:
                constants_by_file[parameter.file] = _literal_constants(candidate_dir / parameter.file)
            constants = constants_by_file[parameter.file]
            result[parameter.name] = constants[parameter.name]
        return result

    def materialize(self, candidate_dir: Path, space: SearchSpace, config: dict[str, Any]) -> None:
        """Freeze a measured configuration into the candidate source tree."""
        for relative in {parameter.file for parameter in space.parameters}:
            values = {
                parameter.name: config[parameter.name]
                for parameter in space.parameters
                if parameter.file == relative and parameter.name in config
            }
            if values:
                _patch_constants(candidate_dir / relative, values)

    def _evaluate(self, candidate_dir: Path, config: dict[str, Any], trial_dir: Path, split: str) -> TrialRecord:
        trial_dir.mkdir(parents=True, exist_ok=True)
        solution = trial_dir / "solution"
        shutil.copytree(candidate_dir, solution, dirs_exist_ok=True)
        for relative in {p.file for p in self._current_space.parameters}:
            values = {p.name: config[p.name] for p in self._current_space.parameters if p.file == relative and p.name in config}
            if values: _patch_constants(solution / relative, values)
        data_dir = self.data_root / self.task["data_subdir"] / split
        output = trial_dir / "result.json"
        replacements = {"{python}": sys.executable, "{solution_dir}": str(solution), "{data_dir}": str(data_dir), "{output}": str(output), "{device}": self.device}
        command = self.task["evaluation_command"][:]
        for i, part in enumerate(command):
            for marker, replacement in replacements.items(): command[i] = command[i].replace(marker, replacement)
        started = time.monotonic()
        try:
            process = subprocess.Popen(command, cwd=self.task_dir, text=True, stdout=subprocess.PIPE, stderr=subprocess.PIPE, start_new_session=True)
            stdout, stderr = process.communicate(timeout=self.timeout)
        except subprocess.TimeoutExpired as exc:
            try:
                os.killpg(process.pid, signal.SIGTERM)
                process.communicate(timeout=5)
            except (ProcessLookupError, subprocess.TimeoutExpired):
                try: os.killpg(process.pid, signal.SIGKILL)
                except ProcessLookupError: pass
            return TrialRecord(-1, config, "failed", error=f"timeout: {exc}", elapsed_seconds=time.monotonic() - started)
        if process.returncode != 0 or not output.is_file():
            return TrialRecord(-1, config, "failed", error=stderr[-4000:] or "evaluation failed", elapsed_seconds=time.monotonic() - started)
        metric = self.task["objective"]["metric"]
        try:
            payload = json.loads(output.read_text(encoding="utf-8"))
            metrics = payload["metrics"]
            if not isinstance(metrics, dict) or metric not in metrics:
                raise ValueError(f"missing metrics.{metric}")
            objective = float(metrics[metric])
            if not math.isfinite(objective):
                raise ValueError(f"metrics.{metric} is not finite")
        except (OSError, json.JSONDecodeError, KeyError, TypeError, ValueError) as exc:
            return TrialRecord(-1, config, "failed", error=f"invalid evaluation result: {exc}", elapsed_seconds=time.monotonic() - started)
        return TrialRecord(-1, config, "success", objective, time.monotonic() - started)

    def run(self, candidate_dir: Path, space: SearchSpace, budget: int, seed: int, output_dir: Path, sampler_name: str = "tpe", anchor: TrialRecord | None = None, tpe_settings: dict[str, Any] | None = None) -> HPOSummary:
        if budget < 1: raise ValueError("budget must be positive")
        self._current_space = space
        direction = self.task["objective"]["direction"]
        defaults = self.config(candidate_dir, space)
        if sampler_name == "tpe_dynamic":
            settings = tpe_settings or {}
            sampler = optuna.samplers.TPESampler(
                seed=seed,
                n_startup_trials=max(0, int(settings.get("n_startup_trials", min(5, budget)))),
                multivariate=bool(settings.get("multivariate", False)),
            )
            study = optuna.create_study(direction=direction, sampler=sampler)
            records: list[TrialRecord] = []
            anchor_added = False
            if anchor is not None:
                anchor_record = TrialRecord(0, defaults, anchor.status, anchor.objective, anchor.elapsed_seconds, anchor.error)
                records.append(anchor_record)
                if anchor.status == "success" and anchor.objective is not None:
                    study.add_trial(optuna.trial.create_trial(
                        params=defaults,
                        distributions=self._distributions(space),
                        value=float(anchor.objective),
                    ))
                    anchor_added = True

            def dynamic_objective(trial: optuna.Trial) -> float:
                config = self._sample(trial, space)
                trial_id = trial.number if anchor_added else trial.number + 1
                record = self._evaluate(candidate_dir, config, output_dir / f"trial-{trial_id:04d}", "development")
                record.trial_id = trial_id
                records.append(record)
                if record.status == "failed":
                    trial.set_user_attr("error", record.error or "failed")
                    return 1e9 if direction == "minimize" else -1e9
                return float(record.objective)

            output_dir.mkdir(parents=True, exist_ok=True)
            study.optimize(dynamic_objective, n_trials=budget)
            return self._summarize(direction, space, records, output_dir)
        if sampler_name == "random":
            sampler = optuna.samplers.RandomSampler(seed=seed)
        else:
            sampler = optuna.samplers.TPESampler(seed=seed, n_startup_trials=min(3, max(0, budget - 1)), multivariate=True)
        study = optuna.create_study(direction=direction, sampler=sampler)
        records: list[TrialRecord] = []
        study.enqueue_trial(defaults)

        def objective(trial: optuna.Trial) -> float:
            config = self._sample(trial, space)
            record = self._evaluate(candidate_dir, config, output_dir / f"trial-{trial.number:04d}", "development")
            record.trial_id = trial.number
            records.append(record)
            if record.status == "failed":
                trial.set_user_attr("error", record.error or "failed")
                return 1e9 if direction == "minimize" else -1e9
            return float(record.objective)

        output_dir.mkdir(parents=True, exist_ok=True)
        study.optimize(objective, n_trials=budget)
        return self._summarize(direction, space, records, output_dir)

    def _summarize(self, direction: str, space: SearchSpace, records: list[TrialRecord], output_dir: Path) -> HPOSummary:
        output_dir.mkdir(parents=True, exist_ok=True)
        successful = [x for x in records if x.status == "success" and x.objective is not None]
        ordered = sorted(successful, key=lambda x: x.objective, reverse=direction == "maximize")
        best = ordered[0] if ordered else None
        best_so_far: list[float] = []
        current: float | None = None
        for item in records:
            if item.objective is None: continue
            if current is None or (item.objective < current if direction == "minimize" else item.objective > current): current = item.objective
            best_so_far.append(current)
        default_objective = next((x.objective for x in records if x.trial_id == 0 and x.status == "success"), None)
        summary = HPOSummary(direction, default_objective, best.objective if best else None, best.params if best else {}, records, best_so_far, 1 - len(successful) / len(records) if records else 1.0, space)
        (output_dir / "summary.json").write_text(json.dumps(summary.to_dict(), ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
        return summary

    def evaluate_frozen(self, candidate_dir: Path, space: SearchSpace, params: dict[str, Any], split: str, output_dir: Path) -> TrialRecord:
        """Evaluate a frozen code/configuration on a named split."""
        self._current_space = space
        return self._evaluate(candidate_dir, params, output_dir, split)
