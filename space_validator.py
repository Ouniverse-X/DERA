"""Deterministic validation and AST inspection for LLM-generated HPO spaces."""

from __future__ import annotations

import ast
from pathlib import Path
from typing import Any

from schema import ParameterSpec, SearchSpace


ALLOWED_KINDS = {"float", "int", "categorical", "bool"}
FORBIDDEN_NAMES = {"SEED", "DATA_SPLIT", "DATA_DIR", "METRIC"}


def top_level_constants(source: str) -> dict[str, Any]:
    tree = ast.parse(source)
    values: dict[str, Any] = {}
    for node in tree.body:
        if not isinstance(node, ast.Assign):
            continue
        if len(node.targets) != 1 or not isinstance(node.targets[0], ast.Name):
            continue
        try:
            values[node.targets[0].id] = ast.literal_eval(node.value)
        except (ValueError, TypeError):
            continue
    return values


def validate_space(space: SearchSpace, files: dict[str, str], editable_files: list[str]) -> tuple[SearchSpace, list[str]]:
    errors: list[str] = []
    seen: set[str] = set()
    accepted: list[ParameterSpec] = []
    for parameter in space.parameters:
        if parameter.name in seen:
            errors.append(f"duplicate parameter: {parameter.name}")
            continue
        seen.add(parameter.name)
        if parameter.file not in editable_files or parameter.file not in files:
            errors.append(f"file is not editable or unavailable: {parameter.file}")
            continue
        constants = top_level_constants(files[parameter.file])
        if parameter.name not in constants:
            errors.append(f"parameter is not a top-level literal constant: {parameter.name}")
            continue
        if parameter.name in FORBIDDEN_NAMES:
            errors.append(f"forbidden budget/protocol parameter: {parameter.name}")
            continue
        if parameter.kind not in ALLOWED_KINDS:
            errors.append(f"unsupported type for {parameter.name}: {parameter.kind}")
            continue
        if parameter.kind in {"float", "int"}:
            if parameter.low is None or parameter.high is None or parameter.low >= parameter.high:
                errors.append(f"invalid numeric range: {parameter.name}")
                continue
            if parameter.log and parameter.low <= 0:
                errors.append(f"log range must be positive: {parameter.name}")
                continue
            default = constants[parameter.name]
            if not isinstance(default, (int, float)) or isinstance(default, bool) or not parameter.low <= default <= parameter.high:
                errors.append(f"current inherited value is outside range: {parameter.name}")
                continue
        elif parameter.kind == "categorical" and len(parameter.choices) < 2:
            errors.append(f"categorical parameter needs at least two choices: {parameter.name}")
            continue
        elif parameter.kind == "categorical" and constants[parameter.name] not in parameter.choices:
            errors.append(f"current inherited value is outside choices: {parameter.name}")
            continue
        accepted.append(parameter)
    return SearchSpace(accepted, space.fixed_parameters, space.rationale), errors
