"""Validated data structures for DERA experiments."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Literal


ParamKind = Literal["float", "int", "categorical", "bool"]


@dataclass(frozen=True)
class ParameterSpec:
    file: str
    name: str
    kind: ParamKind
    low: float | int | None = None
    high: float | int | None = None
    choices: tuple[Any, ...] = ()
    log: bool = False
    reason: str = ""

    @classmethod
    def from_dict(cls, value: dict[str, Any]) -> "ParameterSpec":
        kind = value.get("type", value.get("kind"))
        choices = tuple(value.get("choices", ()))
        return cls(
            file=str(value.get("file", "")), name=str(value["name"]), kind=kind,
            low=value.get("low"), high=value.get("high"), choices=choices,
            log=bool(value.get("log", False)), reason=str(value.get("reason", "")),
        )

    def to_dict(self) -> dict[str, Any]:
        value: dict[str, Any] = {"file": self.file, "name": self.name, "type": self.kind, "log": self.log}
        if self.low is not None: value["low"] = self.low
        if self.high is not None: value["high"] = self.high
        if self.choices: value["choices"] = list(self.choices)
        if self.reason: value["reason"] = self.reason
        return value


@dataclass
class SearchSpace:
    parameters: list[ParameterSpec] = field(default_factory=list)
    fixed_parameters: list[str] = field(default_factory=list)
    rationale: str = ""

    @classmethod
    def from_dict(cls, value: dict[str, Any]) -> "SearchSpace":
        return cls(
            parameters=[ParameterSpec.from_dict(x) for x in value.get("parameters", [])],
            fixed_parameters=[str(x) for x in value.get("fixed_parameters", [])],
            rationale=str(value.get("rationale", "")),
        )

    def to_dict(self) -> dict[str, Any]:
        return {"parameters": [x.to_dict() for x in self.parameters], "fixed_parameters": self.fixed_parameters, "rationale": self.rationale}


@dataclass
class TrialRecord:
    trial_id: int
    params: dict[str, Any]
    status: Literal["success", "failed"]
    objective: float | None = None
    elapsed_seconds: float | None = None
    error: str | None = None

    def to_dict(self) -> dict[str, Any]:
        return self.__dict__.copy()


@dataclass
class DiagnosticSummary:
    direction: Literal["minimize", "maximize"]
    initial_objective: float | None
    best_objective: float | None
    best_params: dict[str, Any]
    trials: list[TrialRecord]
    best_so_far: list[float]
    failure_rate: float
    space: SearchSpace

    def to_dict(self) -> dict[str, Any]:
        return {
            "direction": self.direction, "initial_objective": self.initial_objective,
            "best_objective": self.best_objective, "best_params": self.best_params,
            "trials": [x.to_dict() for x in self.trials], "best_so_far": self.best_so_far,
            "failure_rate": self.failure_rate, "space": self.space.to_dict(),
        }
