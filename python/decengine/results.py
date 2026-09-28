"""Immutable typed result objects."""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass
from math import isfinite
from types import MappingProxyType
from typing import Any, TypeAlias

from .exceptions import EngineError


def _probability(value: Any, field: str) -> float:
    if not isinstance(value, (int, float)):
        raise EngineError(f"native response {field} is not numeric")
    result = float(value)
    if not 0.0 <= result <= 1.0:
        raise EngineError(f"native response {field} is outside [0, 1]")
    return result


def _distribution(value: Any, field: str) -> Mapping[str, float]:
    if not isinstance(value, dict) or not value:
        raise EngineError(f"native response {field} must be a non-empty object")
    output = MappingProxyType(
        {str(key): _probability(probability, field) for key, probability in value.items()}
    )
    if abs(sum(output.values()) - 1.0) > 1e-3:
        raise EngineError(f"native response {field} does not sum to 1")
    return output


def _finite_number(value: Any, field: str) -> float:
    if not isinstance(value, (int, float)) or not isfinite(float(value)):
        raise EngineError(f"native response {field} must be finite and numeric")
    return float(value)


@dataclass(frozen=True, slots=True)
class ChoiceResult:
    selected: str
    probabilities: Mapping[str, float]
    confidence: float


@dataclass(frozen=True, slots=True)
class NoulResult:
    value: float
    confidence: float

    def __bool__(self) -> bool:
        return self.value >= 0.5


@dataclass(frozen=True, slots=True)
class ScoreResult:
    value: float
    distribution: Mapping[str, float]
    legend: Mapping[str, str]
    confidence: float


Result: TypeAlias = ChoiceResult | NoulResult | ScoreResult


@dataclass(frozen=True, slots=True)
class Usage:
    input_characters: int
    questions: int
    candidates: int
    candidate_cache_hits: int
    candidate_cache_misses: int


@dataclass(frozen=True, slots=True)
class DecisionResponse:
    id: str
    model: str
    created: int
    results: Mapping[str, Result]
    usage: Usage

    def __getitem__(self, name: str) -> Result:
        return self.results[name]

    @classmethod
    def from_wire(cls, value: Any) -> DecisionResponse:
        if not isinstance(value, dict):
            raise EngineError("native response must be an object")
        raw_results = value.get("results")
        if not isinstance(raw_results, dict):
            raise EngineError("native response results must be an object")
        results = {
            str(name): _parse_result(str(name), result) for name, result in raw_results.items()
        }
        raw_usage = value.get("usage")
        if not isinstance(raw_usage, dict):
            raise EngineError("native response usage must be an object")
        try:
            usage = Usage(
                input_characters=int(raw_usage["input_characters"]),
                questions=int(raw_usage["questions"]),
                candidates=int(raw_usage["candidates"]),
                candidate_cache_hits=int(raw_usage["candidate_cache_hits"]),
                candidate_cache_misses=int(raw_usage["candidate_cache_misses"]),
            )
            return cls(
                id=str(value["id"]),
                model=str(value["model"]),
                created=int(value["created"]),
                results=MappingProxyType(results),
                usage=usage,
            )
        except (KeyError, TypeError, ValueError) as error:
            raise EngineError(f"malformed native response: {error}") from error


def _parse_result(name: str, value: Any) -> Result:
    if not isinstance(value, dict):
        raise EngineError(f"native result {name!r} must be an object")
    kind = value.get("type")
    try:
        if kind == "choice":
            probabilities = _distribution(value.get("probabilities"), f"{name}.probabilities")
            selected = str(value["selected"])
            if selected not in probabilities:
                raise EngineError(f"native result {name!r} selected unknown candidate")
            return ChoiceResult(
                selected=selected,
                probabilities=probabilities,
                confidence=_probability(value.get("confidence"), f"{name}.confidence"),
            )
        if kind == "noul":
            return NoulResult(
                value=_probability(value.get("value"), f"{name}.value"),
                confidence=_probability(value.get("confidence"), f"{name}.confidence"),
            )
        if kind == "score":
            distribution = _distribution(value.get("distribution"), f"{name}.distribution")
            legend = value.get("legend")
            if not isinstance(legend, dict):
                raise EngineError(f"native result {name!r} legend must be an object")
            return ScoreResult(
                value=_finite_number(value.get("value"), f"{name}.value"),
                distribution=distribution,
                legend=MappingProxyType({str(key): str(item) for key, item in legend.items()}),
                confidence=_probability(value.get("confidence"), f"{name}.confidence"),
            )
    except (KeyError, TypeError, ValueError) as error:
        raise EngineError(f"malformed native result {name!r}: {error}") from error
    raise EngineError(f"native result {name!r} has unknown type {kind!r}")
