"""Validated Python question objects."""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from itertools import pairwise
from types import MappingProxyType
from typing import Any, TypeAlias

from .exceptions import InvalidQuestion

MAX_CANDIDATES = 128


def _nonblank(value: str, field: str) -> str:
    if not isinstance(value, str) or not value.strip():
        raise InvalidQuestion(f"{field} must be a non-blank string")
    return value


@dataclass(frozen=True, slots=True)
class Choice:
    prompt: str
    options: Mapping[str, str]

    def __post_init__(self) -> None:
        _nonblank(self.prompt, "Choice.prompt")
        if not isinstance(self.options, Mapping):
            raise InvalidQuestion("Choice.options must be a mapping")
        copied: dict[str, str] = {}
        for label, criterion in self.options.items():
            copied[_nonblank(label, "Choice option label")] = _nonblank(
                criterion, "Choice option criterion"
            )
        if not 2 <= len(copied) <= MAX_CANDIDATES:
            raise InvalidQuestion(f"Choice requires 2..={MAX_CANDIDATES} options")
        object.__setattr__(self, "options", MappingProxyType(copied))

    def to_wire(self) -> dict[str, Any]:
        return {"type": "choice", "prompt": self.prompt, "options": dict(self.options)}


@dataclass(frozen=True, slots=True)
class Noul:
    prompt: str

    def __post_init__(self) -> None:
        _nonblank(self.prompt, "Noul.prompt")

    def to_wire(self) -> dict[str, Any]:
        return {"type": "noul", "prompt": self.prompt}


@dataclass(frozen=True, slots=True)
class ScoreLevel:
    label: str
    criterion: str
    value: float

    def __post_init__(self) -> None:
        _nonblank(self.label, "ScoreLevel.label")
        _nonblank(self.criterion, "ScoreLevel.criterion")
        if not isinstance(self.value, (int, float)) or not float(self.value) == self.value:
            raise InvalidQuestion("ScoreLevel.value must be numeric")
        if not float("-inf") < float(self.value) < float("inf"):
            raise InvalidQuestion("ScoreLevel.value must be finite")

    def to_wire(self) -> dict[str, Any]:
        return {
            "label": self.label,
            "criterion": self.criterion,
            "value": float(self.value),
        }


@dataclass(frozen=True, slots=True, init=False)
class Score:
    prompt: str
    levels: tuple[ScoreLevel, ...]

    def __init__(
        self,
        prompt: str,
        levels: Sequence[str | ScoreLevel] | Mapping[str, str],
    ) -> None:
        object.__setattr__(self, "prompt", _nonblank(prompt, "Score.prompt"))
        normalized: list[ScoreLevel]
        if isinstance(levels, Mapping):
            normalized = [
                ScoreLevel(str(label), criterion, float(index))
                for index, (label, criterion) in enumerate(levels.items())
            ]
        elif isinstance(levels, Sequence) and not isinstance(levels, (str, bytes)):
            normalized = []
            for index, level in enumerate(levels):
                if isinstance(level, ScoreLevel):
                    normalized.append(level)
                elif isinstance(level, str):
                    normalized.append(ScoreLevel(str(index), level, float(index)))
                else:
                    raise InvalidQuestion("Score levels must be strings or ScoreLevel values")
        else:
            raise InvalidQuestion("Score.levels must be a sequence or mapping")

        if not 2 <= len(normalized) <= MAX_CANDIDATES:
            raise InvalidQuestion(f"Score requires 2..={MAX_CANDIDATES} levels")
        if any(left.value >= right.value for left, right in pairwise(normalized)):
            raise InvalidQuestion("Score level values must be strictly increasing")
        labels = [level.label for level in normalized]
        if len(set(labels)) != len(labels):
            raise InvalidQuestion("Score level labels must be unique")
        object.__setattr__(self, "levels", tuple(normalized))

    def to_wire(self) -> dict[str, Any]:
        return {
            "type": "score",
            "prompt": self.prompt,
            "levels": [level.to_wire() for level in self.levels],
        }


Question: TypeAlias = Choice | Noul | Score
