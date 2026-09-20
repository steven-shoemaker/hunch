from __future__ import annotations

from collections.abc import Callable, Mapping
from dataclasses import dataclass
from typing import TypeVar, Union

from hunch.shapes import Shape

T = TypeVar("T")
Branch = Union[T, Callable[[], T]]


def _take(branch: Branch[T]) -> T:
    return branch() if callable(branch) else branch


@dataclass(frozen=True)
class Answer:
    """A resolved Choice: one label from a closed set, plus its shape."""

    top: str
    probabilities: Mapping[str, float]
    confidence: float
    shape: Shape

    @property
    def p(self) -> float:
        return float(self.probabilities[self.top])

    @property
    def top2(self) -> list[str]:
        return sorted(self.probabilities, key=lambda k: self.probabilities[k], reverse=True)[:2]

    def on(self, *, sure: Branch[T], torn: Branch[T], lost: Branch[T]) -> T:
        if self.shape == "sure":
            return _take(sure)
        if self.shape == "torn":
            return _take(torn)
        if self.shape == "lost":
            return _take(lost)
        unreachable: Shape = self.shape
        raise ValueError(f"Unknown shape {unreachable!r}.")


@dataclass(frozen=True)
class Feeling:
    """A resolved Noul. Truthy when P(yes) is at least 0.5."""

    p: float

    def __bool__(self) -> bool:
        return self.p >= 0.5


@dataclass(frozen=True)
class Rating:
    """A resolved Score: position on ordered levels, plus its shape."""

    score: float
    confidence: float
    legend: Mapping[int, str]
    probabilities: Mapping[int, float]
    shape: Shape

    @property
    def level(self) -> str:
        nearest = min(self.legend, key=lambda key: abs(key - self.score))
        return self.legend[nearest]

    @property
    def top(self) -> str:
        return self.level

    def on(self, *, sure: Branch[T], torn: Branch[T], lost: Branch[T]) -> T:
        if self.shape == "sure":
            return _take(sure)
        if self.shape == "torn":
            return _take(torn)
        if self.shape == "lost":
            return _take(lost)
        unreachable: Shape = self.shape
        raise ValueError(f"Unknown shape {unreachable!r}.")
