"""Rich results. Every verb returns these when detail=True."""

from __future__ import annotations

from collections.abc import Callable, Iterator, Mapping
from dataclasses import dataclass
from typing import Any, TypeVar, Union

from hunch.shapes import Shape

T = TypeVar("T")
Branch = Union[T, Callable[[], T]]


def _take(branch: Branch[T]) -> T:
    return branch() if callable(branch) else branch


class _Shaped:
    shape: Shape

    def on(self, *, sure: Branch[T], split: Branch[T], unsure: Branch[T]) -> T:
        """Branch on the distribution shape. Callables are only invoked on their branch."""
        if self.shape == "sure":
            return _take(sure)
        if self.shape == "split":
            return _take(split)
        return _take(unsure)


@dataclass(frozen=True)
class Answer(_Shaped):
    """A resolved Choice: one label from a closed set, plus the whole distribution."""

    label: Any
    """The chosen label. An Enum member when labels were an Enum, else a str."""
    probabilities: Mapping[str, float]
    """P(label) for every option, keyed by the option's string form."""
    confidence: float
    """How peaked the distribution is (0–1). Not whether the label is true."""
    shape: Shape
    by: str | None = None
    """Who chose the label when an LLM policy was set: "jev" or "llm". None otherwise."""

    @property
    def top(self) -> str:
        return _key(self.label)

    @property
    def p(self) -> float:
        return float(self.probabilities[self.top])

    @property
    def ranked(self) -> list[tuple[str, float]]:
        return sorted(self.probabilities.items(), key=lambda kv: kv[1], reverse=True)

    @property
    def top2(self) -> list[str]:
        return [label for label, _ in self.ranked[:2]]


@dataclass(frozen=True)
class MultiAnswer:
    """A resolved multi-label classification: one Noul per label, thresholded."""

    labels: list[Any]
    """Labels whose P(applies) reached the threshold, highest first."""
    probabilities: Mapping[str, float]
    """P(applies) for every label."""
    threshold: float

    def __iter__(self) -> Iterator[Any]:
        return iter(self.labels)

    def __len__(self) -> int:
        return len(self.labels)

    def __contains__(self, item: Any) -> bool:
        return item in self.labels


@dataclass(frozen=True)
class Feeling:
    """A resolved Noul. Truthy when P(yes) reaches the threshold.

    With a `low` cutoff (check(uncertain=(low, high))), P(yes) between the two is "maybe".
    """

    p: float
    threshold: float = 0.5
    low: float | None = None

    def __bool__(self) -> bool:
        return self.p >= self.threshold

    @property
    def verdict(self) -> str:
        """"yes", "no", or "maybe" (only with a low cutoff)."""
        if self.p >= self.threshold:
            return "yes"
        if self.low is None or self.p <= self.low:
            return "no"
        return "maybe"

    @property
    def value(self) -> bool | None:
        """True / False, or None for "maybe"."""
        return {"yes": True, "no": False, "maybe": None}[self.verdict]


@dataclass(frozen=True)
class Rating(_Shaped):
    """A resolved Score: a position along ordered levels, plus the distribution."""

    score: float
    """Probability-weighted position, 0 .. len(levels)-1. Can land between levels."""
    confidence: float
    legend: Mapping[int, str]
    probabilities: Mapping[int, float]
    shape: Shape

    @property
    def level(self) -> str:
        nearest = min(self.legend, key=lambda key: abs(key - self.score))
        return self.legend[nearest]

    @property
    def normalized(self) -> float:
        """Score rescaled to 0–1 so ratings with different level counts compare."""
        top = max(self.legend) if self.legend else 0
        return self.score / top if top else 0.0


@dataclass(frozen=True)
class Pick(_Shaped):
    """A resolved pick(): the winning candidate and how the field ranked."""

    winner: Any
    """The chosen candidate, or None when pick(none=True) found nothing that fits."""
    ranked: list[tuple[Any, float]]
    """(candidate, probability) pairs, best first."""
    confidence: float
    shape: Shape
    fits: float | None = None
    """With none=True: P(at least one candidate satisfies the task)."""


@dataclass(frozen=True)
class Ranked:
    """One row of rank(): the item, its weighted composite, and per-dimension ratings."""

    item: Any
    composite: float
    """Weighted mean of normalized dimension scores, 0–1."""
    ratings: Mapping[str, Rating]

    def __iter__(self) -> Iterator[Any]:
        yield self.item
        yield self.composite


def _key(label: Any) -> str:
    value = getattr(label, "value", label)
    return str(value)
