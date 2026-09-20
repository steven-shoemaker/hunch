from __future__ import annotations

from types import SimpleNamespace
from typing import Any, Callable, Mapping


class FakeLLM:
    name = "fake"

    def __init__(self, text: str | list[str]) -> None:
        self.queue = [text] if isinstance(text, str) else list(text)
        self.calls: list[tuple[str, str]] = []

    def complete(self, *, system: str, user: str) -> str:
        self.calls.append((system, user))
        if not self.queue:
            raise AssertionError("FakeLLM has no remaining completions.")
        return self.queue.pop(0)


class FakeJev:
    def __init__(self, handler: Callable[[Any, Mapping[str, Any]], Any]) -> None:
        self.handler = handler
        self.calls: list[tuple[Any, Mapping[str, Any]]] = []

    def system_one(self, state=None, questions=None, **kwargs):
        self.calls.append((state, questions))
        return self.handler(state, questions)


def choice_answer(choice: str, probabilities: Mapping[str, float], confidence: float) -> SimpleNamespace:
    return SimpleNamespace(choice=choice, probabilities=dict(probabilities), confidence=confidence)


def noul_answer(noul: float) -> SimpleNamespace:
    return SimpleNamespace(noul=noul)


def score_answer(
    score: float,
    probabilities: Mapping[int, float],
    confidence: float,
    legend: Mapping[int, str],
) -> SimpleNamespace:
    return SimpleNamespace(
        score=score,
        probabilities=dict(probabilities),
        confidence=confidence,
        legend=dict(legend),
    )


def response(**answers: Any) -> SimpleNamespace:
    return SimpleNamespace(
        answers=answers,
        choices=answers,
        nouls=answers,
        scores=answers,
        model="jev-test",
        usage=SimpleNamespace(input_tokens=10, output_tokens=2),
    )
