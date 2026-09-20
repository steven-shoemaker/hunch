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
    """Stands in for TypeSafeClient. handler(state, questions) -> response(...)."""

    def __init__(self, handler: Callable[[Any, Mapping[str, Any]], Any]) -> None:
        self.handler = handler
        self.calls: list[tuple[Any, Mapping[str, Any]]] = []

    def system_one(self, state=None, questions=None, **kwargs):
        self.calls.append((state, dict(questions)))
        return self.handler(state, questions)


def choice_answer(choice: str, probabilities: Mapping[str, float], confidence: float) -> SimpleNamespace:
    return SimpleNamespace(type="choice", choice=choice, probabilities=dict(probabilities), confidence=confidence)


def noul_answer(noul: float) -> SimpleNamespace:
    return SimpleNamespace(type="noul", noul=noul)


def score_answer(
    score: float,
    probabilities: Mapping[int, float],
    confidence: float,
    legend: Mapping[int, str],
) -> SimpleNamespace:
    return SimpleNamespace(
        type="score",
        score=score,
        probabilities=dict(probabilities),
        confidence=confidence,
        legend=dict(legend),
    )


def response(**answers: Any) -> SimpleNamespace:
    return SimpleNamespace(
        answers=answers,
        model="jev-test",
        usage=SimpleNamespace(input_tokens=10, output_tokens=2),
    )


def peaked(qid_to_choice: Mapping[str, str], options: list[str]) -> Callable[[Any, Mapping[str, Any]], Any]:
    """Handler that answers every Choice with a confident pick from qid_to_choice."""

    def handler(state, questions):
        answers = {}
        for qid in questions:
            pick = qid_to_choice[qid]
            probs = {o: (0.9 if o == pick else 0.1 / max(1, len(options) - 1)) for o in options}
            answers[qid] = choice_answer(pick, probs, 0.85)
        return response(**answers)

    return handler
