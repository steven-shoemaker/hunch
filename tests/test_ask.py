from __future__ import annotations

from enum import Enum

import pytest

from hunch import Answer, Check, Classify, Client, Feeling, HunchError, Rate, Rating, ask
from tests.fakes import FakeJev, choice_answer, noul_answer, response, score_answer

LEVELS = ["low", "mid", "high"]


def handler(state, questions):
    assert set(questions) == {"function", "seniority", "urgent", "severity"}
    title = state["input"]
    fn = "Sales" if "Sales" in title else "Engineering"
    return response(
        function=choice_answer(fn, {"Sales": 0.9, "Engineering": 0.1} if fn == "Sales" else {"Sales": 0.1, "Engineering": 0.9}, 0.85),
        seniority=choice_answer("Manager", {"IC": 0.1, "Manager": 0.9}, 0.85),
        urgent=noul_answer(0.7),
        severity=score_answer(1.4, {0: 0.2, 1: 0.4, 2: 0.4}, 0.3, dict(enumerate(LEVELS))),
    )


class Level(Enum):
    IC = "IC"
    MANAGER = "Manager"


QUESTIONS = {
    "function": Classify(["Sales", "Engineering"], "what function?"),
    "seniority": Classify(Level),
    "urgent": Check("needs a reply today", threshold=0.6),
    "severity": Rate(LEVELS, "how severe?"),
}


def test_ask_sends_every_question_in_one_request() -> None:
    fake = FakeJev(handler)
    out = ask("VP Sales", QUESTIONS, client=Client(client=fake))
    assert out == {"function": "Sales", "seniority": Level.MANAGER, "urgent": True, "severity": pytest.approx(1.4)}
    assert len(fake.calls) == 1
    q = fake.calls[0][1]
    assert q["function"].instructions == "what function?"
    assert list(q["seniority"].criteria) == ["IC", "Manager"]
    assert q["urgent"].type == "noul"
    assert list(q["severity"].criteria) == LEVELS


def test_ask_detail_returns_rich_objects() -> None:
    out = ask("VP Sales", QUESTIONS, detail=True, client=Client(client=FakeJev(handler)))
    assert isinstance(out["function"], Answer) and out["function"].shape == "sure"
    assert isinstance(out["urgent"], Feeling) and out["urgent"].p == pytest.approx(0.7)
    assert isinstance(out["severity"], Rating) and out["severity"].level == "mid"


def test_ask_list_and_series() -> None:
    fake = FakeJev(handler)
    jev = Client(client=fake, max_workers=1)
    rows = ask(["VP Sales", "Staff Engineer", "VP Sales"], QUESTIONS, client=jev)
    assert [r["function"] for r in rows] == ["Sales", "Engineering", "Sales"]
    assert len(fake.calls) == 2

    pd = pytest.importorskip("pandas")
    titles = pd.Series(["VP Sales", "Staff Engineer"], index=[7, 9])
    frame = ask(titles, QUESTIONS, client=jev)
    assert list(frame.columns) == ["function", "seniority", "urgent", "severity"]
    assert list(frame.index) == [7, 9]
    assert frame.loc[9, "function"] == "Engineering"
    assert len(fake.calls) == 2  # cache


def test_ask_validation() -> None:
    jev = Client(client=FakeJev(handler))
    with pytest.raises(HunchError, match="at least one question"):
        ask("x", {}, client=jev)
    with pytest.raises(HunchError, match="Classify, Rate, or Check"):
        ask("x", {"q": "not a spec"}, client=jev)
    with pytest.raises(HunchError, match="at least one label"):
        ask("x", {"q": Classify([])}, client=jev)
