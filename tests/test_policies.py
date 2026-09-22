from __future__ import annotations

import pytest

from hunch import Classify, Client, HunchError, ask, classify
from tests.fakes import FakeJev, choice_answer, response

LEVELS = ["IC", "Manager", "Director"]


def handler(state, questions):
    title = state["input"]
    q = questions["q"]
    if len(q.criteria) == 2:  # rematch between the top two
        return response(q=choice_answer("Director", {"Manager": 0.15, "Director": 0.85}, 0.8))
    if title == "Head of Sales":
        return response(q=choice_answer("Manager", {"IC": 0.04, "Manager": 0.50, "Director": 0.46}, 0.5))
    if title == "Consultant":
        return response(q=choice_answer("IC", {"IC": 0.36, "Manager": 0.33, "Director": 0.31}, 0.05))
    return response(q=choice_answer("IC", {"IC": 0.95, "Manager": 0.03, "Director": 0.02}, 0.95))


TITLES = ["Staff Engineer", "Head of Sales", "Consultant", "Head of Sales"]


def test_default_keeps_first_answer() -> None:
    fake = FakeJev(handler)
    out = classify(TITLES, LEVELS, client=Client(client=fake, max_workers=1))
    assert out == ["IC", "Manager", "IC", "Manager"]
    assert len(fake.calls) == 3


def test_rematch_only_for_split_rows_and_literal_for_unsure() -> None:
    fake = FakeJev(handler)
    out = classify(TITLES, LEVELS, split="rematch", unsure="review", client=Client(client=fake, max_workers=1))
    assert out == ["IC", "Director", "review", "Director"]
    assert len(fake.calls) == 4  # 3 first-pass + 1 rematch (the duplicate split row is deduped)
    rematch = fake.calls[-1][1]["q"]
    assert list(rematch.criteria) == ["Manager", "Director"]


def test_literal_split_and_detail_rules() -> None:
    fake = FakeJev(handler)
    jev = Client(client=fake, max_workers=1)
    assert classify(TITLES, LEVELS, split="tie", client=jev) == ["IC", "tie", "IC", "tie"]
    rich = classify("Head of Sales", LEVELS, split="rematch", detail=True, client=jev)
    assert rich.label == "Director" and rich.shape == "sure"
    with pytest.raises(HunchError, match="detail=False"):
        classify("Head of Sales", LEVELS, unsure="review", detail=True, client=jev)


def test_policies_inside_ask() -> None:
    fake = FakeJev(handler)
    out = ask(TITLES, {"q": Classify(LEVELS, split="rematch", unsure="review")}, client=Client(client=fake, max_workers=1))
    assert [r["q"] for r in out] == ["IC", "Director", "review", "Director"]
