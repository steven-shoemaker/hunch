from __future__ import annotations

from hunch import ask, connect, rate
from tests.fakes import FakeJev, choice_answer, response, score_answer

LEGEND = {0: "disposable", 1: "nice to have", 2: "must keep"}


def test_rate_batches_with_ask() -> None:
    fake = FakeJev(
        lambda state, questions: response(
            q1=choice_answer("junk", {"junk": 0.91, "keep": 0.09}, 0.82),
            q2=score_answer(0.2, {0: 0.85, 1: 0.1, 2: 0.05}, 0.8, LEGEND),
        )
    )
    jev = connect(client=fake)
    with jev.session():
        folder = ask("old.dmg", "folder?", among=["junk", "keep"])
        keep = rate(
            "old.dmg",
            "How important is it to keep this?",
            ["disposable", "nice to have", "must keep"],
        )
        assert folder.top == "junk"
        assert keep.level == "disposable"
        assert keep.score == 0.2
        assert keep.shape == "sure"
    assert len(fake.calls) == 1
    assert jev.usage.calls == 1
    assert jev.usage.input_tokens == 10
    assert jev.usage.model == "jev-test"


def test_usage_counts_cache_hits() -> None:
    fake = FakeJev(
        lambda state, questions: response(
            q1=choice_answer("Sales", {"Sales": 0.95, "Other": 0.05}, 0.9)
        )
    )
    jev = connect(client=fake)
    with jev.session():
        assert ask("VP", "function?", among=["Sales", "Other"]).top == "Sales"
    with jev.session():
        assert ask("VP", "function?", among=["Sales", "Other"]).top == "Sales"
    assert jev.usage.calls == 1
    assert jev.usage.hits == 1
