from __future__ import annotations

import pytest

from hunch import Client, classify, engine
from tests.fakes import FakeJev, choice_answer, response


def handler(state, questions):
    return response(q=choice_answer("a", {"a": 0.9, "b": 0.1}, 0.9))


def test_progress_counts_only_uncached_distinct_requests(monkeypatch) -> None:
    seen: list[tuple[int, str]] = []

    def fake_progress(client, total, label):
        seen.append((total, label))
        return lambda it: it

    monkeypatch.setattr(engine, "progress", fake_progress)
    jev = Client(client=FakeJev(handler), max_workers=1)
    classify(["x", "y", "x"], ["a", "b"], client=jev)
    classify(["x", "y", "z"], ["a", "b"], client=jev)
    assert seen == [(2, "classify"), (1, "classify")]  # dedupe, then cache


def test_auto_threshold_and_off() -> None:
    pytest.importorskip("tqdm")
    auto = Client(client=FakeJev(handler), progress="auto")
    it = iter([1])
    assert engine.progress(auto, 9, "x")(it) is it
    assert hasattr(engine.progress(auto, 10, "x")(iter([1])), "total")
    off = Client(client=FakeJev(handler), progress=False)
    it = iter([1])
    assert engine.progress(off, 100, "x")(it) is it
