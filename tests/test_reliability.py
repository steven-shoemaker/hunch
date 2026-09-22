"""Big-column behavior: partial failures, rate limits, dry runs, real async."""

from __future__ import annotations

import asyncio
import time
import warnings

import pytest

from hunch import Check, Classify, Client, HunchError, ask, classify, classify_async, dry_run, rank, where
from tests.fakes import FakeJev, choice_answer, noul_answer, response, score_answer

pd = pytest.importorskip("pandas")


def answer(state, questions):
    def one(q):
        if q.type == "choice":
            return choice_answer("a", {"a": 0.9, "b": 0.1}, 0.8)
        if q.type == "score":
            return score_answer(1.0, {1: 1.0}, 1.0, {0: "low", 1: "high"})
        return noul_answer(0.9)

    return response(**{qid: one(q) for qid, q in questions.items()})


def flaky(state, questions):
    if "boom" in str(state["input"]):
        raise RuntimeError("503 from Jev")
    return answer(state, questions)


def test_errors_raise_by_default() -> None:
    with pytest.raises(RuntimeError, match="503"):
        classify(["ok", "boom"], ["a", "b"], client=Client(client=FakeJev(flaky), max_workers=1))


def test_errors_skip_keeps_good_rows_and_warns() -> None:
    fake = FakeJev(flaky)
    jev = Client(client=fake, max_workers=1, errors="skip")
    with pytest.warns(UserWarning, match="1 of 3 requests failed"):
        out = classify(["ok", "boom", "fine"], ["a", "b"], client=jev)
    assert out == ["a", None, "a"]
    # the good answers were cached, so a retry only re-sends the failed row
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        classify(["ok", "boom", "fine"], ["a", "b"], client=jev)
    assert len(fake.calls) == 4


def test_skipped_rows_flow_through_frames_where_and_rank() -> None:
    jev = Client(client=FakeJev(flaky), max_workers=1, errors="skip")
    df = pd.DataFrame({"t": ["ok", "boom"]}, index=[1, 2])
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        detail = classify(df, ["a", "b"], detail=True, client=jev)
        assert pd.isna(detail.loc[2, "label"]) and detail.loc[1, "label_shape"] == "sure"
        assert list(where(df, "x", client=jev).index) == [1]
        ranked = rank(df, "good?", ["low", "high"], client=jev)
    assert ranked.index[-1] == 2 and pd.isna(ranked.loc[2, "composite"])  # failed row sorts last


def test_max_rps_spaces_requests() -> None:
    jev = Client(client=FakeJev(answer), max_workers=4, max_rps=20)
    start = time.monotonic()
    classify([f"t{i}" for i in range(6)], ["a", "b"], client=jev)
    assert time.monotonic() - start >= 5 / 20 * 0.9  # 6 requests at 20/s need ~0.25s
    with pytest.raises(HunchError, match="max_rps"):
        Client(client=FakeJev(answer), max_rps=0)


def test_dry_run_counts_without_sending() -> None:
    fake = FakeJev(answer)
    jev = Client(client=fake, max_workers=1)
    classify(["cached"], ["a", "b"], client=jev)
    with dry_run() as plan:
        out = classify(["cached", "new1", "new2", "new1"], ["a", "b"], client=jev)
        ask(["x"], {"f": Classify(["a", "b"]), "g": Check("y")}, client=jev)
    assert len(fake.calls) == 1  # nothing sent inside the block
    assert out == ["a", "a", "a", "a"]  # placeholders keep code running (cached row is real)
    assert plan.requests == 3 and plan.questions == 4 and plan.items == 5
    classify(["new1"], ["a", "b"], client=jev)
    assert len(fake.calls) == 2  # placeholders were not cached


def test_async_runs_requests_concurrently_on_the_loop() -> None:
    class SlowAsyncJev:
        def __init__(self) -> None:
            self.live = self.peak = self.calls = 0

        async def system_one(self, state=None, questions=None, **kw):
            self.live += 1
            self.peak = max(self.peak, self.live)
            self.calls += 1
            await asyncio.sleep(0.05)
            self.live -= 1
            return answer(state, questions)

    slow = SlowAsyncJev()
    jev = Client(client=FakeJev(answer), async_client=slow, max_workers=1, max_concurrency=10)
    out = asyncio.run(classify_async([f"t{i}" for i in range(20)], ["a", "b"], client=jev))
    assert out == ["a"] * 20 and slow.calls == 20
    assert slow.peak == 10  # concurrency is real and capped, despite max_workers=1


def test_async_without_async_client_falls_back_to_threads() -> None:
    fake = FakeJev(answer)
    out = asyncio.run(classify_async(["a", "b"], ["a", "b"], client=Client(client=fake)))
    assert out == ["a", "a"] and len(fake.calls) == 2
