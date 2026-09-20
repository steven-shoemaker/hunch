"""DataFrames and Series in; DataFrames and Series out, on the caller's index."""

from __future__ import annotations

import pytest

from hunch import Check, Classify, Client, ask, check, classify, pick, rank, score
from tests.fakes import FakeJev, choice_answer, noul_answer, response, score_answer

pd = pytest.importorskip("pandas")
LEVELS = ["low", "mid", "high"]


def handler(state, questions):
    row = state["input"]
    name = row["name"] if isinstance(row, dict) else row
    answers = {}
    for qid, q in questions.items():
        if q.type == "choice":
            pick_ = "Sales" if "Sales" in name else "Engineering"
            answers[qid] = choice_answer(pick_, {"Sales": 0.9, "Engineering": 0.1} if pick_ == "Sales" else {"Sales": 0.2, "Engineering": 0.8}, 0.8)
        elif q.type == "noul":
            answers[qid] = noul_answer(0.9 if "Sales" in name else 0.1)
        else:
            v = 2.0 if "Sales" in name else 0.5
            answers[qid] = score_answer(v, {int(v): 1.0}, 0.9, dict(enumerate(LEVELS)))
    return response(**answers)


def frame():
    return pd.DataFrame({"name": ["VP Sales", "Staff Engineer"], "age": [40, None]}, index=[10, 20])


def test_dataframe_rows_are_the_state_and_nan_becomes_null() -> None:
    fake = FakeJev(handler)
    out = classify(frame(), ["Sales", "Engineering"], client=Client(client=fake, max_workers=1))
    assert list(out) == ["Sales", "Engineering"] and list(out.index) == [10, 20]
    states = sorted((s["input"] for s, _ in fake.calls), key=lambda r: r["name"])
    assert states == [{"name": "Staff Engineer", "age": None}, {"name": "VP Sales", "age": 40}]


def test_detail_on_pandas_spreads_into_columns() -> None:
    jev = Client(client=FakeJev(handler), max_workers=1)
    fit = score(frame(), LEVELS, instructions="fit?", detail=True, client=jev)
    assert list(fit.columns) == ["score", "score_level", "score_confidence", "score_shape"]
    assert fit.loc[10, "score_level"] == "high"
    ok = check(frame()["name"], "is sales", detail=True, client=jev)
    assert list(ok.columns) == ["check", "check_p"]
    assert bool(ok.loc[10, "check"]) is True
    both = ask(frame(), {"fn": Classify(["Sales", "Engineering"]), "hot": Check("is sales")}, detail=True, client=jev)
    assert list(both.columns) == ["fn", "fn_p", "fn_confidence", "fn_shape", "hot", "hot_p"]
    plain = ask(frame(), {"fn": Classify(["Sales", "Engineering"]), "hot": Check("is sales")}, client=jev)
    assert list(plain.columns) == ["fn", "hot"]
    joined = frame().join(plain)
    assert joined.loc[10, "fn"] == "Sales"


def test_multi_label_on_series_gives_lists() -> None:
    jev = Client(client=FakeJev(handler), max_workers=1)
    out = classify(frame()["name"], ["Sales", "Engineering"], multi_label=True, client=jev)
    assert out.loc[10] == ["Sales", "Engineering"] or out.loc[10] == ["Sales"]
    assert isinstance(out.loc[20], list)


def test_pick_on_frame_returns_index_label() -> None:
    def best_is_sales(state, questions):
        q = questions["q"]
        ids = list(q.criteria)
        text = lambda c: c["name"] if isinstance(c, dict) else c
        winner = next(cid for cid in ids if "Sales" in text(q.criteria[cid]))
        probs = {cid: (0.8 if cid == winner else 0.2) for cid in ids}
        return response(q=choice_answer(winner, probs, 0.8))

    jev = Client(client=FakeJev(best_is_sales))
    df = frame()
    best = pick(df, "the sales person", client=jev)
    assert best == 10
    assert df.loc[best, "name"] == "VP Sales"
    rich = pick(df["name"], "the sales person", detail=True, client=jev)
    assert rich.winner == 10 and rich.ranked[0][0] == 10
    with pytest.raises(Exception, match="list, Series, or DataFrame"):
        pick("just one", "best", client=jev)


def test_rank_on_frame_returns_sorted_frame() -> None:
    jev = Client(client=FakeJev(handler), max_workers=1)
    out = rank(frame(), {"a": "a?", "b": "b?"}, LEVELS, client=jev)
    assert list(out.columns) == ["composite", "a", "b"]
    assert list(out.index) == [10, 20]
    assert out.loc[10, "composite"] == pytest.approx(1.0)
    assert out.loc[20, "composite"] == pytest.approx(0.25)
