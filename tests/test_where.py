from __future__ import annotations

import pytest

from hunch import Client, HunchError, where
from tests.fakes import FakeJev, noul_answer, response

pd = pytest.importorskip("pandas")


def cats(state, questions):
    row = state["input"]
    text = " ".join(str(v) for v in row.values()) if isinstance(row, dict) else str(row)
    p = 0.95 if "cat" in text else (0.6 if "purr" in text else 0.1)
    return response(check=noul_answer(p))


def frame():
    return pd.DataFrame(
        {"name": ["Amanda", "Dave", "Elena"], "age": [28, 34, 33], "bio": ["I have two cats.", "TV after work.", "I purr when happy."]},
        index=[1, 2, 3],
    )


def test_where_filters_and_sorts_by_match() -> None:
    jev = Client(client=FakeJev(cats), max_workers=1)
    out = where(frame(), "probably likes cats", client=jev)
    assert list(out.index) == [1, 3]  # Dave dropped, strongest match first
    assert list(out.columns) == ["name", "age", "bio"]  # whole rows, no extra columns
    strict = where(frame(), "probably likes cats", threshold=0.9, client=jev)
    assert list(strict.index) == [1]


def test_where_columns_limits_what_jev_reads_but_returns_full_rows() -> None:
    fake = FakeJev(cats)
    out = where(frame(), "probably likes cats", columns=["bio"], client=Client(client=fake, max_workers=1))
    assert all(set(s["input"]) == {"bio"} for s, _ in fake.calls)
    assert list(out.columns) == ["name", "age", "bio"]


def test_where_detail_keeps_every_row_with_match_columns() -> None:
    out = where(frame(), "probably likes cats", detail=True, client=Client(client=FakeJev(cats), max_workers=1))
    assert list(out.columns) == ["name", "age", "bio", "match", "match_p"]
    assert list(out["match"]) == [True, False, True]


def test_where_on_list_and_validation() -> None:
    jev = Client(client=FakeJev(cats), max_workers=1)
    assert where(["dog person", "cat lady", "purr"], "likes cats", client=jev) == ["cat lady", "purr"]
    with pytest.raises(HunchError, match="list, Series, or DataFrame"):
        where("one thing", "likes cats", client=jev)
    with pytest.raises(HunchError, match="columns="):
        where(["a"], "x", columns=["a"], client=jev)


def test_accessor() -> None:
    import hunch  # noqa: F401  (registers df.hunch)

    jev = Client(client=FakeJev(cats), max_workers=1)
    df = frame()
    assert list(df.hunch.where("probably likes cats", client=jev).index) == [1, 3]
    assert list(df["bio"].hunch.check("likes cats", client=jev)) == [True, False, True]
