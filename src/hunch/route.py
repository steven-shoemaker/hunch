"""route(): turn answers into outcomes with ordered rules. No requests; it reads answers you already have."""

from __future__ import annotations

from collections.abc import Callable, Mapping
from typing import Any

from hunch.answer import Answer, Feeling, MultiAnswer, Rating
from hunch.exceptions import HunchError

SHAPES = {"sure", "split", "unsure"}


def route(answers: Any, rules: Mapping[str, Mapping[str, Any]], *, default: Any = None) -> Any:
    """Pick an outcome for each row: the first rule whose conditions all hold, else default.

    answers: what ask(), classify(), check(), or score() returned, ideally with detail=True
    so probabilities are available; a DataFrame joined from those works too. rules maps an
    outcome to its conditions, checked in order:

        hunch.route(answers, {
            "page":    {"urgent": 0.8},                   # P(yes) (or P of the label, or the score) >= 0.8
            "billing": {"topic": ("billing", 0.7)},       # label is billing with P >= 0.7
            "refunds": {"topic": ["refund", "chargeback"]},  # label is one of these
            "review":  {"topic.shape": "unsure"},         # the answer's shape
            "vip":     {"tier": "enterprise", "angry": True},  # all conditions must hold
        }, default="triage")

    Names ending in .p, .shape, .level, or .score read that part of the answer. A condition
    may also be a function of the answer. A missing or skipped answer fails its condition.
    Returns one outcome, a list, or a Series on the same index.
    """
    if not rules:
        raise HunchError("route() needs at least one rule.")
    compiled = [(outcome, [(name, _test(name, cond)) for name, cond in conds.items()]) for outcome, conds in rules.items()]

    def one(row: Any) -> Any:
        for outcome, tests in compiled:
            if all(test(_get(row, name)) for name, test in tests):
                return outcome
        return default

    module = type(answers).__module__.split(".")[0]
    if module == "pandas" and hasattr(answers, "columns"):
        import pandas as pd

        return pd.Series([one(row) for _, row in answers.iterrows()], index=answers.index, name="route", dtype=object)
    if isinstance(answers, (list, tuple)) or (module == "pandas" and hasattr(answers, "tolist")):
        values = answers.tolist() if module == "pandas" else list(answers)
        out = [one(row if isinstance(row, Mapping) else {"_": row}) for row in values]
        if module == "pandas":
            import pandas as pd

            return pd.Series(out, index=answers.index, name="route", dtype=object)
        return out
    return one(answers if isinstance(answers, Mapping) else {"_": answers})


def _get(row: Any, name: str) -> Any:
    """Read name (or name.part) from a dict of answers or a DataFrame row."""
    base, _, part = name.partition(".")
    if part not in ("", "p", "shape", "level", "score", "label"):
        base, part = name, ""
    keys = row.keys() if hasattr(row, "keys") else []
    if base not in keys and "_" in keys and len(keys) == 1:  # a bare list of answers: rules name anything
        value = row["_"]
    elif part and f"{base}_{part}" in keys:  # DataFrame column like topic_p / topic_shape
        return _clean(row[f"{base}_{part}"])
    elif base in keys:
        value = row[base]
        if not part and f"{base}_p" in keys:  # a flattened detail column: keep label and P together
            return _Flat(_clean(value), _clean(row[f"{base}_p"]))
    else:
        return _Missing
    value = _clean(value)
    if value is None or not part:
        return value
    if isinstance(value, (Answer, Rating, Feeling, MultiAnswer)):
        if part == "label" and isinstance(value, Answer):
            return value.label
        return getattr(value, part, _Missing)
    return _Missing


class _MissingType:
    pass


class _Flat:
    """A label and its probability read from two DataFrame columns."""

    def __init__(self, label: Any, p: Any) -> None:
        self.label, self.p = label, p


_Missing = _MissingType()


def _clean(value: Any) -> Any:
    try:
        return None if value is None or value != value else value  # NaN from a DataFrame
    except (TypeError, ValueError):
        return value


def _label(value: Any) -> Any:
    if isinstance(value, _Flat):
        return value.label
    if isinstance(value, Answer):
        return value.label
    if isinstance(value, Feeling):
        return value.value
    return value


def _prob(value: Any) -> float | None:
    """The number a threshold compares against."""
    if isinstance(value, _Flat):
        return None if value.p is None else float(value.p)
    if isinstance(value, (Answer, Feeling)):
        return value.p
    if isinstance(value, Rating):
        return value.score
    if isinstance(value, bool):
        return 1.0 if value else 0.0
    if isinstance(value, (int, float)):
        return float(value)
    return None


def _same(a: Any, b: Any) -> bool:
    return a == b or str(getattr(a, "value", a)) == str(getattr(b, "value", b))


def _test(name: str, cond: Any) -> Callable[[Any], bool]:
    def ok(value: Any) -> bool:
        return value is not _Missing and value is not None and not (isinstance(value, _Flat) and value.label is None)

    if callable(cond) and not isinstance(cond, type):
        return lambda v: ok(v) and bool(cond(v))
    if isinstance(cond, bool):
        return lambda v: ok(v) and (_label(v) is cond or (isinstance(v, MultiAnswer) and bool(v.labels) is cond))
    if isinstance(cond, (int, float)):
        return lambda v: ok(v) and (p := _prob(v)) is not None and p >= cond
    if isinstance(cond, tuple):
        if len(cond) != 2 or not isinstance(cond[1], (int, float)):
            raise HunchError(f"route() condition for {name!r}: a tuple is (label, minimum probability).")
        label, floor = cond
        return lambda v: ok(v) and _same(_label(v), label) and (_prob(v) or 0.0) >= floor
    if isinstance(cond, (list, set, frozenset)):
        options = list(cond)
        return lambda v: ok(v) and (
            any(_same(label, o) for o in options for label in v.labels) if isinstance(v, MultiAnswer)
            else any(_same(_label(v), o) for o in options)
        )
    # a single label (or shape, when the name ends in .shape)
    return lambda v: ok(v) and (
        any(_same(label, cond) for label in v.labels) if isinstance(v, MultiAnswer) else _same(_label(v), cond)
    )
