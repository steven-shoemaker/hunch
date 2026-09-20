"""ask / classify / score / check / where / pick / rank. Lists, Series, or DataFrames in; same shape out."""

from __future__ import annotations

import asyncio
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from enum import Enum
from typing import Any, Callable, TypeVar

from typesafe_sdk import Choice, Noul, Score

from hunch import engine
from hunch.answer import Answer, Feeling, MultiAnswer, Pick, Ranked, Rating
from hunch.client import Client, resolve
from hunch.exceptions import HunchError

T = TypeVar("T")

MAX_CHOICE_OPTIONS = 255
MIN_SCORE_LEVELS, MAX_SCORE_LEVELS = 2, 10


# ----------------------------------------------------------------------------- containers


@dataclass
class Box:
    """The caller's data, unpacked: items to judge plus how to pack answers back."""

    items: list[Any]
    kind: str  # "single" | "list" | "series" | "frame"
    index: Any = None

    @property
    def pandas(self) -> bool:
        return self.kind in ("series", "frame")

    def out(self, rows: list[dict[str, Any]], *, detail: bool, squeeze: bool) -> Any:
        """rows: one {name: rich answer} per item. Pack them the way the caller sent data.

        Non-pandas: rich objects when detail, else bare values; squeeze drops the dict
        for single-question verbs. Pandas: a DataFrame on the caller's index; detail
        spreads each answer into name, name_p / name_level, name_confidence, name_shape;
        squeeze returns the lone column as a Series.
        """
        if not self.pandas:
            vals: list[Any] = [{k: (v if detail else bare(v)) for k, v in r.items()} for r in rows]
            if squeeze:
                vals = [next(iter(v.values())) for v in vals]
            return vals[0] if self.kind == "single" else vals
        import pandas as pd

        flat = [
            {col: val for k, v in r.items() for col, val in (flatten(k, v) if detail else {k: bare(v)}).items()}
            for r in rows
        ]
        frame = pd.DataFrame(flat, index=self.index)
        if squeeze and not detail:
            return frame.iloc[:, 0]
        return frame


def box(data: Any) -> Box:
    module = type(data).__module__.split(".")[0]
    if module == "pandas":
        if hasattr(data, "columns"):
            clean = data.astype(object).where(data.notna(), None)
            return Box(clean.to_dict("records"), "frame", data.index)
        if hasattr(data, "tolist"):
            return Box(list(data.tolist()), "series", data.index)
    if isinstance(data, (list, tuple)):
        return Box(list(data), "list")
    return Box([data], "single")


def bare(value: Any) -> Any:
    if isinstance(value, Answer):
        return value.label
    if isinstance(value, Rating):
        return value.score
    if isinstance(value, Feeling):
        return bool(value)
    if isinstance(value, MultiAnswer):
        return list(value.labels)
    return value


def flatten(name: str, value: Any) -> dict[str, Any]:
    if isinstance(value, Answer):
        return {name: value.label, f"{name}_p": value.p, f"{name}_confidence": value.confidence, f"{name}_shape": value.shape}
    if isinstance(value, Rating):
        return {name: value.score, f"{name}_level": value.level, f"{name}_confidence": value.confidence, f"{name}_shape": value.shape}
    if isinstance(value, Feeling):
        return {name: bool(value), f"{name}_p": value.p}
    if isinstance(value, MultiAnswer):
        return {name: list(value.labels), f"{name}_p": dict(value.probabilities)}
    return {name: value}


# ----------------------------------------------------------------------------- ask


@dataclass(frozen=True)
class Classify:
    """One Choice question for ask(): same arguments as classify()."""

    labels: Any
    instructions: str | None = None


@dataclass(frozen=True)
class Rate:
    """One Score question for ask(): same arguments as score()."""

    levels: Sequence[str]
    instructions: str | None = None


@dataclass(frozen=True)
class Check:
    """One Noul question for ask(): same arguments as check()."""

    statement: str
    criteria: Mapping[str, str | None] | None = None
    threshold: float = 0.5


Spec = Classify | Rate | Check


def ask(
    data: Any,
    questions: Mapping[str, Spec],
    *,
    context: Any = None,
    detail: bool = False,
    client: Client | None = None,
) -> Any:
    """Ask several questions about the same data in one Jev request per item.

    questions maps a name to Classify(...), Rate(...), or Check(...). One item returns a
    dict of name -> answer; a list returns a list of dicts; a Series or DataFrame returns
    a DataFrame with one column per question on the same index (detail=True spreads each
    answer into several columns).
    """
    if not questions:
        raise HunchError("ask() needs at least one question.")
    jev = resolve(client)
    data = box(data)
    built: dict[str, Any] = {}
    readers: dict[str, Callable[[dict[str, Any]], Any]] = {}
    for name, spec in questions.items():
        name = str(name)
        if isinstance(spec, Classify):
            keys, describe, back = _labels(spec.labels)
            if not keys:
                raise HunchError(f"ask() question {name!r} needs at least one label.")
            if len(keys) > MAX_CHOICE_OPTIONS:
                raise HunchError(f"ask() question {name!r} takes at most {MAX_CHOICE_OPTIONS} labels.")
            built[name] = Choice(
                instructions=spec.instructions or "Which label best describes the input?",
                criteria={key: describe.get(key) for key in keys},
            )
            readers[name] = lambda raw, back=back: _answer(jev, raw, back)
        elif isinstance(spec, Rate):
            built[name] = Score(
                instructions=spec.instructions or "Where on this scale does the input fall?",
                criteria=_levels(spec.levels),
            )
            readers[name] = lambda raw: _rating(jev, raw)
        elif isinstance(spec, Check):
            crit = None
            if spec.criteria:
                crit = {"true": spec.criteria.get("true"), "false": spec.criteria.get("false")}
            built[name] = Noul(instructions=spec.statement, criteria=crit)
            readers[name] = lambda raw, t=spec.threshold: Feeling(float(raw["noul"]), t)
        else:
            raise HunchError(f"ask() question {name!r} must be Classify, Rate, or Check.")
    states = [engine.build_state(item, context) for item in data.items]
    raws = engine.run(jev, states, built, label="ask")
    rows = [{name: readers[name](raw[name]) for name in built} for raw in raws]
    return data.out(rows, detail=detail, squeeze=False)


# ----------------------------------------------------------------------------- classify


def classify(
    data: Any,
    labels: Any,
    *,
    multi_label: bool = False,
    instructions: str | None = None,
    context: Any = None,
    threshold: float = 0.5,
    detail: bool = False,
    client: Client | None = None,
) -> Any:
    """Assign data to one of the labels (or several with multi_label=True).

    labels: a sequence of strings, an Enum class, or a mapping label -> description.
    Returns the label (an Enum member when labels is an Enum). detail=True returns
    Answer / MultiAnswer with the full distribution.
    """
    jev = resolve(client)
    data = box(data)
    keys, describe, back = _labels(labels)
    if not keys:
        raise HunchError("classify() needs at least one label.")
    states = [engine.build_state(item, context) for item in data.items]

    if multi_label:
        questions = {
            f"l{i}": Noul(
                instructions=_object(
                    question="Does this label apply to the input?",
                    label=key,
                    definition=describe.get(key),
                    guidance=instructions,
                ),
            )
            for i, key in enumerate(keys)
        }
        raws = engine.run(jev, states, questions, label="classify")

        def multi(raw: dict[str, Any]) -> MultiAnswer:
            probs = {key: float(raw[f"l{i}"]["noul"]) for i, key in enumerate(keys)}
            chosen = [back(k) for k, p in sorted(probs.items(), key=lambda kv: -kv[1]) if p >= threshold]
            return MultiAnswer(chosen, probs, threshold)

        return data.out([{"labels": multi(raw)} for raw in raws], detail=detail, squeeze=True)

    if len(keys) > MAX_CHOICE_OPTIONS:
        raise HunchError(f"classify() takes at most {MAX_CHOICE_OPTIONS} labels.")
    question = Choice(
        instructions=instructions or "Which label best describes the input?",
        criteria={key: describe.get(key) for key in keys},
    )
    raws = engine.run(jev, states, {"q": question}, label="classify")
    return data.out([{"label": _answer(jev, raw["q"], back)} for raw in raws], detail=detail, squeeze=True)


# ----------------------------------------------------------------------------- score


def score(
    data: Any,
    levels: Sequence[str],
    *,
    instructions: str | Mapping[str, str] | None = None,
    context: Any = None,
    detail: bool = False,
    client: Client | None = None,
) -> Any:
    """Place data on an ordered scale. Returns the position 0 .. len(levels)-1.

    levels: 2–10 concrete situations, worst first. instructions may be a mapping of
    dimension name -> question to score several dimensions in one request; the result
    is then a dict per item. detail=True returns Rating(s).
    """
    jev = resolve(client)
    data = box(data)
    levels = _levels(levels)
    named = isinstance(instructions, Mapping)
    dims = _dims(instructions, default="Where on this scale does the input fall?", base="score")
    states = [engine.build_state(item, context) for item in data.items]
    questions = {name: Score(instructions=text, criteria=list(levels)) for name, text in dims.items()}
    raws = engine.run(jev, states, questions, label="score")
    rows = [{name: _rating(jev, raw[name]) for name in dims} for raw in raws]
    return data.out(rows, detail=detail, squeeze=not named)


# ----------------------------------------------------------------------------- check


def check(
    data: Any,
    statement: str | Mapping[str, str],
    *,
    criteria: Mapping[str, str | None] | None = None,
    context: Any = None,
    threshold: float = 0.5,
    detail: bool = False,
    client: Client | None = None,
) -> Any:
    """Does the statement hold for the data? Returns bool (P(yes) >= threshold).

    statement may be a mapping name -> statement to check several in one request.
    criteria={"true": ..., "false": ...} sharpens the boundary. detail=True returns Feeling(s).
    """
    jev = resolve(client)
    data = box(data)
    named = isinstance(statement, Mapping)
    dims = _dims(statement, default=None, base="check")
    crit = None
    if criteria:
        crit = {"true": criteria.get("true"), "false": criteria.get("false")}
    states = [engine.build_state(item, context) for item in data.items]
    questions = {name: Noul(instructions=text, criteria=crit) for name, text in dims.items()}
    raws = engine.run(jev, states, questions, label="check")
    rows = [{name: Feeling(float(raw[name]["noul"]), threshold) for name in dims} for raw in raws]
    return data.out(rows, detail=detail, squeeze=not named)


# ----------------------------------------------------------------------------- where


def where(
    data: Any,
    statement: str,
    *,
    columns: Sequence[str] | None = None,
    criteria: Mapping[str, str | None] | None = None,
    context: Any = None,
    threshold: float = 0.5,
    detail: bool = False,
    client: Client | None = None,
) -> Any:
    """Semantic filter: keep the rows (or items) for which the statement holds.

    Results come back sorted by how strongly they match. columns= limits what Jev
    reads from a DataFrame; the whole row is still returned. detail=True returns every
    row with `match` and `match_p` columns instead of filtering.
    """
    data_box = box(data)
    if data_box.kind == "single":
        raise HunchError("where() needs a list, Series, or DataFrame.")
    subject = data
    if columns is not None:
        if data_box.kind != "frame":
            raise HunchError("columns= only applies to a DataFrame.")
        subject = data[list(columns)]
    feelings = check(subject, statement, criteria=criteria, context=context, threshold=threshold, detail=True, client=client)
    if data_box.pandas:
        match, p = feelings["check"], feelings["check_p"]
        if detail:
            return data.assign(match=match, match_p=p)
        return data[match].loc[p[match].sort_values(ascending=False).index]
    if detail:
        return feelings
    kept = [(item, f) for item, f in zip(data_box.items, feelings) if f]
    return [item for item, _ in sorted(kept, key=lambda t: -t[1].p)]


# ----------------------------------------------------------------------------- pick


def pick(
    candidates: Any,
    instructions: str,
    *,
    context: Any = None,
    detail: bool = False,
    client: Client | None = None,
) -> Any:
    """Choose the single best candidate. Jev compares them head to head in one Choice.

    A list returns the winning item. A Series or DataFrame returns the winner's index
    label, so df.loc[winner] is the row. More than 255 candidates run as a tournament.
    detail=True returns Pick with every candidate's probability.
    """
    jev = resolve(client)
    data = box(candidates)
    if data.kind == "single":
        raise HunchError("pick() needs a list, Series, or DataFrame of candidates.")
    field = data.items
    if not field:
        raise HunchError("pick() needs at least one candidate.")
    if not instructions or not instructions.strip():
        raise HunchError("pick() needs instructions saying what 'best' means.")
    keys = list(data.index) if data.pandas else field
    state = engine.build_state({"task": instructions}, context)

    def heat(group: list[int]) -> Pick:
        if len(group) == 1:
            return Pick(group[0], [(group[0], 1.0)], 1.0, "sure")
        criteria = {f"c{i}": engine.jsonable(field[i]) for i in group}
        question = Choice(
            instructions={"task": instructions, "note": "Each option is one candidate."},
            criteria=criteria,
        )
        raw = engine.run(jev, [state], {"q": question}, label="pick")[0]["q"]
        probs = {cid: float(p) for cid, p in raw["probabilities"].items()}
        ranked = sorted(((int(cid[1:]), p) for cid, p in probs.items()), key=lambda t: t[1], reverse=True)
        confidence = float(raw["confidence"])
        return Pick(int(raw["choice"][1:]), ranked, confidence, jev.policy.classify(probs, confidence))

    result = _tournament(list(range(len(field))), heat)
    result = Pick(keys[result.winner], [(keys[i], p) for i, p in result.ranked], result.confidence, result.shape)
    return result if detail else result.winner


def _tournament(field: list[int], heat: Callable[[list[int]], Pick]) -> Pick:
    # ponytail: heats of 255 then a final; a bracket with seeding if fields get huge
    while len(field) > MAX_CHOICE_OPTIONS:
        field = [
            heat(field[i : i + MAX_CHOICE_OPTIONS]).winner
            for i in range(0, len(field), MAX_CHOICE_OPTIONS)
        ]
    return heat(field)


# ----------------------------------------------------------------------------- rank


def rank(
    candidates: Any,
    dimensions: str | Mapping[str, str],
    levels: Sequence[str],
    *,
    weights: Mapping[str, float] | None = None,
    context: Any = None,
    client: Client | None = None,
) -> Any:
    """Score every candidate on each dimension, weight, and sort best first.

    Composite = weighted mean of normalized (0–1) dimension scores. Weights default to 1.
    A list returns Ranked rows. A Series or DataFrame returns a DataFrame on the same
    index with `composite` and one column per dimension, sorted best first.
    """
    data = box(candidates)
    if data.kind == "single":
        raise HunchError("rank() needs a list, Series, or DataFrame of candidates.")
    dims = dimensions if isinstance(dimensions, Mapping) else {"score": dimensions}
    ratings = score(data.items, levels, instructions=dims, context=context, detail=True, client=client)
    names = list(dims)
    w = {name: float((weights or {}).get(name, 1.0)) for name in names}
    total = sum(w.values())
    if total <= 0:
        raise HunchError("rank() weights must sum to more than zero.")
    composites = [sum(w[n] * r[n].normalized for n in names) / total for r in ratings]
    if data.pandas:
        import pandas as pd

        frame = pd.DataFrame(
            [{"composite": c, **{n: r[n].score for n in names}} for c, r in zip(composites, ratings)],
            index=data.index,
        )
        return frame.sort_values("composite", ascending=False)
    rows = [Ranked(item, c, r) for item, c, r in zip(data.items, composites, ratings)]
    return sorted(rows, key=lambda row: row.composite, reverse=True)


# ----------------------------------------------------------------------------- async twins

# ponytail: to_thread twins; a real AsyncTypeSafeClient path if event-loop throughput matters


async def ask_async(*args: Any, **kwargs: Any) -> Any:
    return await asyncio.to_thread(ask, *args, **kwargs)


async def classify_async(*args: Any, **kwargs: Any) -> Any:
    return await asyncio.to_thread(classify, *args, **kwargs)


async def score_async(*args: Any, **kwargs: Any) -> Any:
    return await asyncio.to_thread(score, *args, **kwargs)


async def check_async(*args: Any, **kwargs: Any) -> Any:
    return await asyncio.to_thread(check, *args, **kwargs)


async def where_async(*args: Any, **kwargs: Any) -> Any:
    return await asyncio.to_thread(where, *args, **kwargs)


async def pick_async(*args: Any, **kwargs: Any) -> Any:
    return await asyncio.to_thread(pick, *args, **kwargs)


async def rank_async(*args: Any, **kwargs: Any) -> Any:
    return await asyncio.to_thread(rank, *args, **kwargs)


# ----------------------------------------------------------------------------- helpers


def _labels(labels: Any) -> tuple[list[str], dict[str, Any], Callable[[str], Any]]:
    """Normalize labels to (option keys, descriptions, key -> user label)."""
    if isinstance(labels, type) and issubclass(labels, Enum):
        members = {str(member.value): member for member in labels}
        return list(members), {}, lambda key: members[key]
    if isinstance(labels, Mapping):
        keys = [str(key) for key in labels]
        _unique(keys)
        return keys, {str(k): v for k, v in labels.items()}, lambda key: key
    if isinstance(labels, str):
        raise HunchError("labels must be a sequence, an Enum class, or a mapping, not a string.")
    keys = [str(label) for label in labels]
    _unique(keys)
    return keys, {}, lambda key: key


def _unique(keys: list[str]) -> None:
    if len(set(keys)) != len(keys):
        raise HunchError("Labels must be unique.")


def _levels(levels: Sequence[str]) -> list[str]:
    if isinstance(levels, str):
        raise HunchError("levels must be a sequence of level descriptions.")
    out = [str(level) for level in levels]
    if not MIN_SCORE_LEVELS <= len(out) <= MAX_SCORE_LEVELS:
        raise HunchError(f"levels needs {MIN_SCORE_LEVELS}–{MAX_SCORE_LEVELS} entries.")
    return out


def _dims(spec: Any, default: str | None, base: str) -> dict[str, str]:
    if isinstance(spec, Mapping):
        if not spec:
            raise HunchError("Need at least one dimension.")
        return {str(k): str(v) for k, v in spec.items()}
    if spec is None:
        if default is None:
            raise HunchError("A statement is required.")
        return {base: default}
    return {base: str(spec)}


def _object(**fields: Any) -> dict[str, Any]:
    return {k: v for k, v in fields.items() if v is not None}


def _answer(jev: Client, raw: dict[str, Any], back: Callable[[str], Any]) -> Answer:
    probs = {str(k): float(v) for k, v in raw["probabilities"].items()}
    confidence = float(raw["confidence"])
    return Answer(
        label=back(str(raw["choice"])),
        probabilities=probs,
        confidence=confidence,
        shape=jev.policy.classify(probs, confidence),
    )


def _rating(jev: Client, raw: dict[str, Any]) -> Rating:
    legend = {int(k): str(v) for k, v in raw["legend"].items()}
    probs = {int(k): float(v) for k, v in raw["probabilities"].items()}
    confidence = float(raw["confidence"])
    shape = jev.policy.classify({legend[k]: p for k, p in probs.items()}, confidence)
    return Rating(float(raw["score"]), confidence, legend, probs, shape)
