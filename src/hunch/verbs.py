"""classify / score / check / pick / rank. Lists in, lists out. Jev decides."""

from __future__ import annotations

import asyncio
from collections.abc import Mapping, Sequence
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
    items, wrap = _items(data)
    keys, describe, back = _labels(labels)
    if not keys:
        raise HunchError("classify() needs at least one label.")
    states = [engine.build_state(item, context) for item in items]

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
        raws = engine.run(jev, states, questions)

        def multi(raw: dict[str, Any]) -> Any:
            probs = {key: float(raw[f"l{i}"]["noul"]) for i, key in enumerate(keys)}
            chosen = [back(k) for k, p in sorted(probs.items(), key=lambda kv: -kv[1]) if p >= threshold]
            return MultiAnswer(chosen, probs, threshold) if detail else chosen

        return wrap([multi(raw) for raw in raws])

    if len(keys) > MAX_CHOICE_OPTIONS:
        raise HunchError(f"classify() takes at most {MAX_CHOICE_OPTIONS} labels.")
    question = Choice(
        instructions=instructions or "Which label best describes the input?",
        criteria={key: describe.get(key) for key in keys},
    )
    raws = engine.run(jev, states, {"q": question})

    def single(raw: dict[str, Any]) -> Any:
        answer = _answer(jev, raw["q"], back)
        return answer if detail else answer.label

    return wrap([single(raw) for raw in raws])


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
    items, wrap = _items(data)
    levels = _levels(levels)
    dims = _dims(instructions, default="Where on this scale does the input fall?")
    states = [engine.build_state(item, context) for item in items]
    questions = {name: Score(instructions=text, criteria=list(levels)) for name, text in dims.items()}
    raws = engine.run(jev, states, questions)

    def one(raw: dict[str, Any]) -> Any:
        ratings = {name: _rating(jev, raw[name]) for name in dims}
        out = ratings if detail else {name: r.score for name, r in ratings.items()}
        return out if isinstance(instructions, Mapping) else next(iter(out.values()))

    return wrap([one(raw) for raw in raws])


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
    items, wrap = _items(data)
    dims = _dims(statement, default=None)
    crit = None
    if criteria:
        crit = {"true": criteria.get("true"), "false": criteria.get("false")}
    states = [engine.build_state(item, context) for item in items]
    questions = {name: Noul(instructions=text, criteria=crit) for name, text in dims.items()}
    raws = engine.run(jev, states, questions)

    def one(raw: dict[str, Any]) -> Any:
        feelings = {name: Feeling(float(raw[name]["noul"]), threshold) for name in dims}
        out = feelings if detail else {name: bool(f) for name, f in feelings.items()}
        return out if isinstance(statement, Mapping) else next(iter(out.values()))

    return wrap([one(raw) for raw in raws])


# ----------------------------------------------------------------------------- pick


def pick(
    candidates: Sequence[Any],
    instructions: str,
    *,
    context: Any = None,
    detail: bool = False,
    client: Client | None = None,
) -> Any:
    """Choose the single best candidate. Jev compares them head to head in one Choice.

    More than 255 candidates run as a tournament: heats of 255, then a final.
    detail=True returns Pick with every candidate's probability.
    """
    jev = resolve(client)
    field = list(candidates)
    if not field:
        raise HunchError("pick() needs at least one candidate.")
    if not instructions or not instructions.strip():
        raise HunchError("pick() needs instructions saying what 'best' means.")
    state = engine.build_state({"task": instructions}, context)

    def heat(group: list[Any]) -> Pick:
        if len(group) == 1:
            return Pick(group[0], [(group[0], 1.0)], 1.0, "sure")
        criteria = {f"c{i}": engine.jsonable(item) for i, item in enumerate(group)}
        question = Choice(
            instructions={"task": instructions, "note": "Each option is one candidate."},
            criteria=criteria,
        )
        raw = engine.run(jev, [state], {"q": question})[0]["q"]
        probs = {cid: float(p) for cid, p in raw["probabilities"].items()}
        ranked = sorted(
            ((group[int(cid[1:])], p) for cid, p in probs.items()),
            key=lambda pair: pair[1],
            reverse=True,
        )
        winner = group[int(raw["choice"][1:])]
        confidence = float(raw["confidence"])
        return Pick(winner, ranked, confidence, jev.policy.classify(probs, confidence))

    result = _tournament(field, heat)
    return result if detail else result.winner


def _tournament(field: list[Any], heat: Callable[[list[Any]], Pick]) -> Pick:
    # ponytail: heats of 255 then a final; a bracket with seeding if fields get huge
    while len(field) > MAX_CHOICE_OPTIONS:
        field = [
            heat(field[i : i + MAX_CHOICE_OPTIONS]).winner
            for i in range(0, len(field), MAX_CHOICE_OPTIONS)
        ]
    return heat(field)


# ----------------------------------------------------------------------------- rank


def rank(
    candidates: Sequence[Any],
    dimensions: str | Mapping[str, str],
    levels: Sequence[str],
    *,
    weights: Mapping[str, float] | None = None,
    context: Any = None,
    client: Client | None = None,
) -> list[Ranked]:
    """Score every candidate on each dimension, weight, and sort best first.

    Composite = weighted mean of normalized (0–1) dimension scores. Weights default to 1.
    Every candidate's dimensions go in one request; candidates run in parallel.
    """
    ratings = score(
        list(candidates),
        levels,
        instructions=dimensions if isinstance(dimensions, Mapping) else {"score": dimensions},
        context=context,
        detail=True,
        client=client,
    )
    names = list(ratings[0]) if ratings else []
    w = {name: float((weights or {}).get(name, 1.0)) for name in names}
    total = sum(w.values())
    if total <= 0:
        raise HunchError("rank() weights must sum to more than zero.")
    rows = [
        Ranked(item, sum(w[n] * r[n].normalized for n in names) / total, r)
        for item, r in zip(candidates, ratings)
    ]
    return sorted(rows, key=lambda row: row.composite, reverse=True)


# ----------------------------------------------------------------------------- async twins

# ponytail: to_thread twins; a real AsyncTypeSafeClient path if event-loop throughput matters


async def classify_async(*args: Any, **kwargs: Any) -> Any:
    return await asyncio.to_thread(classify, *args, **kwargs)


async def score_async(*args: Any, **kwargs: Any) -> Any:
    return await asyncio.to_thread(score, *args, **kwargs)


async def check_async(*args: Any, **kwargs: Any) -> Any:
    return await asyncio.to_thread(check, *args, **kwargs)


async def pick_async(*args: Any, **kwargs: Any) -> Any:
    return await asyncio.to_thread(pick, *args, **kwargs)


async def rank_async(*args: Any, **kwargs: Any) -> Any:
    return await asyncio.to_thread(rank, *args, **kwargs)


# ----------------------------------------------------------------------------- helpers


def _items(data: Any) -> tuple[list[Any], Callable[[list[Any]], Any]]:
    """Split data into items plus a function that rebuilds the caller's container."""
    module = type(data).__module__.split(".")[0]
    if module == "pandas":
        if not hasattr(data, "tolist"):
            raise HunchError("Pass a single column (a Series), not a DataFrame.")
        index = data.index
        return list(data.tolist()), lambda out: type(data)(out, index=index, dtype=object)
    if isinstance(data, (list, tuple)):
        return list(data), lambda out: out
    return [data], lambda out: out[0]


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


def _dims(spec: Any, default: str | None) -> dict[str, str]:
    if isinstance(spec, Mapping):
        if not spec:
            raise HunchError("Need at least one dimension.")
        return {str(k): str(v) for k, v in spec.items()}
    if spec is None:
        if default is None:
            raise HunchError("A statement is required.")
        return {"q": default}
    return {"q": str(spec)}


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
