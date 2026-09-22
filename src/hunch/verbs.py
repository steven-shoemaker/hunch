"""ask / classify / score / check / where / pick / rank. Lists, Series, or DataFrames in; same shape out."""

from __future__ import annotations

import asyncio
import json
from collections.abc import Hashable, Mapping, Sequence
from dataclasses import dataclass
from enum import Enum
from typing import TYPE_CHECKING, Any, Callable, TypeVar, overload

from typesafe_sdk import Choice, Noul, Score

from hunch import engine
from hunch.answer import Answer, Feeling, MultiAnswer, Pick, Ranked, Rating
from hunch.client import Client, resolve
from hunch.exceptions import HunchError

if TYPE_CHECKING:
    import pandas as pd

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
        """rows: one {name: rich answer or None} per item. Pack them the way the caller sent data.

        Non-pandas: rich objects when detail, else bare values; squeeze drops the dict
        for single-question verbs. Pandas: a DataFrame on the caller's index; detail
        spreads each answer into name, name_p / name_level, name_confidence, name_shape;
        squeeze returns the lone column as a Series. Skipped requests are None.
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
        for col in [c for c in frame.columns if str(c).endswith("_by")]:
            frame[col] = frame[col].fillna("jev")
        if squeeze and not detail:
            return frame.iloc[:, 0]
        return frame


def box(data: Any, columns: Sequence[str] | None = None) -> Box:
    """Unpack data. columns= narrows a DataFrame to what Jev should read."""
    module = type(data).__module__.split(".")[0]
    if columns is not None:
        if module != "pandas" or not hasattr(data, "columns"):
            raise HunchError("columns= only applies to a DataFrame.")
        missing = [c for c in columns if c not in data.columns]
        if missing:
            raise HunchError(f"columns= names columns the frame doesn't have: {missing}")
        data = data[list(columns)]
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
        return value.value
    if isinstance(value, MultiAnswer):
        return list(value.labels)
    return value


def flatten(name: str, value: Any) -> dict[str, Any]:
    if isinstance(value, Answer):
        out = {name: value.label, f"{name}_p": value.p, f"{name}_confidence": value.confidence, f"{name}_shape": value.shape}
        if value.by is not None:
            out[f"{name}_by"] = value.by
        return out
    if isinstance(value, Rating):
        return {name: value.score, f"{name}_level": value.level, f"{name}_confidence": value.confidence, f"{name}_shape": value.shape}
    if isinstance(value, Feeling):
        return {name: value.value, f"{name}_p": value.p}
    if isinstance(value, MultiAnswer):
        return {name: list(value.labels), f"{name}_p": dict(value.probabilities)}
    return {name: value}


def _skip(read: Callable[[Any], T]) -> Callable[[Any], T | None]:
    """Readers get None for requests skipped under errors="skip"."""
    return lambda raw: None if raw is None else read(raw)


def merge_context(base: Any, extra: Any) -> Any:
    """Call-level context plus question-level context; the question's keys win."""
    if extra is None:
        return base
    if base is None:
        return extra
    left = base if isinstance(base, Mapping) else {"context": base}
    right = extra if isinstance(extra, Mapping) else {"context": extra}
    return {**left, **right}


# ----------------------------------------------------------------------------- pairs


class Tree(dict):
    """A label tree for classify(): {"Parent": {"Child": description or None, ...}, ...}.

    Marked explicitly because a plain dict's values may be object descriptions, not subtrees.
    """



def pairs(a: Any, b: Any, names: tuple[str, str] = ("a", "b")) -> Any:
    """Line up two aligned sequences so any verb can compare them row by row.

    Each item becomes {names[0]: a_i, names[1]: b_i}; DataFrame rows become dicts. The
    result is a Series on the first pandas input's index, or a list. Use it for matching
    records, checking answers against references, or scoring a query against candidates:

        hunch.score(hunch.pairs(crm, vendors), ["different", "related", "same company"],
                    instructions="Are a and b the same company?")
    """
    left, right = box(a), box(b)
    if len(left.items) != len(right.items):
        raise HunchError(f"pairs() got {len(left.items)} and {len(right.items)} items; they must line up.")
    rows = [{names[0]: x, names[1]: y} for x, y in zip(left.items, right.items)]
    index = left.index if left.pandas else (right.index if right.pandas else None)
    if index is None:
        return rows
    import pandas as pd

    return pd.Series(rows, index=index, dtype=object)


# ----------------------------------------------------------------------------- ask


class _Keep:
    def __repr__(self) -> str:
        return "KEEP"


KEEP: Any = _Keep()
"""Default for split= / unsure=: keep the first answer."""


@dataclass(frozen=True)
class Classify:
    """One Choice question for ask(): same arguments as classify().

    context= is extra state for this question only. Questions with different context
    can't share a request, so ask() groups them.
    """

    labels: Any
    instructions: str | None = None
    split: Any = KEEP
    unsure: Any = KEEP
    context: Any = None


@dataclass(frozen=True)
class Rate:
    """One Score question for ask(): same arguments as score()."""

    levels: Sequence[str]
    instructions: str | None = None
    context: Any = None


@dataclass(frozen=True)
class Check:
    """One Noul question for ask(): same arguments as check()."""

    statement: str
    criteria: Mapping[str, str | None] | None = None
    threshold: float = 0.5
    context: Any = None
    uncertain: tuple[float, float] | None = None


Spec = Classify | Rate | Check


@overload
def ask(data: pd.DataFrame | pd.Series, questions: Mapping[str, Spec], *, columns: Sequence[str] | None = ..., context: Any = ..., detail: bool = ..., client: Client | None = ...) -> pd.DataFrame: ...
@overload
def ask(data: list[Any] | tuple[Any, ...], questions: Mapping[str, Spec], *, columns: Sequence[str] | None = ..., context: Any = ..., detail: bool = ..., client: Client | None = ...) -> list[dict[str, Any]]: ...
@overload
def ask(data: Any, questions: Mapping[str, Spec], *, columns: Sequence[str] | None = ..., context: Any = ..., detail: bool = ..., client: Client | None = ...) -> dict[str, Any]: ...
def ask(
    data: Any,
    questions: Mapping[str, Spec],
    *,
    columns: Sequence[str] | None = None,
    context: Any = None,
    detail: bool = False,
    client: Client | None = None,
) -> Any:
    """Ask several questions about the same data, one Jev request per item.

    questions maps a name to Classify(...), Rate(...), or Check(...). One item returns a
    dict of name -> answer; a list returns a list of dicts; a Series or DataFrame returns
    a DataFrame with one column per question on the same index (detail=True spreads each
    answer into several columns). A question's own context= is merged over context= here;
    questions whose merged context differs go in separate requests.
    """
    if not questions:
        raise HunchError("ask() needs at least one question.")
    jev = resolve(client)
    data_box = box(data, columns)
    built: dict[str, Any] = {}
    readers: dict[str, Callable[[Any], Any]] = {}
    contexts: dict[str, Any] = {}
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
            readers[name] = _skip(lambda raw, back=back: _answer(jev, raw, back))
        elif isinstance(spec, Rate):
            built[name] = Score(
                instructions=spec.instructions or "Where on this scale does the input fall?",
                criteria=_levels(spec.levels),
            )
            readers[name] = _skip(lambda raw: _rating(jev, raw))
        elif isinstance(spec, Check):
            crit = None
            if spec.criteria:
                crit = {"true": spec.criteria.get("true"), "false": spec.criteria.get("false")}
            built[name] = Noul(instructions=spec.statement, criteria=crit)
            low, high = _band(spec.uncertain, spec.threshold)
            readers[name] = _skip(lambda raw, t=high, lo=low: Feeling(float(raw["noul"]), t, lo))
        else:
            raise HunchError(f"ask() question {name!r} must be Classify, Rate, or Check.")
        contexts[name] = merge_context(context, spec.context)

    groups: dict[str, list[str]] = {}
    for name in built:
        groups.setdefault(json.dumps(engine.jsonable(contexts[name]), sort_keys=True, default=str), []).append(name)
    rows: list[dict[str, Any]] = [{} for _ in data_box.items]
    for names in groups.values():
        ctx = contexts[names[0]]
        states = [engine.build_state(item, ctx) for item in data_box.items]
        raws = engine.run(jev, states, {n: built[n] for n in names}, label="ask")
        for row, raw in zip(rows, raws):
            for n in names:
                row[n] = readers[n](raw[n])
    rows = [{n: r[n] for n in built} for r in rows]  # keep the caller's question order

    for name, spec in questions.items():
        if isinstance(spec, Classify) and (spec.split is not KEEP or spec.unsure is not KEEP):
            keys, describe, back = _labels(spec.labels)
            answers = _resolve(jev, data_box.items, [r[str(name)] for r in rows], keys, describe, back,
                               spec.instructions, contexts[str(name)], spec.split, spec.unsure, detail)
            for r, a in zip(rows, answers):
                r[str(name)] = a
    return data_box.out(rows, detail=detail, squeeze=False)


# ----------------------------------------------------------------------------- classify


@overload
def classify(data: pd.DataFrame | pd.Series, labels: Any, *, columns: Sequence[str] | None = ..., multi_label: bool = ..., instructions: str | None = ..., context: Any = ..., threshold: float = ..., split: Any = ..., unsure: Any = ..., detail: bool = ..., client: Client | None = ...) -> pd.Series | pd.DataFrame: ...
@overload
def classify(data: list[Any] | tuple[Any, ...], labels: Any, *, columns: Sequence[str] | None = ..., multi_label: bool = ..., instructions: str | None = ..., context: Any = ..., threshold: float = ..., split: Any = ..., unsure: Any = ..., detail: bool = ..., client: Client | None = ...) -> list[Any]: ...
@overload
def classify(data: Any, labels: Any, *, columns: Sequence[str] | None = ..., multi_label: bool = ..., instructions: str | None = ..., context: Any = ..., threshold: float = ..., split: Any = ..., unsure: Any = ..., detail: bool = ..., client: Client | None = ...) -> Any: ...
def classify(
    data: Any,
    labels: Any,
    *,
    columns: Sequence[str] | None = None,
    multi_label: bool = False,
    instructions: str | None = None,
    context: Any = None,
    threshold: float = 0.5,
    split: Any = KEEP,
    unsure: Any = KEEP,
    backoff: Mapping[str, str] | None = None,
    beam: int = 3,
    detail: bool = False,
    client: Client | None = None,
) -> Any:
    """Assign data to one of the labels (or several with multi_label=True).

    labels may be a hunch.Tree({"Parent": {"Child": description or None}}). Jev walks it
    level by level with beam search, keeping the best `beam` paths, and returns the path
    as "Parent > Child > Leaf".

    backoff={child: parent} answers with the parent when Jev can't decide between
    children but is sure of the family ("Laptops" vs "Tablets" -> "Computers").

    labels: a sequence of strings, an Enum class, or a mapping label -> description.
    Returns the label (an Enum member when labels is an Enum). detail=True returns
    Answer / MultiAnswer with the full distribution. columns= narrows a DataFrame.

    split= and unsure= are policies for shaky answers. split="rematch" re-asks between the
    top two options for rows where two labels were close. An LLM (hunch.anthropic(), any
    adapter, or a function (system, user) -> str) escalates those rows: it picks from the
    same labels, and Answer.by says "llm". Any other value is returned as the label for
    those rows. unsure= does the same for flat distributions. Both default to keeping the
    first answer.
    """
    jev = resolve(client)
    data_box = box(data, columns)
    if isinstance(labels, Tree):
        if multi_label:
            raise HunchError("multi_label=True doesn't work with a label tree.")
        answers = _classify_tree(jev, data_box.items, labels, beam, instructions, context)
        return data_box.out([{"label": a} for a in answers], detail=detail, squeeze=True)
    keys, describe, back = _labels(labels)
    if not keys:
        raise HunchError("classify() needs at least one label.")
    states = [engine.build_state(item, context) for item in data_box.items]

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

        def multi(raw: dict[str, Any]) -> MultiAnswer | None:
            if any(raw[f"l{i}"] is None for i in range(len(keys))):
                return None
            probs = {key: float(raw[f"l{i}"]["noul"]) for i, key in enumerate(keys)}
            chosen = [back(k) for k, p in sorted(probs.items(), key=lambda kv: -kv[1]) if p >= threshold]
            return MultiAnswer(chosen, probs, threshold)

        return data_box.out([{"labels": multi(raw)} for raw in raws], detail=detail, squeeze=True)

    if len(keys) > MAX_CHOICE_OPTIONS:
        raise HunchError(f"classify() takes at most {MAX_CHOICE_OPTIONS} labels.")
    question = Choice(
        instructions=instructions or "Which label best describes the input?",
        criteria={key: describe.get(key) for key in keys},
    )
    raws = engine.run(jev, states, {"q": question}, label="classify")
    read = _skip(lambda raw: _answer(jev, raw, back))
    answers = [read(raw["q"]) for raw in raws]
    if backoff:
        answers = [_back_off(jev, a, backoff) for a in answers]
    answers = _resolve(jev, data_box.items, answers, keys, describe, back, instructions, context, split, unsure, detail)
    return data_box.out([{"label": a} for a in answers], detail=detail, squeeze=True)


def _back_off(jev: Client, answer: Any, parents: Mapping[str, str]) -> Any:
    """For a shaky answer, sum probability by parent; answer with the parent if it's sure.

    Every answer gets a `by`, so the label_by column exists whether or not anything backed off."""
    from dataclasses import replace

    if not isinstance(answer, Answer):
        return answer
    if answer.by is None:
        answer = replace(answer, by="jev")
    if answer.shape == "sure":
        return answer
    mass: dict[str, float] = {}
    for label, p in answer.probabilities.items():
        parent = parents.get(label, label)
        mass[parent] = mass.get(parent, 0.0) + p
    parent, p = max(mass.items(), key=lambda kv: kv[1])
    if parent == answer.top or p < jev.policy.sure_peak:
        return answer
    return Answer(parent, mass, answer.confidence, jev.policy.classify(mass), by="parent")


def _classify_tree(jev: Client, items: list[Any], tree: Mapping[str, Any], beam: int,
                   instructions: str | None, context: Any) -> list[Any]:
    """Beam search down a label tree. One Choice per (level, node) batch; paths scored by
    the geometric mean of their step probabilities."""
    import math

    if beam < 1:
        raise HunchError("beam= must be at least 1.")

    def node(path: tuple[str, ...]) -> Any:
        here: Any = tree
        for step in path:
            here = here[step]
        return here

    def describe(child: Any) -> Any:
        if isinstance(child, Mapping):  # inside a Tree, a nested dict is a sub-tree
            return {"includes": [str(k) for k in child]}
        return child

    # per item: list of (path, log-probability sum)
    beams: list[list[tuple[tuple[str, ...], float]]] = [[((), 0.0)] for _ in items]
    alive = [True] * len(items)
    while True:
        todo: dict[tuple[str, ...], list[int]] = {}
        for i, paths in enumerate(beams):
            for path, _ in paths:
                if alive[i] and isinstance(node(path), Mapping):
                    todo.setdefault(path, []).append(i)
        if not todo:
            break
        answers: dict[tuple[tuple[str, ...], int], dict[str, float] | None] = {}
        for path, idxs in todo.items():
            children = node(path)
            question = Choice(
                instructions=_object(question=instructions or "Which category best describes the input?",
                                     within=" > ".join(path) or None),
                criteria={str(k): describe(v) for k, v in children.items()},
            )
            idxs = list(dict.fromkeys(idxs))
            raws = engine.run(jev, [engine.build_state(items[i], context) for i in idxs], {"q": question},
                              label="classify " + (path[-1] if path else "tree"))
            for i, raw in zip(idxs, raws):
                q = raw["q"]
                answers[(path, i)] = None if q is None else {str(k): float(v) for k, v in q["probabilities"].items()}
        for i, paths in enumerate(beams):
            grown: list[tuple[tuple[str, ...], float]] = []
            for path, logp in paths:
                if not isinstance(node(path), Mapping):
                    grown.append((path, logp))
                    continue
                probs = answers.get((path, i))
                if probs is None:
                    continue
                grown += [(path + (child,), logp + math.log(max(p, 1e-12))) for child, p in probs.items()]
            if not grown:
                alive[i] = False
                beams[i] = []
                continue
            grown.sort(key=lambda t: t[1] / len(t[0]), reverse=True)
            beams[i] = grown[:beam]
    out: list[Any] = []
    for paths in beams:
        if not paths:
            out.append(None)
            continue
        scores = {" > ".join(p): math.exp(logp / len(p)) for p, logp in paths}
        total = sum(scores.values()) or 1.0
        probs = {k: v / total for k, v in scores.items()}
        best = max(scores, key=scores.get)
        out.append(Answer(best, probs, scores[best], jev.policy.classify(probs)))
    return out


def _resolve(
    jev: Client,
    items: list[Any],
    answers: list[Any],
    keys: list[str],
    describe: dict[str, Any],
    back: Callable[[str], Any],
    instructions: str | None,
    context: Any,
    split: Any,
    unsure: Any,
    detail: bool,
) -> list[Any]:
    """Apply split= / unsure= policies. Rematches are batched by top-two pair."""
    if split is KEEP and unsure is KEEP:
        return answers
    literal_split = split not in (KEEP, "rematch") and not _is_llm(split)
    literal_unsure = unsure is not KEEP and not _is_llm(unsure)
    if detail and (literal_split or literal_unsure):
        raise HunchError("Literal split= / unsure= values need detail=False; use .on() for custom detail logic.")
    out = list(answers)
    for policy, shape in ((split, "split"), (unsure, "unsure")):
        if _is_llm(policy):
            from hunch.combine import escalate

            idxs = [i for i, a in enumerate(out) if isinstance(a, Answer) and a.shape == shape]
            out = escalate(jev, policy, items, idxs, out, keys, describe, back, instructions, context)
    split = KEEP if _is_llm(split) else split
    unsure = KEEP if _is_llm(unsure) else unsure
    if split == "rematch":
        groups: dict[tuple[str, ...], list[int]] = {}
        for i, a in enumerate(answers):
            if isinstance(a, Answer) and a.shape == "split":
                groups.setdefault(tuple(a.top2), []).append(i)
        for pair, idxs in groups.items():
            question = Choice(
                instructions=_object(question=instructions or "Which of these two labels fits the input better?", note="Only these two options apply."),
                criteria={key: describe.get(key) for key in pair},
            )
            states = [engine.build_state(items[i], context) for i in idxs]
            raws = engine.run(jev, states, {"q": question}, label="rematch")
            read = _skip(lambda raw: _answer(jev, raw, back))
            for i, raw in zip(idxs, raws):
                out[i] = read(raw["q"]) or out[i]  # a skipped rematch keeps the first answer
    elif split is not KEEP:
        out = [split if isinstance(a, Answer) and a.shape == "split" else a for a in out]
    if unsure is not KEEP:
        out = [unsure if isinstance(a, Answer) and a.shape == "unsure" else a for a in out]
    return out


# ----------------------------------------------------------------------------- score


@overload
def score(data: pd.DataFrame | pd.Series, levels: Sequence[str], *, columns: Sequence[str] | None = ..., instructions: str | Mapping[str, str] | None = ..., context: Any = ..., detail: bool = ..., client: Client | None = ...) -> pd.Series | pd.DataFrame: ...
@overload
def score(data: list[Any] | tuple[Any, ...], levels: Sequence[str], *, columns: Sequence[str] | None = ..., instructions: str | Mapping[str, str] | None = ..., context: Any = ..., detail: bool = ..., client: Client | None = ...) -> list[Any]: ...
@overload
def score(data: Any, levels: Sequence[str], *, columns: Sequence[str] | None = ..., instructions: str | Mapping[str, str] | None = ..., context: Any = ..., detail: bool = ..., client: Client | None = ...) -> Any: ...
def score(
    data: Any,
    levels: Sequence[str],
    *,
    columns: Sequence[str] | None = None,
    instructions: str | Mapping[str, str] | None = None,
    context: Any = None,
    detail: bool = False,
    client: Client | None = None,
) -> Any:
    """Place data on an ordered scale. Returns the position 0 .. len(levels)-1.

    levels: 2–10 concrete situations, worst first. instructions may be a mapping of
    dimension name -> question to score several dimensions in one request; the result
    is then a dict per item. detail=True returns Rating(s). columns= narrows a DataFrame.
    """
    jev = resolve(client)
    data_box = box(data, columns)
    levels = _levels(levels)
    named = isinstance(instructions, Mapping)
    dims = _dims(instructions, default="Where on this scale does the input fall?", base="score")
    states = [engine.build_state(item, context) for item in data_box.items]
    questions = {name: Score(instructions=text, criteria=list(levels)) for name, text in dims.items()}
    raws = engine.run(jev, states, questions, label="score")
    read = _skip(lambda raw: _rating(jev, raw))
    rows = [{name: read(raw[name]) for name in dims} for raw in raws]
    return data_box.out(rows, detail=detail, squeeze=not named)


# ----------------------------------------------------------------------------- check


@overload
def check(data: pd.DataFrame | pd.Series, statement: str | Mapping[str, str], *, columns: Sequence[str] | None = ..., criteria: Mapping[str, str | None] | None = ..., context: Any = ..., threshold: float = ..., uncertain: tuple[float, float] | None = ..., detail: bool = ..., client: Client | None = ...) -> pd.Series | pd.DataFrame: ...
@overload
def check(data: list[Any] | tuple[Any, ...], statement: str | Mapping[str, str], *, columns: Sequence[str] | None = ..., criteria: Mapping[str, str | None] | None = ..., context: Any = ..., threshold: float = ..., uncertain: tuple[float, float] | None = ..., detail: bool = ..., client: Client | None = ...) -> list[Any]: ...
@overload
def check(data: Any, statement: str | Mapping[str, str], *, columns: Sequence[str] | None = ..., criteria: Mapping[str, str | None] | None = ..., context: Any = ..., threshold: float = ..., uncertain: tuple[float, float] | None = ..., detail: bool = ..., client: Client | None = ...) -> Any: ...
def check(
    data: Any,
    statement: str | Mapping[str, str],
    *,
    columns: Sequence[str] | None = None,
    criteria: Mapping[str, str | None] | None = None,
    context: Any = None,
    threshold: float = 0.5,
    uncertain: tuple[float, float] | None = None,
    detail: bool = False,
    client: Client | None = None,
) -> Any:
    """Does the statement hold for the data? Returns bool (P(yes) >= threshold).

    uncertain=(0.3, 0.7) adds a "maybe": P(yes) at or above 0.7 is True, at or below 0.3
    is False, and in between is None, so borderline rows can go to review instead of
    being forced to a side. It replaces threshold.

    statement may be a mapping name -> statement to check several in one request.
    criteria={"true": ..., "false": ...} sharpens the boundary. detail=True returns Feeling(s).
    columns= narrows a DataFrame.
    """
    low, high = _band(uncertain, threshold)
    jev = resolve(client)
    data_box = box(data, columns)
    named = isinstance(statement, Mapping)
    dims = _dims(statement, default=None, base="check")
    crit = None
    if criteria:
        crit = {"true": criteria.get("true"), "false": criteria.get("false")}
    states = [engine.build_state(item, context) for item in data_box.items]
    questions = {name: Noul(instructions=text, criteria=crit) for name, text in dims.items()}
    raws = engine.run(jev, states, questions, label="check")
    read = _skip(lambda raw: Feeling(float(raw["noul"]), high, low))
    rows = [{name: read(raw[name]) for name in dims} for raw in raws]
    return data_box.out(rows, detail=detail, squeeze=not named)


# ----------------------------------------------------------------------------- where


@overload
def where(data: pd.DataFrame, statement: str, *, columns: Sequence[str] | None = ..., criteria: Mapping[str, str | None] | None = ..., context: Any = ..., threshold: float = ..., detail: bool = ..., client: Client | None = ...) -> pd.DataFrame: ...
@overload
def where(data: pd.Series, statement: str, *, columns: Sequence[str] | None = ..., criteria: Mapping[str, str | None] | None = ..., context: Any = ..., threshold: float = ..., detail: bool = ..., client: Client | None = ...) -> pd.Series | pd.DataFrame: ...
@overload
def where(data: list[T] | tuple[T, ...], statement: str, *, columns: Sequence[str] | None = ..., criteria: Mapping[str, str | None] | None = ..., context: Any = ..., threshold: float = ..., detail: bool = ..., client: Client | None = ...) -> list[T]: ...
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
    row with `match` and `match_p` columns instead of filtering. Skipped requests don't match.
    """
    data_box = box(data)
    if data_box.kind == "single":
        raise HunchError("where() needs a list, Series, or DataFrame.")
    if columns is not None and data_box.kind != "frame":
        raise HunchError("columns= only applies to a DataFrame.")
    feelings = check(data, statement, columns=columns, criteria=criteria, context=context,
                     threshold=threshold, detail=True, client=client)
    if data_box.pandas:
        match = feelings["check"].fillna(False).astype(bool)
        p = feelings["check_p"] if "check_p" in feelings else match.astype(float) * float("nan")
        if detail:
            frame = data if data_box.kind == "frame" else data.to_frame(name=data.name if data.name is not None else "value")
            return frame.assign(match=match, match_p=p)
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
    none: bool = False,
    none_threshold: float = 0.5,
    columns: Sequence[str] | None = None,
    context: Any = None,
    detail: bool = False,
    client: Client | None = None,
) -> Any:
    """Choose the single best candidate. Jev compares them head to head in one Choice.

    A list returns the winning item. A Series or DataFrame returns the winner's index
    label, so df.loc[winner] is the row; columns= limits what Jev reads. More than 255
    candidates run as a tournament. detail=True returns Pick with every candidate's probability.

    A Choice always crowns someone. none=True also asks, in the same request, whether any
    candidate actually satisfies the task, and returns None when P(fits) is below
    none_threshold.
    """
    jev = resolve(client)
    data_box = box(candidates, columns)
    if data_box.kind == "single":
        raise HunchError("pick() needs a list, Series, or DataFrame of candidates.")
    field = data_box.items
    if not field:
        raise HunchError("pick() needs at least one candidate.")
    if not instructions or not instructions.strip():
        raise HunchError("pick() needs instructions saying what 'best' means.")
    keys: list[Hashable] = list(data_box.index) if data_box.pandas else field

    def heat(group: list[int], final: bool) -> Pick:
        ask_fits = none and final
        if len(group) == 1 and not ask_fits:
            return Pick(group[0], [(group[0], 1.0)], 1.0, "sure")
        criteria = {f"c{i}": engine.jsonable(field[i]) for i in group}
        questions: dict[str, Any] = {}
        if len(group) > 1:
            questions["q"] = Choice(
                instructions={"task": instructions, "note": "Each option is one candidate."},
                criteria=criteria,
            )
        task: dict[str, Any] = {"task": instructions}
        if ask_fits:
            task["candidates"] = criteria
            questions["fits"] = Noul(
                instructions="Does at least one of the candidates actually satisfy the task well?",
                criteria={"true": "at least one candidate is a genuinely good fit for the task",
                          "false": "none of the candidates is a good fit; picking any would be settling"},
            )
        raw = engine.run(jev, [engine.build_state(task, context)], questions, label="pick")[0]
        if any(v is None for v in raw.values()):
            raise HunchError("pick() request failed; see the warning above.")
        fits = float(raw["fits"]["noul"]) if ask_fits else None
        if "q" not in raw:
            return Pick(group[0], [(group[0], 1.0)], 1.0, "sure", fits)
        probs = {cid: float(p) for cid, p in raw["q"]["probabilities"].items()}
        ranked = sorted(((int(cid[1:]), p) for cid, p in probs.items()), key=lambda t: t[1], reverse=True)
        return Pick(int(raw["q"]["choice"][1:]), ranked, float(raw["q"]["confidence"]), jev.policy.classify(probs), fits)

    result = _tournament(list(range(len(field))), heat)
    winner = keys[result.winner]
    if none and result.fits is not None and result.fits < none_threshold:
        winner = None
    result = Pick(winner, [(keys[i], p) for i, p in result.ranked], result.confidence, result.shape, result.fits)
    return result if detail else result.winner


def _tournament(field: list[int], heat: Callable[[list[int], bool], Pick]) -> Pick:
    # ponytail: heats of 255 then a final; a bracket with seeding if fields get huge
    while len(field) > MAX_CHOICE_OPTIONS:
        field = [
            heat(field[i : i + MAX_CHOICE_OPTIONS], False).winner
            for i in range(0, len(field), MAX_CHOICE_OPTIONS)
        ]
    return heat(field, True)


# ----------------------------------------------------------------------------- rank


@overload
def rank(candidates: pd.DataFrame | pd.Series, dimensions: str | Mapping[str, str], levels: Sequence[str], *, columns: Sequence[str] | None = ..., weights: Mapping[str, float] | None = ..., query: Any = ..., context: Any = ..., client: Client | None = ...) -> pd.DataFrame: ...
@overload
def rank(candidates: Sequence[Any], dimensions: str | Mapping[str, str], levels: Sequence[str], *, columns: Sequence[str] | None = ..., weights: Mapping[str, float] | None = ..., query: Any = ..., context: Any = ..., client: Client | None = ...) -> list[Ranked]: ...
def rank(
    candidates: Any,
    dimensions: str | Mapping[str, str],
    levels: Sequence[str],
    *,
    columns: Sequence[str] | None = None,
    weights: Mapping[str, float] | None = None,
    query: Any = None,
    context: Any = None,
    client: Client | None = None,
) -> Any:
    """Score every candidate on each dimension, weight, and sort best first.

    query= is what the candidates are being ranked for (a search, a job description, a
    buyer profile); every candidate is scored against it, which makes rank a reranker.

    Composite = weighted mean of normalized (0–1) dimension scores. Weights default to 1.
    A list returns Ranked rows. A Series or DataFrame returns a DataFrame on the same
    index with `composite` and one column per dimension, sorted best first; columns=
    limits what Jev reads. Skipped requests get a NaN composite and sort last.
    """
    data_box = box(candidates, columns)
    if data_box.kind == "single":
        raise HunchError("rank() needs a list, Series, or DataFrame of candidates.")
    dims = dimensions if isinstance(dimensions, Mapping) else {"score": dimensions}
    names = list(dims)
    w = {name: float((weights or {}).get(name, 1.0)) for name in names}
    total = sum(w.values())
    if total <= 0:
        raise HunchError("rank() weights must sum to more than zero.")
    if query is not None:
        context = merge_context(context, {"query": query})
    ratings = score(data_box.items, levels, instructions=dims, context=context, detail=True, client=client)
    nan = float("nan")
    composites = [
        nan if any(r[n] is None for n in names) else sum(w[n] * r[n].normalized for n in names) / total
        for r in ratings
    ]
    if data_box.pandas:
        import pandas as pd

        frame = pd.DataFrame(
            [{"composite": c, **{n: (r[n].score if r[n] is not None else nan) for n in names}} for c, r in zip(composites, ratings)],
            index=data_box.index,
        )
        return frame.sort_values("composite", ascending=False, na_position="last")
    items = list(candidates)
    rows = [Ranked(item, c, r) for item, c, r in zip(items, composites, ratings)]
    return sorted(rows, key=lambda row: -row.composite if row.composite == row.composite else float("inf"))


# ----------------------------------------------------------------------------- async twins


def _async(fn: Callable[..., T]) -> Callable[..., Any]:
    """The verb runs in a worker thread; its Jev requests run as coroutines on the caller's
    event loop through the async SDK client, max_concurrency at a time."""

    async def run(*args: Any, **kwargs: Any) -> T:
        token = engine.LOOP.set(asyncio.get_running_loop())
        try:
            return await asyncio.to_thread(fn, *args, **kwargs)
        finally:
            engine.LOOP.reset(token)

    run.__name__ = f"{fn.__name__}_async"
    run.__doc__ = f"Async twin of {fn.__name__}(). Same arguments.\n\n{fn.__doc__}"
    return run


ask_async = _async(ask)
classify_async = _async(classify)
score_async = _async(score)
check_async = _async(check)
where_async = _async(where)


def verify(*args: Any, **kwargs: Any) -> Any:  # re-exported for the accessor; defined in combine
    from hunch.combine import verify as _verify

    return _verify(*args, **kwargs)


verify_async = _async(verify)
pick_async = _async(pick)
rank_async = _async(rank)


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


def _band(uncertain: tuple[float, float] | None, threshold: float) -> tuple[float | None, float]:
    if uncertain is None:
        return None, threshold
    low, high = uncertain
    if not 0 <= low < high <= 1:
        raise HunchError("uncertain= needs (low, high) with 0 <= low < high <= 1.")
    return low, high


def _is_llm(value: Any) -> bool:
    """An LLM policy: a LanguageModel, or a function (system, user) -> str."""
    return value is not KEEP and not isinstance(value, (str, int, float, bool)) and (
        hasattr(value, "complete") or callable(value)
    )


def _object(**fields: Any) -> dict[str, Any]:
    return {k: v for k, v in fields.items() if v is not None}


def _answer(jev: Client, raw: dict[str, Any], back: Callable[[str], Any]) -> Answer:
    probs = {str(k): float(v) for k, v in raw["probabilities"].items()}
    return Answer(
        label=back(str(raw["choice"])),
        probabilities=probs,
        confidence=float(raw["confidence"]),
        shape=jev.policy.classify(probs),
    )


def _rating(jev: Client, raw: dict[str, Any]) -> Rating:
    legend = {int(k): str(v) for k, v in raw["legend"].items()}
    probs = {int(k): float(v) for k, v in raw["probabilities"].items()}
    shape = jev.policy.classify({legend[k]: p for k, p in probs.items()})
    return Rating(float(raw["score"]), float(raw["confidence"]), legend, probs, shape)
