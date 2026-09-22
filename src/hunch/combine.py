"""An LLM and Jev together: escalate, discover, refine, verify. The LLM proposes; Jev decides or checks."""

from __future__ import annotations

import asyncio
import json
import warnings
from collections.abc import Callable, Mapping, Sequence
from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass, field
from typing import Any

from pydantic_core import to_jsonable_python

from hunch import engine
from hunch.answer import Answer, Feeling
from hunch.client import Client, resolve
from hunch.exceptions import HunchError
from hunch.llm import as_llm


# ----------------------------------------------------------------------------- escalate


def escalate(
    jev: Client,
    llm: Any,
    items: list[Any],
    idxs: list[int],
    answers: list[Any],
    keys: list[str],
    describe: Mapping[str, Any],
    back: Callable[[str], Any],
    instructions: str | None,
    context: Any,
) -> list[Any]:
    """Send the rows at idxs to an LLM, which must pick one of the same labels.

    Used by classify(split=llm / unsure=llm). Decisions are cached; a reply that isn't one
    of the labels keeps Jev's answer and is reported in a warning.
    """
    from dataclasses import replace

    model = as_llm(llm)
    # every row gets a `by`, so the label_by column exists whether or not anything escalated
    out = [replace(a, by="jev") if isinstance(a, Answer) and a.by is None else a for a in answers]
    if not idxs:
        return out
    system = (
        "You are the careful second opinion for a fast classifier that was unsure about this input. "
        "Choose exactly one label from `labels`. Reply with ONLY JSON: {\"label\": \"<one of the labels>\"}."
    )
    by_item: dict[str, list[int]] = {}
    for i in idxs:
        by_item.setdefault(engine._dump(engine.jsonable(items[i])), []).append(i)

    def decide(item_key: str) -> str | None:
        i = by_item[item_key][0]
        first: Answer = answers[i]
        payload = {
            "task": instructions or "Which label best describes the input?",
            "labels": {k: describe.get(k) for k in keys},
            "input": engine.jsonable(items[i]),
            "context": engine.jsonable(context),
            "first_pass_probabilities": dict(first.ranked[:3]),
        }
        cache_key = "escalate|" + json.dumps({"model": getattr(model, "name", None), **payload}, sort_keys=True, default=str)
        cached = jev.cache.get(cache_key)
        if cached is not None:
            return cached["label"]
        user = json.dumps({k: v for k, v in payload.items() if v is not None})
        for _ in range(2):
            label = _pick_label(model.complete(system=system, user=user), keys)
            if label is not None:
                jev.cache.set(cache_key, {"label": label})
                return label
            user += f"\n\nYour reply was not one of the labels. Choose exactly one of: {keys}"
        return None

    order = list(by_item)
    bar = engine.progress(jev, len(order), "escalate")
    with ThreadPoolExecutor(max_workers=jev.max_workers) as pool:
        decided = list(bar(pool.map(decide, order)))
    misses = 0
    for item_key, label in zip(order, decided):
        for i in by_item[item_key]:
            if label is None:
                misses += 1
                continue
            first = answers[i]
            out[i] = Answer(back(label), first.probabilities, first.confidence, first.shape, by="llm")
    if misses:
        warnings.warn(f"hunch escalate: the LLM gave no valid label for {misses} rows; kept Jev's answer.", stacklevel=4)
    return out


def _pick_label(text: str, keys: list[str]) -> str | None:
    raw = _strip_fence(text)
    try:
        data = json.loads(raw)
        candidate = str(data.get("label")) if isinstance(data, dict) else str(data)
    except json.JSONDecodeError:
        candidate = raw.strip().strip('"')
    if candidate in keys:
        return candidate
    lowered = {k.lower(): k for k in keys}
    return lowered.get(candidate.lower())


# ----------------------------------------------------------------------------- discover


@dataclass
class _Category:
    name: str
    description: str


def discover(
    data: Any,
    n: int = 8,
    *,
    instructions: str | None = None,
    columns: Sequence[str] | None = None,
    sample: int = 100,
    other: bool = True,
    llm: Any = None,
    fresh: bool = False,
    client: Client | None = None,
) -> dict[str, str]:
    """Let an LLM read a sample and propose n categories, as {name: description}.

    The result plugs straight into classify(data, labels=...), so Jev files every row into
    categories nobody had to write by hand. instructions= steers what to group by
    ("by the customer's complaint", "by buying intent"). other=True adds an "other"
    category so rows that fit nothing aren't forced into one. Cached like generate().
    """
    from hunch.generate import generate
    from hunch.verbs import box

    if n < 2:
        raise HunchError("discover() needs n >= 2.")
    items = box(data, columns).items
    distinct = list({engine._dump(engine.jsonable(x)): x for x in items if x is not None}.values())
    if not distinct:
        raise HunchError("discover() needs some data.")
    step = max(1, len(distinct) // sample)
    examples = [engine.jsonable(x) for x in distinct[::step][:sample]]
    proposed = generate(
        _Category,
        n=n,
        instructions=(
            f"Propose {n} categories that together cover these examples. Short names in plain words. "
            "Mutually exclusive. Each description says in one sentence what belongs and what does not. "
            "No catch-all category." + (f" Group them {instructions}." if instructions else "")
        ),
        context={"examples": examples},
        llm=llm,
        fresh=fresh,
        client=client,
    )
    categories: dict[str, str] = {}
    for cat in proposed if isinstance(proposed, list) else [proposed]:
        name = str(cat.name).strip()
        if name and name.lower() not in {k.lower() for k in categories} and name.lower() != "other":
            categories[name] = str(cat.description).strip()
    if other:
        categories["other"] = "fits none of the other categories"
    return categories


# ----------------------------------------------------------------------------- refine


@dataclass(frozen=True)
class Refined:
    """One refine() result."""

    text: str
    passed: bool | None
    """True when every check held. None when a check request was skipped (errors="skip")."""
    rounds: int
    """LLM rewrites it took. 0 means the input already passed."""
    failed: list[str] = field(default_factory=list)
    """Checks still failing at the end."""
    original: str = ""


def refine(
    data: Any,
    checks: Sequence[str] | Mapping[str, str],
    *,
    instructions: str | None = None,
    rounds: int = 3,
    threshold: float = 0.5,
    context: Any = None,
    llm: Any = None,
    detail: bool = False,
    client: Client | None = None,
) -> Any:
    """Rewrite text with an LLM until Jev agrees every check holds, or rounds run out.

    checks are statements a good version makes true ("has no hype words", "shows the
    install command"), as a list or {name: statement}. Each round, Jev checks every draft
    at once; only the failing drafts go back to the LLM, told which checks failed.
    Returns the final text in the caller's container. detail=True returns Refined objects,
    or for pandas a DataFrame with text / passed / rounds / failed.
    """
    from hunch.verbs import box, check

    jev = resolve(client)
    model = as_llm(llm) or jev.llm
    if model is None:
        raise HunchError("refine() needs a language model: pass llm= here or on configure().")
    named = dict(checks) if isinstance(checks, Mapping) else {f"c{i}": str(c) for i, c in enumerate(checks)}
    if not named:
        raise HunchError("refine() needs at least one check.")
    if rounds < 0:
        raise HunchError("refine() rounds must be >= 0.")
    data_box = box(data)
    originals = ["" if x is None else str(x) for x in data_box.items]
    drafts = list(originals)
    used = [0] * len(drafts)
    failing: list[list[str] | None] = [None] * len(drafts)
    pending = list(range(len(drafts)))

    system = (
        "Rewrite the text so every requirement holds. Keep everything that already works, "
        "including meaning and length, unless a requirement says otherwise. Reply with ONLY the rewritten text."
    )

    for round_no in range(rounds + 1):
        if not pending:
            break
        verdicts = check([drafts[i] for i in pending], named, context=context, threshold=threshold, detail=True, client=jev)
        still: list[int] = []
        for i, verdict in zip(pending, verdicts):
            if any(v is None for v in verdict.values()):
                failing[i] = None  # unknown: don't rewrite blind
                continue
            bad = [name for name, f in verdict.items() if not f]
            failing[i] = bad
            if bad:
                still.append(i)
        if round_no == rounds or not still:
            break

        def rewrite(i: int) -> str:
            payload = {
                "text": drafts[i],
                "failed_requirements": [named[n] for n in failing[i] or []],
                "all_requirements": list(named.values()),
                "instructions": instructions,
                "context": engine.jsonable(context),
            }
            return _strip_fence(model.complete(system=system, user=json.dumps({k: v for k, v in payload.items() if v is not None})))

        bar = engine.progress(jev, len(still), f"refine {round_no + 1}")
        with ThreadPoolExecutor(max_workers=jev.max_workers) as pool:
            for i, text in zip(still, bar(pool.map(rewrite, still))):
                drafts[i] = text
                used[i] += 1
        pending = still

    labels = {v: k for k, v in named.items()} if isinstance(checks, Mapping) else {}
    results = [
        Refined(
            text=drafts[i],
            passed=None if failing[i] is None else not failing[i],
            rounds=used[i],
            failed=[named[n] if not labels else n for n in (failing[i] or [])],
            original=originals[i],
        )
        for i in range(len(drafts))
    ]
    if data_box.pandas:
        import pandas as pd

        if detail:
            return pd.DataFrame([{"text": r.text, "passed": r.passed, "rounds": r.rounds, "failed": r.failed} for r in results], index=data_box.index)
        return pd.Series([r.text for r in results], index=data_box.index, name="text")
    out: list[Any] = results if detail else [r.text for r in results]
    return out[0] if data_box.kind == "single" else out


# ----------------------------------------------------------------------------- verify


def verify(
    claims: Any,
    source: Any,
    *,
    threshold: float = 0.5,
    context: Any = None,
    detail: bool = False,
    client: Client | None = None,
) -> Any:
    """Is each claim supported by its source? For checking what an LLM or agent produced.

    source is one document for all claims, or a sequence / Series aligned with claims
    (one source per claim). Returns bools in the caller's container (a Series named
    `supported` for pandas); detail=True gives probabilities too. Only what the source
    states counts: missing, contradicted, or partly supported claims are False.
    """
    from hunch.verbs import box, check, merge_context

    claim_box = box(claims)
    statement = {
        "question": "Is the claim fully supported by the source?",
        "rules": "Judge only what the source states. Claims the source contradicts, doesn't mention, or only partly supports are not supported.",
    }
    criteria = {
        "true": "the source states or directly implies every part of the claim",
        "false": "the source contradicts the claim, doesn't mention it, or supports only part of it",
    }
    aligned = not isinstance(source, (str, bytes, Mapping)) and hasattr(source, "__len__")
    if aligned:
        sources = list(source.tolist() if hasattr(source, "tolist") else source)
        if len(sources) != len(claim_box.items):
            raise HunchError(f"verify() got {len(claim_box.items)} claims and {len(sources)} sources.")
        pairs = [{"claim": c, "source": s} for c, s in zip(claim_box.items, sources)]
        if claim_box.pandas:
            import pandas as pd

            data: Any = pd.Series(pairs, index=claim_box.index)
        else:
            data = pairs if claim_box.kind != "single" else pairs[0]
        ctx = context
    else:
        data = claims
        ctx = merge_context(context, {"source": source})
    out = check(data, json.dumps(statement), criteria=criteria, context=ctx, threshold=threshold, detail=detail, client=client)
    if claim_box.pandas:
        return out.rename("supported") if not detail else out.rename(columns={"check": "supported", "check_p": "supported_p"})
    return out


# ----------------------------------------------------------------------------- async twins


def _to_thread(fn: Callable[..., Any]) -> Callable[..., Any]:
    async def run(*args: Any, **kwargs: Any) -> Any:
        return await asyncio.to_thread(fn, *args, **kwargs)

    run.__name__ = f"{fn.__name__}_async"
    return run


discover_async = _to_thread(discover)
refine_async = _to_thread(refine)


def _strip_fence(raw: str) -> str:
    text = raw.strip()
    if not text.startswith("```"):
        return text
    lines = text.splitlines()[1:]
    if lines and lines[-1].strip() == "```":
        lines = lines[:-1]
    return "\n".join(lines).strip()
