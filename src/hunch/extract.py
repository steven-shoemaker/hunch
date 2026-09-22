"""extract(): pull values out of text. Code finds the candidates; Jev picks which one is the answer.

The value returned is always copied from the text, never written by a model.
"""

from __future__ import annotations

import asyncio
import re
from collections.abc import Callable, Mapping, Sequence
from concurrent.futures import ThreadPoolExecutor
from typing import Any

from typesafe_sdk import Choice

from hunch import engine
from hunch.answer import Answer
from hunch.client import Client, resolve
from hunch.exceptions import HunchError

_MONTH = r"(?:Jan|Feb|Mar|Apr|May|Jun|Jul|Aug|Sep|Sept|Oct|Nov|Dec)[a-z]*\.?"
PATTERNS: dict[str, str] = {
    "email": r"[\w.+-]+@[\w-]+(?:\.[\w-]+)+",
    "url": r"https?://[^\s<>\"')]+",
    "money": r"(?:[$€£¥]\s?\d[\d,]*(?:\.\d+)?(?:\s?[kKmM]\b)?|\d[\d,]*(?:\.\d+)?\s?(?:USD|EUR|GBP|dollars|euros)\b)",
    "number": r"-?\d[\d,]*(?:\.\d+)?%?",
    "percent": r"-?\d+(?:\.\d+)?\s?%",
    "phone": r"\+?\d[\d\s().-]{7,}\d",
    "date": (
        r"\b(?:\d{4}-\d{2}-\d{2}|\d{1,2}[/.-]\d{1,2}[/.-]\d{2,4}"
        rf"|{_MONTH}\s\d{{1,2}}(?:st|nd|rd|th)?,?\s\d{{4}}|\d{{1,2}}(?:st|nd|rd|th)?\s{_MONTH},?\s\d{{4}})\b"
    ),
}
"""Built-in candidate finders. Use a name ("email") or your own regex or function."""

MAX_CANDIDATES = 254  # a Choice takes 255 options; one is "none"
NONE = "none"


def extract(
    data: Any,
    fields: Mapping[str, Any],
    *,
    columns: Sequence[str] | None = None,
    context: Any = None,
    detail: bool = False,
    client: Client | None = None,
) -> Any:
    """Pull named values out of text. Returns None for a field the text doesn't state.

    fields maps a name to how to find its candidates: a built-in name ("email", "url",
    "money", "number", "percent", "phone", "date"), a regex, a compiled pattern, a function
    text -> list of strings, or a (finder, description) tuple when the name needs explaining:

        hunch.extract(invoices["body"], {
            "total": ("money", "the amount due, not a subtotal or tax line"),
            "due_date": "date",
            "billing_email": "email",
        })

    Every candidate goes to Jev with the words around it, plus a "none" option, and all
    fields for a row go in one request. A row with no candidates for a field costs nothing
    for that field. One value returns a dict, a list returns a list of dicts, and a Series
    or DataFrame returns a DataFrame with one column per field (detail=True adds _p and
    _shape columns).
    """
    from hunch.verbs import box

    if not fields:
        raise HunchError("extract() needs at least one field.")
    jev = resolve(client)
    specs = {str(name): _spec(name, spec) for name, spec in fields.items()}
    data_box = box(data, columns)
    texts = [_text(item) for item in data_box.items]

    def one(text: str) -> dict[str, Any]:
        questions: dict[str, Any] = {}
        options: dict[str, dict[str, str]] = {}
        for name, (finder, description) in specs.items():
            found = _candidates(text, finder)
            if not found:
                continue
            label = name.replace("_", " ")
            ids = {f"c{i}": value for i, (value, _) in enumerate(found)}
            options[name] = ids
            questions[name] = Choice(
                instructions={
                    "question": f"Which candidate is the {label} the text states?",
                    "field": description,
                    "rules": f"Pick the candidate the text gives as the {label}. Pick none if the text "
                             f"doesn't state a {label}, or if no candidate is it.",
                },
                criteria={
                    **{f"c{i}": {"value": value, "in_context": snippet} for i, (value, snippet) in enumerate(found)},
                    NONE: f"the text doesn't state a {label}, or none of the candidates is it",
                },
            )
        row: dict[str, Any] = {name: None for name in specs}
        if not questions:
            return row
        raw = engine.run(jev, [engine.build_state(text, context)], questions, label="extract")[0]
        for name in questions:
            answer = raw[name]
            if answer is None:
                continue
            ids = options[name]
            probs: dict[str, float] = {}
            for cid, p in answer["probabilities"].items():
                key = "None" if cid == NONE else ids[cid]
                probs[key] = probs.get(key, 0.0) + float(p)
            value = None if answer["choice"] == NONE else ids[answer["choice"]]
            row[name] = Answer(value, probs, float(answer["confidence"]), jev.policy.classify(probs))
        return row

    distinct = list(dict.fromkeys(texts))
    bar = engine.progress(jev, len(distinct), "extract")
    if len(distinct) <= 1 or jev.max_workers <= 1:
        done = list(bar(map(one, distinct)))
    else:
        with ThreadPoolExecutor(max_workers=jev.max_workers) as pool:
            done = list(bar(pool.map(one, distinct)))
    by_text = dict(zip(distinct, done))
    rows = [by_text[t] for t in texts]
    return data_box.out(rows, detail=detail, squeeze=False)


async def extract_async(*args: Any, **kwargs: Any) -> Any:
    token = engine.LOOP.set(asyncio.get_running_loop())
    try:
        return await asyncio.to_thread(extract, *args, **kwargs)
    finally:
        engine.LOOP.reset(token)


def _spec(name: Any, spec: Any) -> tuple[Callable[[str], list[str]], str | None]:
    description = None
    if isinstance(spec, tuple):
        if len(spec) != 2:
            raise HunchError(f"extract() field {name!r}: use (finder, description).")
        spec, description = spec
    if isinstance(spec, str):
        pattern = PATTERNS.get(spec, spec)
        try:
            spec = re.compile(pattern)
        except re.error as error:
            raise HunchError(f"extract() field {name!r}: {spec!r} is not a known finder or a valid regex ({error}).") from None
    if isinstance(spec, re.Pattern):
        compiled = spec
        return (lambda text: [m.group(0) for m in compiled.finditer(text)]), description
    if callable(spec):
        return (lambda text: [str(v) for v in spec(text) or []]), description
    raise HunchError(f"extract() field {name!r} must be a finder name, regex, pattern, or function.")


def _candidates(text: str, finder: Callable[[str], list[str]]) -> list[tuple[str, str]]:
    """Distinct values with the words around their first appearance."""
    out: dict[str, str] = {}
    for value in finder(text):
        value = value.strip()
        if not value or value in out:
            continue
        at = text.find(value)
        start, end = max(0, at - 60), at + len(value) + 60
        out[value] = ("…" if start else "") + text[start:end].replace("\n", " ") + ("…" if end < len(text) else "")
        if len(out) >= MAX_CANDIDATES:
            break
    return list(out.items())


def _text(item: Any) -> str:
    if item is None:
        return ""
    if isinstance(item, Mapping):
        return "\n".join(f"{k}: {v}" for k, v in item.items() if v is not None)
    return str(item)
