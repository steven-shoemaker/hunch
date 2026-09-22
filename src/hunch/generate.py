"""generate(): an LLM proposes typed data. It never decides anything; that is Jev's job."""

from __future__ import annotations

import asyncio
import json
from typing import Any

from pydantic import TypeAdapter, ValidationError
from pydantic_core import to_jsonable_python

from hunch import engine
from hunch.client import Client, resolve
from hunch.exceptions import HunchError
from hunch.llm import LanguageModel, as_llm


BATCH = 25
"""Items per LLM call. Large n is drawn in batches, deduped, and topped up."""


def generate(
    target: Any = str,
    n: int = 1,
    *,
    instructions: str | None = None,
    context: Any = None,
    llm: LanguageModel | None = None,
    fresh: bool = False,
    client: Client | None = None,
) -> Any:
    """Create n examples of target: str, int, a dataclass, a TypedDict, list[str], ...

    Returns one value when n == 1, else a list of exactly n distinct items. Uses llm= here,
    else the client's llm= from configure(). Output is validated against the target type.
    n above 25 is drawn in batches that avoid repeating earlier items. Results are cached
    on the client (and on disk with cache=), so re-running the same call returns the same
    items; fresh=True draws new ones.
    """
    if n < 1:
        raise HunchError("generate() needs n >= 1.")
    jev = resolve(client)
    model = as_llm(llm) or jev.llm
    if model is None:
        raise HunchError("generate() needs a language model: pass llm= here or on configure().")
    name = getattr(target, "__name__", str(target))
    one: TypeAdapter[Any] = TypeAdapter(target)
    key = "generate|" + json.dumps(
        {"model": getattr(model, "name", None), "schema": one.json_schema(), "instructions": instructions,
         "context": to_jsonable_python(context), "n": n},
        sort_keys=True, default=str,
    )
    if not fresh:
        cached = jev.cache.get(key)
        if cached is not None:
            jev.meter.hit()
            return one.validate_python(cached) if n == 1 else [one.validate_python(v) for v in cached]

    if n == 1:
        value = _draw(model, jev, target, 1, instructions, context, [], f"generate 1 {name}")
        jev.cache.set(key, to_jsonable_python(value))
        return value

    items: list[Any] = []
    seen: set[str] = set()
    rounds = 0
    while len(items) < n:
        rounds += 1
        if rounds > 2 * -(-n // BATCH) + 2:
            raise HunchError(f"generate() got only {len(items)} distinct items of {n} after {rounds - 1} batches.")
        want = min(BATCH, n - len(items))
        avoid = [to_jsonable_python(v) for v in items[-50:]]
        label = f"generate {n} {name}" if n <= BATCH else f"generate {name} {len(items)}/{n}"
        for value in _draw(model, jev, target, want, instructions, context, avoid, label):
            fingerprint = json.dumps(to_jsonable_python(value), sort_keys=True, default=str)
            if fingerprint not in seen:
                seen.add(fingerprint)
                items.append(value)
    items = items[:n]
    jev.cache.set(key, [to_jsonable_python(v) for v in items])
    return items


def _draw(model: Any, jev: Client, target: Any, n: int, instructions: str | None, context: Any,
          avoid: list[Any], label: str) -> Any:
    """One validated LLM call for n items (or one value when n == 1), with one retry."""
    adapter: TypeAdapter[Any] = TypeAdapter(list[target] if n > 1 else target)  # type: ignore[valid-type]
    system = (
        "You generate example data. Reply with ONLY a JSON value that matches this JSON Schema. "
        "No prose, no markdown fences, no keys beyond the schema.\n"
        f"{json.dumps(adapter.json_schema())}"
    )
    if n > 1:
        system += f"\nReturn a JSON array of exactly {n} distinct items."
    payload = {"instructions": instructions, "context": to_jsonable_python(context), "n": n,
               "already_have_do_not_repeat": avoid or None}
    user = json.dumps({k: v for k, v in payload.items() if v is not None})

    last_error = ""
    for _ in range(2):
        prompt = user if not last_error else f"{user}\n\nYour previous reply was invalid: {last_error}\nReply again with only valid JSON."
        with engine.working(jev, label):
            text = model.complete(system=system, user=prompt)
        try:
            value = _parse(adapter, text, target, n)
        except (ValidationError, ValueError) as error:
            last_error = str(error)[:400]
            continue
        if n > 1 and len(value) < n:
            last_error = f"expected {n} items, got {len(value)}"
            continue
        return value[:n] if n > 1 else value
    raise HunchError(f"generate() could not get valid {target!r} from the model: {last_error}")


async def generate_async(*args: Any, **kwargs: Any) -> Any:
    return await asyncio.to_thread(generate, *args, **kwargs)


def _parse(adapter: TypeAdapter[Any], text: str, target: Any, n: int) -> Any:
    raw = _strip_fence(text)
    try:
        return adapter.validate_json(raw)
    except ValidationError:
        if target is str and n == 1:
            return raw  # the model answered in plain text; that is a valid str
        raise


def _strip_fence(raw: str) -> str:
    text = raw.strip()
    if not text.startswith("```"):
        return text
    lines = text.splitlines()[1:]
    if lines and lines[-1].strip() == "```":
        lines = lines[:-1]
    return "\n".join(lines).strip()
