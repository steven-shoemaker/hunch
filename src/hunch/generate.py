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
from hunch.llm import LanguageModel


def generate(
    target: Any = str,
    n: int = 1,
    *,
    instructions: str | None = None,
    context: Any = None,
    llm: LanguageModel | None = None,
    client: Client | None = None,
) -> Any:
    """Create n examples of target: str, int, a dataclass, a TypedDict, list[str], ...

    Returns one value when n == 1, else a list of exactly n. Uses llm= here, else the
    client's llm= from configure(). Output is validated against the target type.
    """
    if n < 1:
        raise HunchError("generate() needs n >= 1.")
    jev = resolve(client)
    model = llm or jev.llm
    if model is None:
        raise HunchError("generate() needs a language model: pass llm= here or on configure().")
    adapter: TypeAdapter[Any] = TypeAdapter(list[target] if n > 1 else target)  # type: ignore[valid-type]
    schema = adapter.json_schema()
    system = (
        "You generate example data. Reply with ONLY a JSON value that matches this JSON Schema. "
        "No prose, no markdown fences, no keys beyond the schema.\n"
        f"{json.dumps(schema)}"
    )
    if n > 1:
        system += f"\nReturn a JSON array of exactly {n} distinct items."
    payload = {"instructions": instructions, "context": to_jsonable_python(context), "n": n}
    user = json.dumps({k: v for k, v in payload.items() if v is not None})

    last_error = ""
    for attempt in range(2):
        prompt = user if not last_error else f"{user}\n\nYour previous reply was invalid: {last_error}\nReply again with only valid JSON."
        with engine.working(jev, f"generate {n} {getattr(target, '__name__', target)}"):
            text = model.complete(system=system, user=prompt)
        try:
            value = _parse(adapter, text, target, n)
        except (ValidationError, ValueError) as error:
            last_error = str(error)[:400]
            continue
        if n > 1:
            if len(value) < n:
                last_error = f"expected {n} items, got {len(value)}"
                continue
            return value[:n]
        return value
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
