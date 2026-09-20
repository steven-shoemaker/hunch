"""Run a fixed set of questions over many states: dedupe, cache, thread pool, normalize answers."""

from __future__ import annotations

import json
from collections.abc import Mapping
from concurrent.futures import ThreadPoolExecutor
from typing import Any

from pydantic_core import to_jsonable_python

from hunch.client import Client
from hunch.exceptions import HunchError

RawAnswer = dict[str, Any]
"""One Jev answer as plain JSON: {"type": "choice"|"noul"|"score", ...wire fields}."""


def jsonable(value: Any) -> Any:
    """Anything Jev can take as state: str, dict, list, dataclass, Enum, pydantic model, ..."""
    return to_jsonable_python(value)


def build_state(item: Any, context: Any) -> Any:
    state: dict[str, Any] = {"input": jsonable(item)}
    if context is None:
        return state
    if isinstance(context, Mapping):
        for key, value in context.items():
            if key == "input":
                raise HunchError("context= cannot use the reserved key 'input'.")
            state[str(key)] = jsonable(value)
    else:
        state["context"] = jsonable(context)
    return state


def run(
    client: Client,
    states: list[Any],
    questions: Mapping[str, Any],
) -> list[dict[str, RawAnswer]]:
    """Answer every question for every state. One request per distinct uncached state."""
    if not questions:
        raise HunchError("No questions to ask.")
    encoded = {qid: _dump(q.model_dump(mode="json")) for qid, q in questions.items()}
    results: list[dict[str, RawAnswer]] = [{} for _ in states]
    todo: dict[str, list[int]] = {}
    for index, state in enumerate(states):
        state_key = _dump(state)
        for qid, qkey in encoded.items():
            cached = client.cache.get(f"{state_key}|{qkey}")
            if cached is not None:
                results[index][qid] = cached
                client.meter.hit()
        if len(results[index]) < len(questions):
            todo.setdefault(state_key, []).append(index)

    def one(state_key: str) -> dict[str, RawAnswer]:
        first = todo[state_key][0]
        missing = {qid: q for qid, q in questions.items() if qid not in results[first]}
        raw = client.jev.system_one(state=states[first], questions=missing)
        client.meter.call(raw)
        answers = {qid: normalize(raw, qid) for qid in missing}
        for qid, answer in answers.items():
            client.cache.set(f"{state_key}|{encoded[qid]}", answer)
        return answers

    keys = list(todo)
    if len(keys) <= 1 or client.max_workers <= 1:
        fetched = [one(key) for key in keys]
    else:
        with ThreadPoolExecutor(max_workers=client.max_workers) as pool:
            fetched = list(pool.map(one, keys))
    for key, answers in zip(keys, fetched):
        for index in todo[key]:
            results[index].update(answers)
    return results


def normalize(raw: Any, qid: str) -> RawAnswer:
    """Lift one answer out of a SystemOneResponse (or a duck-typed fake) into plain JSON."""
    payload = _lookup(_field(raw, "answers"), qid)
    if payload is None:
        for bucket in ("choices", "nouls", "scores"):
            payload = _lookup(_field(raw, bucket), qid)
            if payload is not None:
                break
    if payload is None:
        raise HunchError(f"Jev response is missing an answer for {qid!r}.")
    if _field(payload, "noul") is not None:
        return {"type": "noul", "noul": float(_field(payload, "noul"))}
    if _field(payload, "legend") is not None:
        return {
            "type": "score",
            "score": float(_field(payload, "score")),
            "confidence": float(_field(payload, "confidence") or 0.0),
            "legend": {str(k): str(v) for k, v in dict(_field(payload, "legend")).items()},
            "probabilities": {
                str(k): float(v) for k, v in dict(_field(payload, "probabilities")).items()
            },
        }
    choice = _field(payload, "choice")
    probabilities = _field(payload, "probabilities")
    if choice is None or probabilities is None:
        raise HunchError(f"Jev answer {qid!r} has an unrecognized shape.")
    return {
        "type": "choice",
        "choice": str(choice),
        "confidence": float(_field(payload, "confidence") or 0.0),
        "probabilities": {str(k): float(v) for k, v in dict(probabilities).items()},
    }


def _field(obj: Any, name: str) -> Any:
    if obj is None:
        return None
    if isinstance(obj, Mapping):
        return obj.get(name)
    return getattr(obj, name, None)


def _lookup(container: Any, qid: str) -> Any:
    return _field(container, qid)


def _dump(value: Any) -> str:
    return json.dumps(value, sort_keys=True, default=str)
