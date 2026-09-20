"""A Jev connection plus cache, usage meter, shape policy, and optional LLM."""

from __future__ import annotations

import os
from pathlib import Path
from typing import Any

from hunch.cache import Cache
from hunch.shapes import ShapePolicy
from hunch.usage import Meter, Usage


class Client:
    def __init__(
        self,
        api_key: str | None = None,
        *,
        model: str | None = None,
        client: Any | None = None,
        llm: Any | None = None,
        cache: str | Path | None = None,
        policy: ShapePolicy | None = None,
        max_workers: int = 8,
        progress: bool | str = "auto",
    ) -> None:
        """Open a Jev client. Reads TYPESAFE_API_KEY and TYPESAFE_DEFAULT_MODEL when omitted.

        client= injects any object with a system_one(state=, questions=) method (tests, fakes).
        llm= is a LanguageModel for generate(). cache= persists answers on disk.
        progress= shows a progress bar per verb call: True, False, or "auto" (10+ requests).
        """
        if client is None:
            from typesafe_sdk import TypeSafeClient

            kwargs: dict[str, Any] = {}
            key = api_key if api_key is not None else os.environ.get("TYPESAFE_API_KEY")
            if key:
                kwargs["api_key"] = key
            chosen = model or os.environ.get("TYPESAFE_DEFAULT_MODEL")
            if chosen:
                kwargs["model"] = chosen
            client = TypeSafeClient(**kwargs)
        self.jev = client
        self.llm = llm
        self.cache = Cache(cache)
        self.policy = policy or ShapePolicy()
        self.max_workers = max(1, max_workers)
        self.progress = progress
        self.meter = Meter()

    @property
    def usage(self) -> Usage:
        return self.meter.snapshot()

    def clear_cache(self) -> None:
        self.cache.clear()


_default: Client | None = None


def configure(*args: Any, **kwargs: Any) -> Client:
    """Build the module-level default client. Same arguments as Client()."""
    global _default
    _default = Client(*args, **kwargs)
    return _default


def default() -> Client:
    global _default
    if _default is None:
        _default = Client()
    return _default


def resolve(client: Client | None) -> Client:
    return client if client is not None else default()
