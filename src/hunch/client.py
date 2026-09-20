from __future__ import annotations

import os
import threading
from pathlib import Path
from typing import Any

from typesafe_sdk import TypeSafeClient

from hunch.cache import Cache
from hunch.session import activate
from hunch.shapes import ShapePolicy
from hunch.usage import Meter, Usage


class Hunch:
    """A Jev connection plus a cache shared across @over runs."""

    def __init__(
        self,
        client: Any,
        *,
        policy: ShapePolicy | None = None,
        llm: Any | None = None,
        cache: Cache | None = None,
    ) -> None:
        self.client = client
        self.policy = policy or ShapePolicy()
        self.llm = llm
        self.cache = cache or Cache()
        self.meter = Meter()
        self._lock = threading.Lock()

    def session(self):
        return activate(
            self.client,
            self.policy,
            self.cache,
            self._lock,
            llm=self.llm,
            meter=self.meter,
        )

    def clear_cache(self) -> None:
        with self._lock:
            self.cache.clear()

    @property
    def usage(self) -> Usage:
        return self.meter.snapshot()


def connect(
    api_key: str | None = None,
    *,
    client: Any | None = None,
    model: str | None = None,
    policy: ShapePolicy | None = None,
    llm: Any | None = None,
    cache: str | Path | None = None,
) -> Hunch:
    """Open a Jev client. Reads TYPESAFE_API_KEY when api_key is omitted."""
    store = Cache(Path(cache).expanduser() if cache is not None else None)
    if client is not None:
        return Hunch(client, policy=policy, llm=llm, cache=store)
    key = api_key if api_key is not None else os.environ.get("TYPESAFE_API_KEY")
    kwargs: dict[str, Any] = {}
    if key:
        kwargs["api_key"] = key
    chosen = model or os.environ.get("TYPESAFE_DEFAULT_MODEL")
    if chosen:
        kwargs["model"] = chosen
    return Hunch(TypeSafeClient(**kwargs), policy=policy, llm=llm, cache=store)
