"""A Jev connection plus cache, usage meter, shape policy, rate limit, and optional LLM."""

from __future__ import annotations

import os
import threading
import time
from pathlib import Path
from typing import Any

from hunch.cache import Cache
from hunch.exceptions import HunchError
from hunch.shapes import ShapePolicy
from hunch.usage import Meter, Usage


class Client:
    def __init__(
        self,
        api_key: str | None = None,
        *,
        model: str | None = None,
        client: Any | None = None,
        async_client: Any | None = None,
        llm: Any | None = None,
        cache: str | Path | None = None,
        policy: ShapePolicy | None = None,
        max_workers: int = 8,
        max_concurrency: int = 64,
        max_rps: float | None = None,
        errors: str = "raise",
        progress: bool | str = "auto",
        gateway: str = "typesafe",
    ) -> None:
        """Open a Jev client. Reads TYPESAFE_API_KEY and TYPESAFE_DEFAULT_MODEL when omitted.

        client= / async_client= inject objects with a system_one(state=, questions=) method.
        llm= is a LanguageModel for generate(). cache= persists answers on disk.
        max_workers= threads for sync calls; max_concurrency= requests in flight for _async calls.
        max_rps= caps requests per second across both. errors="skip" turns a failed request
        into None for its rows (with a warning) instead of raising.
        progress= shows a progress bar per verb call: True, False, or "auto" (10+ requests).
        gateway= reaches Jev through "openrouter" (OPENROUTER_API_KEY) or "vercel"
        (AI_GATEWAY_API_KEY) instead of TypeSafe; api_key= and model= apply to the gateway.
        """
        if errors not in ("raise", "skip"):
            raise HunchError('errors= must be "raise" or "skip".')
        if max_rps is not None and max_rps <= 0:
            raise HunchError("max_rps= must be positive.")
        self._sdk: dict[str, Any] = {}
        key = api_key if api_key is not None else os.environ.get("TYPESAFE_API_KEY")
        if key:
            self._sdk["api_key"] = key
        chosen = model or os.environ.get("TYPESAFE_DEFAULT_MODEL")
        if chosen:
            self._sdk["model"] = chosen
        if client is None and gateway != "typesafe":
            from hunch.gateway import make

            client = make(gateway, api_key, model)
        self.injected = client is not None
        if client is None:
            from typesafe_sdk import TypeSafeClient

            client = TypeSafeClient(**self._sdk)
        self.jev = client
        self._async = async_client
        from hunch.llm import as_llm

        self.llm = as_llm(llm)
        self.cache = Cache(cache)
        self.policy = policy or ShapePolicy()
        self.max_workers = max(1, max_workers)
        self.max_concurrency = max(1, max_concurrency)
        self.max_rps = max_rps
        self.errors = errors
        self.progress = progress
        self.meter = Meter()
        self._rate_lock = threading.Lock()
        self._next_slot = 0.0

    @property
    def usage(self) -> Usage:
        return self.meter.snapshot()

    def clear_cache(self) -> None:
        self.cache.clear()

    def async_jev(self) -> Any:
        """The async SDK client, built on first use. None when only a sync client was injected."""
        if self._async is None:
            if self.injected:
                return None
            from typesafe_sdk import AsyncTypeSafeClient

            self._async = AsyncTypeSafeClient(**self._sdk)
        return self._async

    def wait_for_slot(self) -> float:
        """Reserve the next request slot under max_rps. Returns seconds to wait (0 if none)."""
        if not self.max_rps:
            return 0.0
        with self._rate_lock:
            now = time.monotonic()
            start = max(now, self._next_slot)
            self._next_slot = start + 1.0 / self.max_rps
        return start - now


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
