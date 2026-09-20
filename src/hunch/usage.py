from __future__ import annotations

import threading
from dataclasses import dataclass


@dataclass
class Usage:
    """Tallies for Jev calls on this client. Cache hits do not add tokens."""

    calls: int = 0
    hits: int = 0
    input_tokens: int = 0
    output_tokens: int = 0
    model: str | None = None

    def snapshot(self) -> Usage:
        return Usage(
            calls=self.calls,
            hits=self.hits,
            input_tokens=self.input_tokens,
            output_tokens=self.output_tokens,
            model=self.model,
        )


class Meter:
    def __init__(self) -> None:
        self._lock = threading.Lock()
        self._usage = Usage()

    def hit(self) -> None:
        with self._lock:
            self._usage.hits += 1

    def call(self, raw: object) -> None:
        model = getattr(raw, "model", None)
        blob = getattr(raw, "usage", None)
        incoming = getattr(blob, "input_tokens", None) if blob is not None else None
        outgoing = getattr(blob, "output_tokens", None) if blob is not None else None
        if incoming is None and isinstance(blob, dict):
            incoming = blob.get("input_tokens")
            outgoing = blob.get("output_tokens")
        with self._lock:
            self._usage.calls += 1
            if isinstance(model, str) and model:
                self._usage.model = model
            if incoming is not None:
                self._usage.input_tokens += int(incoming)
            if outgoing is not None:
                self._usage.output_tokens += int(outgoing)

    def snapshot(self) -> Usage:
        with self._lock:
            return self._usage.snapshot()
