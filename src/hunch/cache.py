from __future__ import annotations

import hashlib
import json
import threading
from pathlib import Path
from typing import Any

from hunch.answer import Answer, Feeling, Rating
from hunch.role import Draft, Role


class Cache:
    """Memory cache with optional hashed files under cache=."""

    def __init__(self, path: Path | None = None) -> None:
        self.path = Path(path) if path is not None else None
        self._mem: dict[str, Any] = {}
        self._lock = threading.Lock()
        if self.path is not None:
            self.path.mkdir(parents=True, exist_ok=True)

    def get(self, key: str) -> Any | None:
        with self._lock:
            if key in self._mem:
                return self._mem[key]
            if self.path is None:
                return None
            file = self._file(key)
            if not file.is_file():
                return None
            payload = json.loads(file.read_text())
            value = decode(payload)
            self._mem[key] = value
            return value

    def set(self, key: str, value: Any) -> None:
        with self._lock:
            self._mem[key] = value
            if self.path is None:
                return
            file = self._file(key)
            file.parent.mkdir(parents=True, exist_ok=True)
            tmp = file.with_suffix(".tmp")
            tmp.write_text(json.dumps(encode(value), sort_keys=True))
            tmp.replace(file)

    def clear(self) -> None:
        with self._lock:
            self._mem.clear()
            if self.path is None or not self.path.exists():
                return
            for child in self.path.rglob("*.json"):
                if child.name == "taxonomy.json":
                    continue
                child.unlink()

    def _file(self, key: str) -> Path:
        digest = hashlib.sha256(key.encode()).hexdigest()
        return self.path / digest[:2] / f"{digest}.json"  # type: ignore[operator]


def encode(value: Any) -> dict[str, Any]:
    if isinstance(value, Answer):
        return {
            "type": "choice",
            "top": value.top,
            "probabilities": dict(value.probabilities),
            "confidence": value.confidence,
            "shape": value.shape,
        }
    if isinstance(value, Feeling):
        return {"type": "noul", "p": value.p}
    if isinstance(value, Rating):
        return {
            "type": "score",
            "score": value.score,
            "confidence": value.confidence,
            "legend": {str(k): v for k, v in value.legend.items()},
            "probabilities": {str(k): v for k, v in value.probabilities.items()},
            "shape": value.shape,
        }
    if isinstance(value, Draft):
        return {
            "type": "draft",
            "text": value.text,
            "labels": list(value.labels),
            "instructions": value.role.instructions,
            "emit": value.role.emit,
            "max_loops": value.role.max_loops,
        }
    raise TypeError(f"Cannot cache {type(value)!r}.")


def decode(payload: dict[str, Any]) -> Any:
    kind = payload.get("type")
    if kind == "choice":
        return Answer(
            top=str(payload["top"]),
            probabilities=payload["probabilities"],
            confidence=float(payload["confidence"]),
            shape=payload["shape"],
        )
    if kind == "noul":
        return Feeling(p=float(payload["p"]))
    if kind == "score":
        legend = {int(k): str(v) for k, v in payload["legend"].items()}
        probabilities = {int(k): float(v) for k, v in payload["probabilities"].items()}
        return Rating(
            score=float(payload["score"]),
            confidence=float(payload["confidence"]),
            legend=legend,
            probabilities=probabilities,
            shape=payload["shape"],
        )
    if kind == "draft":
        return Draft(
            text=str(payload["text"]),
            labels=list(payload["labels"]),
            role=Role(
                instructions=str(payload["instructions"]),
                emit=payload["emit"],
                max_loops=int(payload["max_loops"]),
            ),
        )
    raise ValueError(f"Unknown cache entry {kind!r}.")
