"""Raw Jev answers, keyed by state + question. Memory always; hashed JSON files under cache=."""

from __future__ import annotations

import hashlib
import json
import threading
from pathlib import Path
from typing import Any


class Cache:
    def __init__(self, path: Path | None = None) -> None:
        self.path = Path(path).expanduser() if path is not None else None
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
            value = json.loads(file.read_text())
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
            tmp.write_text(json.dumps(value, sort_keys=True))
            tmp.replace(file)

    def clear(self) -> None:
        with self._lock:
            self._mem.clear()
            if self.path is None or not self.path.exists():
                return
            for child in self.path.rglob("*.json"):
                child.unlink()

    def _file(self, key: str) -> Path:
        digest = hashlib.sha256(key.encode()).hexdigest()
        return self.path / digest[:2] / f"{digest}.json"  # type: ignore[operator]
