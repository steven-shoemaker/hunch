from __future__ import annotations

from typing import Any


class Subject:
    """A cell value with .feels() — the thing @over passes into classify()."""

    def __init__(self, value: Any, field: str = "value") -> None:
        self.value = value
        self.field = field

    def feels(self, description: str, *, true: str | None = None, false: str | None = None):
        from hunch.session import current_session

        return current_session().noul(self, description, true=true, false=false)

    def as_state(self) -> dict[str, Any]:
        return {self.field: self.value}

    def __str__(self) -> str:
        return str(self.value)

    def __repr__(self) -> str:
        return f"Subject({self.value!r}, field={self.field!r})"

    def __eq__(self, other: object) -> bool:
        if isinstance(other, Subject):
            return self.value == other.value and self.field == other.field
        return self.value == other

    def __hash__(self) -> int:
        return hash((self.field, _freeze(self.value)))


def encode_state(value: Any) -> Any:
    from hunch.role import Draft

    if isinstance(value, Subject):
        return value.as_state()
    if isinstance(value, Draft):
        return value.as_state()
    if isinstance(value, (dict, list)):
        return value
    return {"value": value}


def _freeze(value: Any) -> Any:
    if isinstance(value, dict):
        return tuple(sorted((k, _freeze(v)) for k, v in value.items()))
    if isinstance(value, list):
        return tuple(_freeze(v) for v in value)
    return value
