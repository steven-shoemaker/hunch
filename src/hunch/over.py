from __future__ import annotations

from collections.abc import Callable
from concurrent.futures import ThreadPoolExecutor, as_completed
from typing import Any

import pandas as pd

from hunch.client import Hunch
from hunch.exceptions import HunchError
from hunch.subject import Subject


class OverJob:
    def __init__(self, frame: pd.DataFrame, column: str, fn: Callable[[Subject], Any]) -> None:
        self.frame = frame
        self.column = column
        self.fn = fn

    def run(
        self,
        jev: Hunch,
        *,
        max_workers: int = 1,
        on_item: Callable[[int, int, Any], None] | None = None,
    ) -> pd.DataFrame:
        if self.column not in self.frame.columns:
            raise HunchError(f"Column {self.column!r} is not in the frame.")
        series = self.frame[self.column]
        unique = [value for value in series.dropna().unique().tolist()]
        total = len(unique)
        rows: list[dict[str, Any]] = []

        def one(value: Any) -> dict[str, Any]:
            with jev.session():
                result = self.fn(Subject(value, field=self.column))
            return _row(self.column, value, result)

        if max_workers <= 1:
            for index, value in enumerate(unique, start=1):
                row = one(value)
                rows.append(row)
                if on_item:
                    on_item(index, total, value)
        else:
            done = 0
            with ThreadPoolExecutor(max_workers=max_workers) as pool:
                futures = [pool.submit(one, value) for value in unique]
                for future in as_completed(futures):
                    rows.append(future.result())
                    done += 1
                    if on_item:
                        on_item(done, total, None)

        if not rows:
            return self.frame.copy()
        classified = pd.DataFrame(rows)
        return self.frame.merge(classified, on=self.column, how="left")


def over(frame: pd.DataFrame, column: str) -> Callable[[Callable[[Subject], Any]], OverJob]:
    """Classify distinct values in a column, then join the answers back."""

    def decorator(fn: Callable[[Subject], Any]) -> OverJob:
        return OverJob(frame, column, fn)

    return decorator


def _row(column: str, value: Any, result: Any) -> dict[str, Any]:
    if isinstance(result, dict):
        return {column: value, **result}
    return {column: value, "result": result}
