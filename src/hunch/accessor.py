"""df.hunch.<verb>(...) and series.hunch.<verb>(...). Registered on import when pandas is present."""

from __future__ import annotations

from typing import Any

import pandas as pd

from hunch import verbs


@pd.api.extensions.register_dataframe_accessor("hunch")
@pd.api.extensions.register_series_accessor("hunch")
class HunchAccessor:
    def __init__(self, obj: Any) -> None:
        self._obj = obj

    def ask(self, questions: Any, **kw: Any) -> Any:
        return verbs.ask(self._obj, questions, **kw)

    def classify(self, labels: Any, **kw: Any) -> Any:
        return verbs.classify(self._obj, labels, **kw)

    def score(self, levels: Any, **kw: Any) -> Any:
        return verbs.score(self._obj, levels, **kw)

    def check(self, statement: Any, **kw: Any) -> Any:
        return verbs.check(self._obj, statement, **kw)

    def where(self, statement: str, **kw: Any) -> Any:
        return verbs.where(self._obj, statement, **kw)

    def verify(self, source: Any, **kw: Any) -> Any:
        return verbs.verify(self._obj, source, **kw)

    def discover(self, n: int = 8, **kw: Any) -> Any:
        from hunch.combine import discover

        return discover(self._obj, n, **kw)

    def refine(self, checks: Any, **kw: Any) -> Any:
        from hunch.combine import refine

        return refine(self._obj, checks, **kw)

    def pick(self, instructions: str, **kw: Any) -> Any:
        return verbs.pick(self._obj, instructions, **kw)

    def rank(self, dimensions: Any, levels: Any, **kw: Any) -> Any:
        return verbs.rank(self._obj, dimensions, levels, **kw)
