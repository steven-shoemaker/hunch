"""Measure hunch against labeled data: how often it's right, where it's wrong, which cutoff to use."""

from __future__ import annotations

from collections import Counter
from dataclasses import dataclass, field
from typing import Any

from hunch.answer import Answer, Feeling
from hunch.exceptions import HunchError


@dataclass
class Evaluation:
    """How predicted labels compare with the truth."""

    n: int
    """Rows compared. Rows with no prediction (skipped requests) are counted in `missing`."""
    accuracy: float
    confusion: dict[tuple[str, str], int]
    """(truth, predicted) -> count."""
    by_shape: dict[str, tuple[int, float]] = field(default_factory=dict)
    """shape -> (rows, accuracy). Empty unless the predictions carry shapes (detail=True)."""
    errors: list[tuple[Any, str, str]] = field(default_factory=list)
    """(row key, truth, predicted) for every miss."""
    missing: int = 0

    def __repr__(self) -> str:
        shapes = ", ".join(f"{s}: {acc:.0%} of {k}" for s, (k, acc) in self.by_shape.items())
        extra = f", by shape: {shapes}" if shapes else ""
        return f"Evaluation(accuracy={self.accuracy:.1%} on {self.n} rows{extra}, missing={self.missing})"

    def table(self) -> Any:
        """Confusion matrix as a DataFrame: truth down the side, predicted across the top."""
        import pandas as pd

        truths = sorted({t for t, _ in self.confusion})
        preds = sorted({p for _, p in self.confusion})
        return pd.DataFrame(
            [[self.confusion.get((t, p), 0) for p in preds] for t in truths],
            index=pd.Index(truths, name="truth"),
            columns=pd.Index(preds, name="predicted"),
        )


@dataclass
class Threshold:
    """A cutoff for check() / where(), with what it achieves on the labeled rows."""

    threshold: float
    precision: float
    recall: float
    n: int
    positives: int

    def __float__(self) -> float:
        return self.threshold


def evaluate(predicted: Any, truth: Any) -> Evaluation:
    """Compare classify() output with true labels.

    predicted: labels, Answer objects, or the DataFrame from classify(..., detail=True).
    truth: the correct labels, aligned by position (or by index when both are pandas).
    With shapes available, by_shape shows whether "sure" rows really are more accurate.
    """
    rows = _align(_predictions(predicted), truth)
    confusion: Counter[tuple[str, str]] = Counter()
    shapes: dict[str, list[bool]] = {}
    errors: list[tuple[Any, str, str]] = []
    missing = 0
    for key, (label, shape), actual in rows:
        if label is None:
            missing += 1
            continue
        got, want = _text(label), _text(actual)
        confusion[(want, got)] += 1
        if shape is not None:
            shapes.setdefault(shape, []).append(got == want)
        if got != want:
            errors.append((key, want, got))
    n = sum(confusion.values())
    if n == 0:
        raise HunchError("evaluate() found no rows with both a prediction and a truth.")
    correct = sum(c for (t, p), c in confusion.items() if t == p)
    order = {"sure": 0, "split": 1, "unsure": 2}
    by_shape = {s: (len(v), sum(v) / len(v)) for s, v in sorted(shapes.items(), key=lambda kv: order.get(kv[0], 9))}
    return Evaluation(n, correct / n, dict(confusion), by_shape, errors, missing)


def tune_threshold(
    probabilities: Any,
    truth: Any,
    *,
    precision: float | None = None,
    recall: float | None = None,
) -> Threshold:
    """Find the cutoff for check() / where() from labeled rows.

    probabilities: P(yes) per row, as floats, Feelings, or the DataFrame from
    check(..., detail=True) / where(..., detail=True). truth: True/False per row.
    precision=0.9 returns the lowest cutoff that keeps at least 90% of matches correct,
    which lets through as many true matches as possible. recall=0.9 returns the highest
    cutoff that still catches 90% of true matches. With neither, it maximizes F1.
    """
    if precision is not None and recall is not None:
        raise HunchError("Pass precision= or recall=, not both.")
    ps = _probs(probabilities)
    rows = [(float(p), bool(t)) for _, (p, _), t in _align([(k, (v, None)) for k, v in ps], truth) if p is not None]
    if not rows:
        raise HunchError("tune_threshold() found no rows with both a probability and a truth.")
    positives = sum(t for _, t in rows)
    if positives == 0:
        raise HunchError("tune_threshold() needs at least one True row in truth.")

    def at(cut: float) -> Threshold:
        kept = [t for p, t in rows if p >= cut]
        hits = sum(kept)
        prec = hits / len(kept) if kept else 1.0
        return Threshold(cut, prec, hits / positives, len(rows), positives)

    candidates = [at(c) for c in sorted({p for p, _ in rows})]
    if precision is not None:
        ok = [c for c in candidates if c.precision >= precision]
        if not ok:
            best = max(candidates, key=lambda c: c.precision)
            raise HunchError(f"No cutoff reaches precision {precision:.0%}; best is {best.precision:.0%} at {best.threshold:.2f}.")
        return min(ok, key=lambda c: c.threshold)
    if recall is not None:
        ok = [c for c in candidates if c.recall >= recall]
        return max(ok, key=lambda c: c.threshold)  # the lowest candidate always has recall 1.0

    def f1(c: Threshold) -> float:
        return 0.0 if c.precision + c.recall == 0 else 2 * c.precision * c.recall / (c.precision + c.recall)

    return max(candidates, key=f1)


# ----------------------------------------------------------------------------- helpers


def _is_pandas(x: Any) -> bool:
    return type(x).__module__.split(".")[0] == "pandas"


def _predictions(predicted: Any) -> list[tuple[Any, tuple[Any, str | None]]]:
    """[(row key, (label, shape or None))]"""
    if _is_pandas(predicted) and hasattr(predicted, "columns"):
        shape_cols = [c for c in predicted.columns if str(c).endswith("_shape")]
        if not shape_cols:
            if len(predicted.columns) != 1:
                raise HunchError("Pass the label column, or the DataFrame from classify(..., detail=True).")
            col, shape_col = predicted.columns[0], None
        else:
            shape_col = shape_cols[0]
            col = str(shape_col)[: -len("_shape")]
        return [(k, (_none(r[col]), None if shape_col is None else _none(r[shape_col]))) for k, r in predicted.iterrows()]
    keys = list(predicted.index) if _is_pandas(predicted) else list(range(len(predicted)))
    values = list(predicted.tolist() if _is_pandas(predicted) else predicted)
    return [(k, (v.label, v.shape) if isinstance(v, Answer) else (_none(v), None)) for k, v in zip(keys, values)]


def _probs(probabilities: Any) -> list[tuple[Any, float | None]]:
    if _is_pandas(probabilities) and hasattr(probabilities, "columns"):
        cols = [c for c in probabilities.columns if str(c).endswith("_p")]
        if not cols:
            raise HunchError("Pass P(yes) values, or the DataFrame from check()/where() with detail=True.")
        return [(k, _none(v)) for k, v in probabilities[cols[-1]].items()]
    keys = list(probabilities.index) if _is_pandas(probabilities) else list(range(len(probabilities)))
    values = list(probabilities.tolist() if _is_pandas(probabilities) else probabilities)
    return [(k, v.p if isinstance(v, Feeling) else _none(v)) for k, v in zip(keys, values)]


def _align(pairs: list[tuple[Any, Any]], truth: Any) -> list[tuple[Any, Any, Any]]:
    if _is_pandas(truth):
        lookup = dict(truth.items())
        if all(k in lookup for k, _ in pairs):
            return [(k, v, lookup[k]) for k, v in pairs]
        truth = truth.tolist()
    truth = list(truth)
    if len(truth) != len(pairs):
        raise HunchError(f"predictions and truth differ in length ({len(pairs)} vs {len(truth)}).")
    return [(k, v, t) for (k, v), t in zip(pairs, truth)]


def _none(v: Any) -> Any:
    try:
        return None if v is None or v != v else v  # NaN != NaN
    except (TypeError, ValueError):
        return v


def _text(label: Any) -> str:
    return str(getattr(label, "value", label))
