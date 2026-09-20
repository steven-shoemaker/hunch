from __future__ import annotations

from typing import Literal, Mapping

Shape = Literal["sure", "torn", "lost"]


class ShapePolicy:
    """Map a Choice distribution onto sure / torn / lost.

    These cutoffs are yours, not Jev's. Confidence is how peaked the
    distribution is, not whether the label is true.
    """

    def __init__(
        self,
        *,
        sure_confidence: float = 0.75,
        lost_confidence: float = 0.35,
        torn_margin: float = 0.15,
        lost_peak: float = 0.40,
    ) -> None:
        if not 0 <= lost_confidence <= sure_confidence <= 1:
            raise ValueError("Need 0 <= lost_confidence <= sure_confidence <= 1.")
        if not 0 <= torn_margin <= 1:
            raise ValueError("torn_margin must be between 0 and 1.")
        if not 0 <= lost_peak <= 1:
            raise ValueError("lost_peak must be between 0 and 1.")
        self.sure_confidence = sure_confidence
        self.lost_confidence = lost_confidence
        self.torn_margin = torn_margin
        self.lost_peak = lost_peak

    def classify(self, probabilities: Mapping[str, float], confidence: float) -> Shape:
        ranked = sorted(probabilities.values(), reverse=True)
        peak = ranked[0] if ranked else 0.0
        second = ranked[1] if len(ranked) > 1 else 0.0
        if confidence < self.lost_confidence or peak < self.lost_peak:
            return "lost"
        if peak - second < self.torn_margin:
            return "torn"
        if confidence >= self.sure_confidence:
            return "sure"
        return "torn"
