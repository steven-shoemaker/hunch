from __future__ import annotations

from typing import Literal, Mapping

Shape = Literal["sure", "split", "unsure"]


class ShapePolicy:
    """Map a Choice distribution onto sure / split / unsure.

    These cutoffs are yours, not Jev's. Confidence is how peaked the
    distribution is, not whether the label is true.
    """

    def __init__(
        self,
        *,
        sure_confidence: float = 0.75,
        unsure_confidence: float = 0.35,
        split_margin: float = 0.15,
        unsure_peak: float = 0.40,
    ) -> None:
        if not 0 <= unsure_confidence <= sure_confidence <= 1:
            raise ValueError("Need 0 <= unsure_confidence <= sure_confidence <= 1.")
        if not 0 <= split_margin <= 1:
            raise ValueError("split_margin must be between 0 and 1.")
        if not 0 <= unsure_peak <= 1:
            raise ValueError("unsure_peak must be between 0 and 1.")
        self.sure_confidence = sure_confidence
        self.unsure_confidence = unsure_confidence
        self.split_margin = split_margin
        self.unsure_peak = unsure_peak

    def classify(self, probabilities: Mapping[str, float], confidence: float) -> Shape:
        ranked = sorted(probabilities.values(), reverse=True)
        peak = ranked[0] if ranked else 0.0
        second = ranked[1] if len(ranked) > 1 else 0.0
        if confidence < self.unsure_confidence or peak < self.unsure_peak:
            return "unsure"
        if peak - second < self.split_margin:
            return "split"
        if confidence >= self.sure_confidence:
            return "sure"
        return "split"
