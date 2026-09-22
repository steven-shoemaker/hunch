from __future__ import annotations

from typing import Literal, Mapping

Shape = Literal["sure", "split", "unsure"]


class ShapePolicy:
    """Map a distribution onto sure / split / unsure from the probabilities alone.

    Jev's confidence is derived from the top probability, so it adds nothing here.
    These cutoffs are yours, not Jev's, and none of them says whether a label is true.

    split:  the top two are within split_margin and together hold at least split_mass.
    unsure: otherwise, the top option is below unsure_peak.
    sure:   the top option reaches sure_peak.
    Anything else is split: a leader, but not a decisive one.
    """

    def __init__(
        self,
        *,
        sure_peak: float = 0.80,
        unsure_peak: float = 0.50,
        split_margin: float = 0.15,
        split_mass: float = 0.75,
    ) -> None:
        if not 0 <= unsure_peak <= sure_peak <= 1:
            raise ValueError("Need 0 <= unsure_peak <= sure_peak <= 1.")
        for name, value in (("split_margin", split_margin), ("split_mass", split_mass)):
            if not 0 <= value <= 1:
                raise ValueError(f"{name} must be between 0 and 1.")
        self.sure_peak = sure_peak
        self.unsure_peak = unsure_peak
        self.split_margin = split_margin
        self.split_mass = split_mass

    def classify(self, probabilities: Mapping[str, float], confidence: float | None = None) -> Shape:
        ranked = sorted(probabilities.values(), reverse=True)
        top = ranked[0] if ranked else 0.0
        second = ranked[1] if len(ranked) > 1 else 0.0
        if top - second < self.split_margin and top + second >= self.split_mass:
            return "split"
        if top < self.unsure_peak:
            return "unsure"
        if top >= self.sure_peak:
            return "sure"
        return "split"
