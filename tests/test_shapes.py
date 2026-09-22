import pytest

from hunch.shapes import ShapePolicy


def test_sure_when_peaked() -> None:
    assert ShapePolicy().classify({"a": 0.92, "b": 0.05, "c": 0.03}) == "sure"


def test_split_when_two_close_and_they_hold_the_mass() -> None:
    assert ShapePolicy().classify({"a": 0.48, "b": 0.44, "c": 0.08}) == "split"
    assert ShapePolicy().classify({"a": 0.6, "b": 0.4}) == "split"  # a leader, not decisive


def test_unsure_when_flat() -> None:
    assert ShapePolicy().classify({"a": 0.34, "b": 0.33, "c": 0.33}) == "unsure"
    assert ShapePolicy(unsure_peak=0.7).classify({"a": 0.6, "b": 0.4}) == "unsure"


def test_confidence_is_ignored() -> None:
    probs = {"a": 0.92, "b": 0.08}
    assert ShapePolicy().classify(probs, 0.0) == ShapePolicy().classify(probs, 1.0) == "sure"


def test_validation() -> None:
    with pytest.raises(ValueError):
        ShapePolicy(unsure_peak=0.9, sure_peak=0.8)
