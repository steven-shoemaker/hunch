from hunch.shapes import ShapePolicy


def test_sure_when_peaked() -> None:
    policy = ShapePolicy()
    assert policy.classify({"a": 0.92, "b": 0.05, "c": 0.03}, 0.88) == "sure"


def test_torn_when_two_close() -> None:
    policy = ShapePolicy()
    assert policy.classify({"a": 0.48, "b": 0.44, "c": 0.08}, 0.50) == "split"


def test_lost_when_flat() -> None:
    policy = ShapePolicy()
    assert policy.classify({"a": 0.34, "b": 0.33, "c": 0.33}, 0.01) == "unsure"
