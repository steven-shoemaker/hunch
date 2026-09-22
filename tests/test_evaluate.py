from __future__ import annotations

from enum import Enum

import pytest

from hunch import Answer, Feeling, HunchError, evaluate, tune_threshold

pd = pytest.importorskip("pandas")


def test_evaluate_labels_with_confusion_and_misses() -> None:
    ev = evaluate(["Sales", "Sales", "Eng", None], ["Sales", "Eng", "Eng", "Eng"])
    assert ev.n == 3 and ev.missing == 1
    assert ev.accuracy == pytest.approx(2 / 3)
    assert ev.confusion[("Eng", "Sales")] == 1
    assert ev.errors == [(1, "Eng", "Sales")]
    assert ev.table().loc["Eng", "Sales"] == 1


def test_by_shape_shows_whether_sure_means_right() -> None:
    detail = pd.DataFrame(
        {"label": ["a", "a", "b", "a"], "label_shape": ["sure", "sure", "split", "unsure"]}, index=[10, 11, 12, 13]
    )
    truth = pd.Series(["a", "a", "a", "b"], index=[10, 11, 12, 13])
    ev = evaluate(detail, truth)
    assert ev.by_shape == {"sure": (2, 1.0), "split": (1, 0.0), "unsure": (1, 0.0)}
    assert "sure: 100% of 2" in repr(ev)


def test_evaluate_accepts_answers_and_enums() -> None:
    class F(Enum):
        S = "Sales"

    preds = [Answer(F.S, {"Sales": 0.9}, 0.9, "sure")]
    assert evaluate(preds, ["Sales"]).accuracy == 1.0
    with pytest.raises(HunchError, match="length"):
        evaluate(["a"], ["a", "b"])


def test_tune_threshold_for_precision_recall_and_f1() -> None:
    p = [0.95, 0.9, 0.8, 0.7, 0.6, 0.4, 0.3]
    truth = [True, True, False, True, False, True, False]
    hi = tune_threshold(p, truth, precision=1.0)
    assert hi.threshold == 0.9 and hi.recall == 0.5  # lowest cutoff that is still all-correct
    catch = tune_threshold(p, truth, recall=0.75)
    assert catch.threshold == 0.7 and catch.precision == 0.75
    assert 0 < float(tune_threshold(p, truth)) <= 0.95
    with pytest.raises(HunchError, match="No cutoff reaches"):
        tune_threshold([0.9, 0.8], [False, True], precision=1.0)


def test_tune_threshold_reads_detail_frames_and_feelings() -> None:
    frame = pd.DataFrame({"match": [True, False], "match_p": [0.9, 0.2]}, index=["x", "y"])
    assert tune_threshold(frame, pd.Series([True, False], index=["x", "y"])).threshold == 0.9
    assert tune_threshold([Feeling(0.9), Feeling(0.2)], [True, False]).precision == 1.0
