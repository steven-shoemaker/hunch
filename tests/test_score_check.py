from __future__ import annotations

import pytest

from hunch import Client, Feeling, HunchError, Rating, check, score
from tests.fakes import FakeJev, noul_answer, response, score_answer

LEVELS = ["weak", "okay", "strong", "excellent"]
LEGEND = dict(enumerate(LEVELS))


def scorer(state, questions):
    answers = {}
    for qid, q in questions.items():
        assert q.type == "score" and list(q.criteria) == LEVELS
        value = 2.6 if qid in ("q", "hook") else 0.4
        probs = {0: 0.05, 1: 0.05, 2: 0.2, 3: 0.7} if value > 2 else {0: 0.7, 1: 0.2, 2: 0.05, 3: 0.05}
        answers[qid] = score_answer(value, probs, 0.8, LEGEND)
    return response(**answers)


def test_score_returns_position_and_detail_returns_rating() -> None:
    fake = FakeJev(scorer)
    jev = Client(client=fake)
    assert score("great tweet", LEVELS, instructions="How strong is the hook?", client=jev) == pytest.approx(2.6)
    rating = score("great tweet", LEVELS, instructions="How strong is the hook?", detail=True, client=jev)
    assert isinstance(rating, Rating)
    assert rating.level == "excellent"
    assert rating.normalized == pytest.approx(2.6 / 3)
    assert rating.shape == "sure"
    assert len(fake.calls) == 1  # detail came from cache
    assert fake.calls[0][1]["q"].instructions == "How strong is the hook?"


def test_dimensions_go_in_one_request() -> None:
    fake = FakeJev(scorer)
    out = score(
        ["t1", "t2"],
        LEVELS,
        instructions={"hook": "How strong is the hook?", "clarity": "How clear is it?"},
        client=Client(client=fake, max_workers=1),
    )
    assert out == [{"hook": pytest.approx(2.6), "clarity": pytest.approx(0.4)}] * 2
    assert len(fake.calls) == 2
    assert set(fake.calls[0][1]) == {"hook", "clarity"}


def test_level_limits() -> None:
    jev = Client(client=FakeJev(scorer))
    with pytest.raises(HunchError, match="2–10"):
        score("x", ["only"], client=jev)
    with pytest.raises(HunchError, match="2–10"):
        score("x", [str(i) for i in range(11)], client=jev)
    with pytest.raises(HunchError, match="sequence"):
        score("x", "weak strong", client=jev)


def test_check_returns_bool_with_threshold_and_criteria() -> None:
    def handler(state, questions):
        q = questions["q"]
        assert q.type == "noul"
        assert q.instructions == "is spam"
        assert q.criteria == {"true": "unsolicited ads", "false": None}
        return response(q=noul_answer(0.62))

    fake = FakeJev(handler)
    jev = Client(client=fake)
    assert check("BUY NOW", "is spam", criteria={"true": "unsolicited ads"}, client=jev) is True
    assert check("BUY NOW", "is spam", criteria={"true": "unsolicited ads"}, threshold=0.7, client=jev) is False
    feeling = check("BUY NOW", "is spam", criteria={"true": "unsolicited ads"}, detail=True, client=jev)
    assert isinstance(feeling, Feeling) and feeling.p == pytest.approx(0.62)
    assert len(fake.calls) == 1


def test_check_mapping_form() -> None:
    fake = FakeJev(lambda s, q: response(spam=noul_answer(0.9), urgent=noul_answer(0.1)))
    out = check("BUY NOW", {"spam": "is spam", "urgent": "needs reply today"}, client=Client(client=fake))
    assert out == {"spam": True, "urgent": False}
    assert len(fake.calls) == 1
