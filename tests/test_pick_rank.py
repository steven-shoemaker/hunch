from __future__ import annotations

import pytest

from hunch import Client, HunchError, Pick, pick, rank
from tests.fakes import FakeJev, choice_answer, response, score_answer


def best_is_longest(state, questions):
    q = questions["q"]
    assert q.type == "choice"
    ids = list(q.criteria)
    winner = max(ids, key=lambda cid: len(str(q.criteria[cid])))
    probs = {cid: (0.8 if cid == winner else 0.2 / max(1, len(ids) - 1)) for cid in ids}
    return response(q=choice_answer(winner, probs, 0.8))


def test_pick_compares_candidates_as_choice_options() -> None:
    fake = FakeJev(best_is_longest)
    tweets = ["short", "a much longer tweet here", "mid length"]
    winner = pick(tweets, "the best intro tweet", client=Client(client=fake))
    assert winner == "a much longer tweet here"
    state, questions = fake.calls[0]
    assert state == {"input": {"task": "the best intro tweet"}}
    assert questions["q"].criteria == {"c0": tweets[0], "c1": tweets[1], "c2": tweets[2]}
    rich = pick(tweets, "the best intro tweet", detail=True, client=Client(client=fake))
    assert isinstance(rich, Pick)
    assert rich.winner == tweets[1]
    assert rich.ranked[0] == (tweets[1], pytest.approx(0.8))
    assert rich.shape == "sure"


def test_pick_single_candidate_needs_no_call_and_validates() -> None:
    fake = FakeJev(best_is_longest)
    jev = Client(client=fake)
    assert pick(["only"], "best", client=jev) == "only"
    assert fake.calls == []
    with pytest.raises(HunchError, match="at least one"):
        pick([], "best", client=jev)
    with pytest.raises(HunchError, match="instructions"):
        pick(["a", "b"], "  ", client=jev)


def test_pick_tournament_over_255() -> None:
    fake = FakeJev(best_is_longest)
    field = [f"c{i:04d}" for i in range(300)]
    field[299] = "x" * 40  # longest overall
    winner = pick(field, "best", client=Client(client=fake))
    assert winner == "x" * 40
    assert len(fake.calls) == 3  # two heats and a final
    assert all(len(q["q"].criteria) <= 255 for _, q in fake.calls)


def test_rank_weights_normalized_dimension_scores() -> None:
    legend = {0: "weak", 1: "okay", 2: "strong"}

    def handler(state, questions):
        item = state["input"]
        answers = {}
        for qid in questions:
            value = {"a": {"hook": 2.0, "clarity": 0.0}, "b": {"hook": 1.0, "clarity": 2.0}}[item][qid]
            answers[qid] = score_answer(value, {int(value): 1.0}, 0.9, legend)
        return response(**answers)

    fake = FakeJev(handler)
    jev = Client(client=fake, max_workers=1)
    dims = {"hook": "hook?", "clarity": "clear?"}
    rows = rank(["a", "b"], dims, list(legend.values()), client=jev)
    assert [r.item for r in rows] == ["b", "a"]  # equal weights: b = 0.75, a = 0.5
    assert rows[0].composite == pytest.approx(0.75)
    weighted = rank(["a", "b"], dims, list(legend.values()), weights={"hook": 3, "clarity": 1}, client=jev)
    assert [r.item for r in weighted] == ["a", "b"]  # a = 0.75, b = 0.625
    assert len(fake.calls) == 2  # re-weighting is code, not inference
    item, composite = weighted[0]
    assert (item, composite) == ("a", pytest.approx(0.75))
    with pytest.raises(HunchError, match="weights"):
        rank(["a"], dims, list(legend.values()), weights={"hook": 0, "clarity": 0}, client=jev)
