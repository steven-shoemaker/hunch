from __future__ import annotations

import pytest

from hunch import Check, Classify, Client, HunchError, Rate, ask, check, classify, pick, rank, score
from tests.fakes import FakeJev, choice_answer, noul_answer, response, score_answer

pd = pytest.importorskip("pandas")


def handler(state, questions):
    out = {}
    for qid, q in questions.items():
        if q.type == "choice":
            out[qid] = choice_answer(list(q.criteria)[0], {k: (0.9 if i == 0 else 0.1) for i, k in enumerate(q.criteria)}, 0.8)
        elif q.type == "noul":
            out[qid] = noul_answer(0.9)
        else:
            out[qid] = score_answer(1.0, {1: 1.0}, 0.9, {0: "lo", 1: "hi"})
    return response(**out)


def frame():
    return pd.DataFrame({"merchant": ["Acme"], "memo": ["lunch"], "secret": ["ssn"]}, index=[7])


def test_columns_limits_what_every_verb_sends() -> None:
    fake = FakeJev(handler)
    jev = Client(client=fake, max_workers=1)
    classify(frame(), ["meals", "travel"], columns=["merchant", "memo"], client=jev)
    score(frame(), ["lo", "hi"], columns=["memo"], client=jev)
    check(frame(), "is food", columns=["memo"], client=jev)
    ask(frame(), {"x": Check("is food")}, columns=["merchant"], client=jev)
    rank(frame(), "good?", ["lo", "hi"], columns=["memo"], client=jev)
    assert all("secret" not in s["input"] for s, _ in fake.calls)
    assert pick(frame(), "best", columns=["memo"], client=jev) == 7
    with pytest.raises(HunchError, match="doesn't have"):
        classify(frame(), ["a"], columns=["nope"], client=jev)
    with pytest.raises(HunchError, match="only applies to a DataFrame"):
        classify(["x"], ["a"], columns=["memo"], client=jev)


def test_question_context_stays_on_its_question() -> None:
    fake = FakeJev(handler)
    jev = Client(client=fake, max_workers=1)
    out = ask(["bio"], {
        "vibe": Classify(["outdoorsy", "homebody"]),
        "flag": Check("has a red flag"),
        "fit": Rate(["lo", "hi"], "fit with the person in context?", context={"looking_for": "runner"}),
    }, context={"city": "Denver"}, client=jev)
    assert out[0]["vibe"] == "outdoorsy" and out[0]["fit"] == 1.0
    assert len(fake.calls) == 2  # vibe + flag share one request; fit gets its own
    by_questions = {tuple(sorted(q)): s for s, q in fake.calls}
    assert by_questions[("flag", "vibe")] == {"input": "bio", "city": "Denver"}
    assert by_questions[("fit",)] == {"input": "bio", "city": "Denver", "looking_for": "runner"}
