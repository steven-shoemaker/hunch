from __future__ import annotations

import pandas as pd

from hunch import ask, connect, over
from tests.fakes import FakeJev, choice_answer, noul_answer, response


def test_over_classifies_distinct_titles_and_joins() -> None:
    def handler(state, questions):
        title = state["JOB_TITLE"]
        answers = {}
        for qid, question in questions.items():
            if question.type == "noul":
                answers[qid] = noul_answer(0.1)
            elif "Sales" in question.criteria and "Engineering" in question.criteria:
                pick = "Sales" if "Sales" in title else "Engineering"
                answers[qid] = choice_answer(pick, {"Sales": 0.9, "Engineering": 0.1}, 0.8)
            else:
                answers[qid] = choice_answer(
                    "Manager",
                    {"Manager": 0.85, "Director": 0.15},
                    0.7,
                )
        return response(**answers)

    fake = FakeJev(handler)
    jev = connect(client=fake)
    prospects = pd.DataFrame(
        {
            "JOB_TITLE": ["VP Sales", "Staff Engineer", "VP Sales"],
            "id": [1, 2, 3],
        }
    )

    @over(prospects, "JOB_TITLE")
    def classify(title):
        function = ask(title, "function?", among=["Sales", "Engineering"])
        level = ask(title, "level?", among=["Manager", "Director"])
        seniority = level.on(sure=level.top, torn=level.top, lost="review")
        return {
            "function": function.top,
            "function_shape": function.shape,
            "seniority": seniority,
            "seniority_shape": level.shape,
        }

    results = classify.run(jev)
    assert list(results.function) == ["Sales", "Engineering", "Sales"]
    assert list(results.seniority) == ["Manager", "Manager", "Manager"]
    assert len(fake.calls) == 2
    assert all(len(questions) == 2 for _, questions in fake.calls)


def test_cache_skips_repeat_questions() -> None:
    fake = FakeJev(
        lambda state, questions: response(
            **{
                qid: choice_answer("Sales", {"Sales": 0.95, "Other": 0.05}, 0.9)
                for qid in questions
            }
        )
    )
    jev = connect(client=fake)
    frame = pd.DataFrame({"JOB_TITLE": ["VP Sales"]})

    @over(frame, "JOB_TITLE")
    def classify(title):
        return {"function": ask(title, "function?", among=["Sales", "Other"]).top}

    first = classify.run(jev)
    second = classify.run(jev)
    assert first.function.iloc[0] == "Sales"
    assert second.function.iloc[0] == "Sales"
    assert len(fake.calls) == 1
