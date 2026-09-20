from __future__ import annotations

import pytest

from hunch import NoSessionError, ask, connect
from hunch.answer import Answer
from tests.fakes import FakeJev, choice_answer, noul_answer, response


def test_ask_requires_a_session() -> None:
    with pytest.raises(NoSessionError):
        ask("VP Sales", "function?", among=["Sales", "Engineering"])


def test_independent_asks_share_one_request() -> None:
    fake = FakeJev(
        lambda state, questions: response(
            q1=choice_answer("Sales", {"Sales": 0.91, "Engineering": 0.09}, 0.82),
            q2=choice_answer(
                "Manager",
                {"Manager": 0.88, "Director": 0.08, "Individual Contributor": 0.04},
                0.80,
            ),
        )
    )
    jev = connect(client=fake)
    with jev.session():
        function = ask("VP Sales", "function?", among=["Sales", "Engineering"])
        level = ask(
            "VP Sales",
            "level?",
            among=["Individual Contributor", "Manager", "Director"],
        )
        assert function.top == "Sales"
        assert level.top == "Manager"
        assert function.shape == "sure"
        assert level.shape == "sure"
    assert len(fake.calls) == 1
    state, questions = fake.calls[0]
    assert state == {"value": "VP Sales"}
    assert len(questions) == 2


def test_torn_rematch_is_a_second_request() -> None:
    def handler(state, questions):
        question = next(iter(questions.values()))
        qid = next(iter(questions))
        if len(question.criteria) == 2:
            return response(
                **{qid: choice_answer("Manager", {"Manager": 0.62, "Director": 0.38}, 0.24)}
            )
        return response(
            **{
                qid: choice_answer(
                    "Manager",
                    {"Manager": 0.52, "Director": 0.45, "Individual Contributor": 0.03},
                    0.50,
                )
            }
        )

    fake = FakeJev(handler)
    jev = connect(client=fake)
    with jev.session():
        level = ask(
            "Head of Sales",
            "level?",
            among=["Individual Contributor", "Manager", "Director"],
        )
        seniority = level.on(
            sure=level.top,
            torn=lambda: ask(
                "Head of Sales",
                "which of these two fits better?",
                among=level.top2,
            ).top,
            lost="review",
        )
        assert level.shape == "torn"
        assert seniority == "Manager"
    assert len(fake.calls) == 2


def test_feels_uses_noul() -> None:
    fake = FakeJev(lambda state, questions: response(q1=noul_answer(0.81)))
    jev = connect(client=fake)
    from hunch.subject import Subject

    with jev.session():
        title = Subject("Account Manager", field="JOB_TITLE")
        feeling = title.feels("a rank word governing an account rather than people")
        assert feeling
        assert feeling.p == pytest.approx(0.81)
    assert fake.calls[0][0] == {"JOB_TITLE": "Account Manager"}


def test_on_branches() -> None:
    sure = Answer("Sales", {"Sales": 0.9, "Other": 0.1}, 0.8, "sure")
    torn = Answer("Sales", {"Sales": 0.51, "Other": 0.49}, 0.02, "torn")
    lost = Answer("Other", {"Sales": 0.4, "Other": 0.6}, 0.1, "lost")
    assert sure.on(sure="a", torn="b", lost="c") == "a"
    assert torn.on(sure="a", torn=lambda: "b", lost="c") == "b"
    assert lost.on(sure="a", torn="b", lost="c") == "c"
