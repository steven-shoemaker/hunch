from __future__ import annotations

from enum import Enum

import pytest

from hunch import Answer, Client, HunchError, MultiAnswer, classify, configure
from tests.fakes import FakeJev, choice_answer, noul_answer, response


def sales_or_eng(state, questions):
    title = state["input"]
    pick = "Sales" if "Sales" in title else "Engineering"
    return response(q=choice_answer(pick, {"Sales": 0.9, "Engineering": 0.1} if pick == "Sales" else {"Sales": 0.1, "Engineering": 0.9}, 0.85))


def test_single_item_returns_bare_label_and_sends_input_state() -> None:
    fake = FakeJev(sales_or_eng)
    jev = Client(client=fake)
    assert classify("VP Sales", ["Sales", "Engineering"], client=jev) == "Sales"
    state, questions = fake.calls[0]
    assert state == {"input": "VP Sales"}
    assert list(questions["q"].criteria) == ["Sales", "Engineering"]


def test_list_in_list_out_and_distinct_values_run_once() -> None:
    fake = FakeJev(sales_or_eng)
    jev = Client(client=fake, max_workers=1)
    out = classify(["VP Sales", "Staff Engineer", "VP Sales"], ["Sales", "Engineering"], client=jev)
    assert out == ["Sales", "Engineering", "Sales"]
    assert len(fake.calls) == 2  # duplicate value is not re-asked


def test_enum_labels_return_members() -> None:
    class Function(Enum):
        SALES = "Sales"
        ENG = "Engineering"

    fake = FakeJev(sales_or_eng)
    result = classify("VP Sales", Function, client=Client(client=fake))
    assert result is Function.SALES
    assert list(fake.calls[0][1]["q"].criteria) == ["Sales", "Engineering"]


def test_mapping_labels_send_descriptions_as_criteria() -> None:
    fake = FakeJev(sales_or_eng)
    classify(
        "VP Sales",
        {"Sales": "sells things", "Engineering": "builds things"},
        instructions="what function?",
        client=Client(client=fake),
    )
    question = fake.calls[0][1]["q"]
    assert question.criteria == {"Sales": "sells things", "Engineering": "builds things"}
    assert question.instructions == "what function?"


def test_context_is_merged_into_state() -> None:
    fake = FakeJev(sales_or_eng)
    classify("VP Sales", ["Sales", "Engineering"], context={"company": "Acme"}, client=Client(client=fake))
    assert fake.calls[0][0] == {"input": "VP Sales", "company": "Acme"}
    with pytest.raises(HunchError, match="reserved"):
        classify("x", ["a", "b"], context={"input": 1}, client=Client(client=fake))


def test_detail_returns_answer_with_shape_and_branching() -> None:
    fake = FakeJev(sales_or_eng)
    answer = classify("VP Sales", ["Sales", "Engineering"], detail=True, client=Client(client=fake))
    assert isinstance(answer, Answer)
    assert answer.label == "Sales"
    assert answer.p == pytest.approx(0.9)
    assert answer.shape == "sure"
    assert answer.top2 == ["Sales", "Engineering"]
    assert answer.on(sure="s", split=lambda: "t", unsure="l") == "s"


def test_multi_label_is_one_noul_per_label_thresholded() -> None:
    def handler(state, questions):
        assert all(q.type == "noul" for q in questions.values())
        assert len(questions) == 3
        return response(l0=noul_answer(0.9), l1=noul_answer(0.2), l2=noul_answer(0.6))

    fake = FakeJev(handler)
    jev = Client(client=fake)
    genres = classify("romcom about AI", ["comedy", "action", "sci-fi"], multi_label=True, client=jev)
    assert genres == ["comedy", "sci-fi"]  # ordered by probability, above 0.5
    assert len(fake.calls) == 1
    strict = classify("romcom about AI", ["comedy", "action", "sci-fi"], multi_label=True, threshold=0.8, client=jev)
    assert strict == ["comedy"]
    assert len(fake.calls) == 1  # thresholding is code, no new inference
    rich = classify("romcom about AI", ["comedy", "action", "sci-fi"], multi_label=True, detail=True, client=jev)
    assert isinstance(rich, MultiAnswer)
    assert rich.probabilities["sci-fi"] == pytest.approx(0.6)
    assert "comedy" in rich


def test_series_in_series_out_keeps_index() -> None:
    pd = pytest.importorskip("pandas")
    fake = FakeJev(sales_or_eng)
    titles = pd.Series(["VP Sales", "Staff Engineer"], index=[10, 20])
    out = classify(titles, ["Sales", "Engineering"], client=Client(client=fake))
    assert list(out.index) == [10, 20]
    assert out.loc[20] == "Engineering"
    assert out.name == "label"


def test_label_validation() -> None:
    jev = Client(client=FakeJev(sales_or_eng))
    with pytest.raises(HunchError, match="unique"):
        classify("x", ["a", "a"], client=jev)
    with pytest.raises(HunchError, match="at least one"):
        classify("x", [], client=jev)
    with pytest.raises(HunchError, match="not a string"):
        classify("x", "ab", client=jev)
    with pytest.raises(HunchError, match="at most 255"):
        classify("x", [str(i) for i in range(256)], client=jev)


def test_configure_sets_the_default_client() -> None:
    fake = FakeJev(sales_or_eng)
    configure(client=fake)
    assert classify("VP Sales", ["Sales", "Engineering"]) == "Sales"
    assert len(fake.calls) == 1
