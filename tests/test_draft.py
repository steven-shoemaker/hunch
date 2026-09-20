from __future__ import annotations

import pytest

from hunch import HunchError, NoSessionError, ask, connect, draft, role
from tests.fakes import FakeJev, FakeLLM, choice_answer, noul_answer, response


def test_draft_requires_a_session() -> None:
    writer = role("Name a folder.", emit=str)
    with pytest.raises(NoSessionError):
        draft("invoice.pdf", writer)


def test_draft_needs_a_model() -> None:
    writer = role("Name a folder.", emit=str)
    jev = connect(client=FakeJev(lambda state, questions: response()))
    with jev.session():
        with pytest.raises(HunchError, match="language model"):
            draft("invoice.pdf", writer)


def test_draft_text_then_ask_among_proposal() -> None:
    llm = FakeLLM("q3-invoices")
    fake = FakeJev(
        lambda state, questions: response(
            q1=choice_answer(
                "q3-invoices",
                {"q3-invoices": 0.9, "review": 0.07, "junk": 0.03},
                0.85,
            )
        )
    )
    jev = connect(client=fake, llm=llm)
    writer = role("Name one kebab-case folder.", emit=str)
    with jev.session():
        proposed = draft("Q3 invoice final.pdf", writer)
        folder = ask(
            {"name": "Q3 invoice final.pdf", "proposed": proposed.text},
            "which folder?",
            among=[proposed.text, "review", "junk"],
        )
        assert proposed.text == "q3-invoices"
        assert folder.top == "q3-invoices"
    assert len(llm.calls) == 1
    assert len(fake.calls) == 1


def test_draft_labels_and_cache() -> None:
    llm = FakeLLM('["haulage-docs", "screenshots", "review", "junk"]')
    jev = connect(client=FakeJev(lambda state, questions: response()), llm=llm)
    taxonomist = role("Propose folders.", emit=list[str])
    listing = [{"name": "a.pdf"}, {"name": "b.png"}]
    with jev.session():
        first = draft(listing, taxonomist)
        second = draft(listing, taxonomist)
    assert first.labels == ["haulage-docs", "screenshots", "review", "junk"]
    assert second.labels == first.labels
    assert len(llm.calls) == 1


def test_feels_on_draft() -> None:
    llm = FakeLLM("Please find attached herein the aforementioned.")
    fake = FakeJev(lambda state, questions: response(q1=noul_answer(0.93)))
    jev = connect(client=fake, llm=llm)
    writer = role("Rewrite the email.", emit=str)
    with jev.session():
        proposed = draft("hi", writer)
        assert proposed.feels("stiff or corporate")
    assert fake.calls[0][0] == {
        "draft": "Please find attached herein the aforementioned.",
        "labels": ["Please find attached herein the aforementioned."],
    }


def test_role_loop_limit() -> None:
    llm = FakeLLM(["one", "two", "three"])
    jev = connect(client=FakeJev(lambda state, questions: response()), llm=llm)
    writer = role("Rewrite.", emit=str, max_loops=2)
    with jev.session():
        draft("a", writer)
        draft("b", writer)
        with pytest.raises(HunchError, match="loop limit"):
            draft("c", writer)


def test_via_on_role_overrides_connect() -> None:
    ignored = FakeLLM("nope")
    used = FakeLLM("keep")
    jev = connect(client=FakeJev(lambda state, questions: response()), llm=ignored)
    writer = role("Name it.", via=used, emit=str)
    with jev.session():
        assert draft("x", writer).text == "keep"
    assert used.calls
    assert not ignored.calls
