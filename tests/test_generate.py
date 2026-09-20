from __future__ import annotations

from dataclasses import dataclass

import pytest

from hunch import Client, HunchError, generate
from tests.fakes import FakeJev, FakeLLM, response


def jev_with(llm):
    return Client(client=FakeJev(lambda s, q: response()), llm=llm)


def test_generate_n_strings() -> None:
    llm = FakeLLM('["one", "two", "three"]')
    out = generate(str, n=3, instructions="tweets", client=jev_with(llm))
    assert out == ["one", "two", "three"]
    system, user = llm.calls[0]
    assert "exactly 3" in system
    assert '"instructions": "tweets"' in user


def test_generate_dataclass() -> None:
    @dataclass
    class Recipe:
        title: str
        steps: list[str]

    llm = FakeLLM('```json\n{"title": "Toast", "steps": ["bread", "heat"]}\n```')
    out = generate(Recipe, client=jev_with(llm))
    assert out == Recipe("Toast", ["bread", "heat"])


def test_plain_text_is_a_valid_str() -> None:
    out = generate(str, instructions="a name", llm=FakeLLM("Thaelar Moonweaver"), client=jev_with(None))
    assert out == "Thaelar Moonweaver"


def test_retries_once_on_invalid_then_fails_loud() -> None:
    llm = FakeLLM(["not json", '["a", "b"]'])
    assert generate(str, n=2, client=jev_with(llm)) == ["a", "b"]
    assert "previous reply was invalid" in llm.calls[1][1]

    short = FakeLLM(['["a"]', '["a"]'])
    with pytest.raises(HunchError, match="expected 2 items"):
        generate(str, n=2, client=jev_with(short))


def test_needs_a_language_model() -> None:
    with pytest.raises(HunchError, match="language model"):
        generate(str, client=jev_with(None))
    with pytest.raises(HunchError, match="n >= 1"):
        generate(str, n=0, client=jev_with(FakeLLM("x")))
