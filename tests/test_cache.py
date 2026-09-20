from __future__ import annotations

from hunch import Client, ShapePolicy, classify
from tests.fakes import FakeJev, choice_answer, response


def handler(state, questions):
    return response(q=choice_answer("Sales", {"Sales": 0.6, "Other": 0.4}, 0.5))


def test_disk_cache_survives_a_new_client(tmp_path) -> None:
    fake = FakeJev(handler)
    first = Client(client=fake, cache=tmp_path)
    assert classify("VP", ["Sales", "Other"], client=first) == "Sales"
    assert len(fake.calls) == 1

    later = Client(client=fake, cache=tmp_path)
    assert classify("VP", ["Sales", "Other"], client=later) == "Sales"
    assert len(fake.calls) == 1
    assert later.usage.hits == 1 and later.usage.calls == 0
    assert first.usage.calls == 1 and first.usage.model == "jev-test"
    assert first.usage.input_tokens == 10


def test_policy_changes_do_not_refetch(tmp_path) -> None:
    fake = FakeJev(handler)
    loose = Client(client=fake, cache=tmp_path)
    assert classify("VP", ["Sales", "Other"], detail=True, client=loose).shape == "split"
    strict = Client(client=fake, cache=tmp_path, policy=ShapePolicy(unsure_confidence=0.6))
    assert classify("VP", ["Sales", "Other"], detail=True, client=strict).shape == "unsure"
    assert len(fake.calls) == 1  # shape is a code policy over cached raw answers


def test_clear_cache(tmp_path) -> None:
    fake = FakeJev(handler)
    jev = Client(client=fake, cache=tmp_path)
    classify("VP", ["Sales", "Other"], client=jev)
    jev.clear_cache()
    classify("VP", ["Sales", "Other"], client=jev)
    assert len(fake.calls) == 2
