from __future__ import annotations

from hunch import ask, connect
from tests.fakes import FakeJev, choice_answer, response


def test_disk_cache_survives_a_new_client(tmp_path) -> None:
    fake = FakeJev(
        lambda state, questions: response(
            q1=choice_answer("Sales", {"Sales": 0.95, "Other": 0.05}, 0.9)
        )
    )
    first = connect(client=fake, cache=tmp_path)
    with first.session():
        assert ask("VP", "function?", among=["Sales", "Other"]).top == "Sales"
    assert len(fake.calls) == 1

    later = connect(client=fake, cache=tmp_path)
    with later.session():
        assert ask("VP", "function?", among=["Sales", "Other"]).top == "Sales"
    assert len(fake.calls) == 1
    assert later.usage.hits == 1
    assert later.usage.calls == 0
