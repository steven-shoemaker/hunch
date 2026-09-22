"""LLM + Jev: escalate, discover, refine, verify, and the provider adapters."""

from __future__ import annotations

import json
from types import SimpleNamespace

import pytest

from hunch import Answer, Client, HunchError, anthropic, azure, classify, discover, ollama, refine, verify
from hunch.llm import AnthropicModel, as_llm
from tests.fakes import FakeJev, FakeLLM, choice_answer, noul_answer, response

pd = pytest.importorskip("pandas")
LEVELS = ["IC", "Manager", "Director"]


def levels(state, questions):
    title = state["input"]
    if title == "Head of Growth":  # flat -> unsure
        return response(q=choice_answer("Manager", {"IC": 0.34, "Manager": 0.36, "Director": 0.30}, 0.04))
    return response(q=choice_answer("IC", {"IC": 0.95, "Manager": 0.03, "Director": 0.02}, 0.9))


# ----------------------------------------------------------------------------- escalate


def test_only_unsure_rows_reach_the_llm_and_are_marked() -> None:
    llm = FakeLLM('{"label": "Director"}')
    jev = Client(client=FakeJev(levels), max_workers=1)
    out = classify(["Engineer", "Head of Growth", "Head of Growth"], LEVELS, unsure=llm, client=jev)
    assert out == ["IC", "Director", "Director"]
    assert len(llm.calls) == 1  # one distinct unsure row, one LLM call
    sent = json.loads(llm.calls[0][1])
    assert sent["input"] == "Head of Growth" and list(sent["labels"]) == LEVELS
    assert "first_pass_probabilities" in sent

    detail = classify(pd.Series(["Engineer", "Head of Growth"]), LEVELS, unsure=llm, detail=True, client=jev)
    assert detail["label_by"].tolist() == ["jev", "llm"]
    assert len(llm.calls) == 1  # the decision was cached

    calm = classify(pd.Series(["Engineer"]), LEVELS, unsure=llm, detail=True, client=jev)
    assert calm["label_by"].tolist() == ["jev"]  # column exists even when nothing escalated
    plain = classify(pd.Series(["Engineer"]), LEVELS, detail=True, client=jev)
    assert "label_by" not in plain.columns  # no policy, no column


def test_llm_must_answer_from_the_labels() -> None:
    llm = FakeLLM(['{"label": "VP"}', '"vp of nothing"'])
    jev = Client(client=FakeJev(levels), max_workers=1)
    with pytest.warns(UserWarning, match="no valid label"):
        out = classify(["Head of Growth"], LEVELS, unsure=llm, client=jev)
    assert out == ["Manager"]  # kept Jev's answer
    assert "Choose exactly one of" in llm.calls[1][1]


def test_a_plain_function_is_an_llm() -> None:
    def my_model(system: str, user: str) -> str:
        return "director"  # case-insensitive match, plain text is fine

    out = classify("Head of Growth", LEVELS, unsure=my_model, client=Client(client=FakeJev(levels)))
    assert out == "Director"


# ----------------------------------------------------------------------------- discover


def test_discover_returns_labels_ready_for_classify() -> None:
    llm = FakeLLM(json.dumps([
        {"name": "shipping", "description": "late, lost, or damaged deliveries"},
        {"name": "billing", "description": "charges, refunds, invoices"},
        {"name": "Other", "description": "dropped: the library adds its own"},
    ]))
    jev = Client(client=FakeJev(levels), llm=llm)
    reviews = ["box arrived crushed", "charged twice", "box arrived crushed"] * 50
    cats = discover(reviews, 3, instructions="by the customer's complaint", client=jev)
    assert cats == {
        "shipping": "late, lost, or damaged deliveries",
        "billing": "charges, refunds, invoices",
        "other": "fits none of the other categories",
    }
    system, user = llm.calls[0]
    assert "by the customer's complaint" in user
    assert json.loads(user)["context"]["examples"] == ["box arrived crushed", "charged twice"]  # deduped sample


# ----------------------------------------------------------------------------- refine


def has_install(state, questions):
    text = state["input"]
    return response(**{qid: noul_answer(0.9 if ("pip install" in text or q.instructions != "shows the install command") else 0.1)
                       for qid, q in questions.items()})


def test_refine_rewrites_only_failing_drafts_until_checks_pass() -> None:
    llm = FakeLLM(["hunch: pip install hunch-jev"])
    jev = Client(client=FakeJev(has_install), llm=llm, max_workers=1)
    out = refine(["try hunch", "pip install hunch-jev now"], ["shows the install command", "is under 280 characters"], client=jev)
    assert out == ["hunch: pip install hunch-jev", "pip install hunch-jev now"]
    assert len(llm.calls) == 1  # the passing draft was never sent
    sent = json.loads(llm.calls[0][1])
    assert sent["failed_requirements"] == ["shows the install command"]


def test_refine_stops_after_rounds_and_reports_what_failed() -> None:
    llm = FakeLLM(["still no command", "nope", "nah"])
    jev = Client(client=FakeJev(has_install), llm=llm, max_workers=1)
    result = refine("try hunch", {"install": "shows the install command"}, rounds=2, detail=True, client=jev)
    assert result.passed is False and result.rounds == 2 and result.failed == ["install"]
    assert result.original == "try hunch"
    frame = refine(pd.Series(["pip install hunch-jev"], index=[5]), ["shows the install command"], detail=True, client=jev)
    assert frame.loc[5, "passed"] and frame.loc[5, "rounds"] == 0
    with pytest.raises(HunchError, match="language model"):
        refine("x", ["y"], client=Client(client=FakeJev(has_install)))


# ----------------------------------------------------------------------------- verify


def supported(state, questions):
    q = questions["check"]
    assert q.criteria["false"].startswith("the source contradicts")
    row = state["input"]
    claim, source = (row["claim"], row["source"]) if isinstance(row, dict) else (row, state["source"])
    return response(check=noul_answer(0.9 if claim in source else 0.1))


def test_verify_against_one_source_and_per_row_sources() -> None:
    jev = Client(client=FakeJev(supported), max_workers=1)
    diff = "- return x\n+ return x + 1"
    assert verify(["return x + 1", "deletes the database"], diff, client=jev) == [True, False]
    claims = pd.Series(["a", "b"], index=[3, 4])
    out = verify(claims, ["has a", "has c"], client=jev)
    assert out.name == "supported" and out.tolist() == [True, False] and list(out.index) == [3, 4]
    detail = verify(claims, ["has a", "has c"], detail=True, client=jev)
    assert list(detail.columns) == ["supported", "supported_p"]
    with pytest.raises(HunchError, match="2 claims and 1 sources"):
        verify(["a", "b"], ["only one"], client=jev)


# ----------------------------------------------------------------------------- providers


def test_adapters() -> None:
    assert as_llm(None) is None
    assert as_llm(lambda s, u: "x").complete(system="", user="") == "x"
    with pytest.raises(HunchError):
        as_llm(42)
    assert ollama("qwen3").base_url == "http://localhost:11434/v1"
    assert azure("https://acme.openai.azure.com/", deployment="gpt", api_key="k").base_url == "https://acme.openai.azure.com/openai/v1"


def test_anthropic_adapter_reads_text_and_refusals() -> None:
    class FakeMessages:
        def __init__(self, stop: str) -> None:
            self.stop, self.kwargs = stop, {}

        def create(self, **kwargs):
            self.kwargs = kwargs
            return SimpleNamespace(stop_reason=self.stop, content=[SimpleNamespace(type="thinking"), SimpleNamespace(type="text", text=" hi ")])

    ok = FakeMessages("end_turn")
    model = AnthropicModel(api_key=None, model="claude-opus-5", client=SimpleNamespace(messages=ok))
    assert model.complete(system="s", user="u") == "hi"
    assert ok.kwargs["system"] == "s" and ok.kwargs["messages"] == [{"role": "user", "content": "u"}]
    refused = AnthropicModel(api_key=None, model="claude-opus-5", client=SimpleNamespace(messages=FakeMessages("refusal")))
    with pytest.raises(HunchError, match="declined"):
        refused.complete(system="s", user="u")
    assert callable(anthropic)
