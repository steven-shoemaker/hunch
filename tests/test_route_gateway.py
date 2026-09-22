from __future__ import annotations

import pytest

from hunch import Check, Classify, Client, HunchError, Rate, ask, classify, route
from hunch import gateway
from tests.fakes import FakeJev, choice_answer, noul_answer, response, score_answer

pd = pytest.importorskip("pandas")

LEVELS = ["low", "medium", "high"]


def tickets_jev(state, questions):
    text = state["input"]
    urgent = 0.95 if "down" in text else 0.1
    topic = "billing" if "charged" in text else ("bug" if "error" in text or "down" in text else "other")
    topic_p = 0.9 if topic != "other" else 0.4
    probs = {t: (topic_p if t == topic else (1 - topic_p) / 2) for t in ["billing", "bug", "other"]}
    return response(
        urgent=noul_answer(urgent),
        topic=choice_answer(topic, probs, 0.8),
        severity=score_answer(2.0 if urgent > 0.5 else 0.5, {2: 0.9, 1: 0.1} if urgent > 0.5 else {0: 0.5, 1: 0.5}, 0.8, dict(enumerate(LEVELS))),
    )


TICKETS = ["Checkout is down for everyone", "I was charged twice", "How do I export?", "Weird error on login"]
QUESTIONS = {
    "urgent": Check("needs a human within the hour"),
    "topic": Classify(["billing", "bug", "other"]),
    "severity": Rate(LEVELS),
}
RULES = {
    "page": {"urgent": 0.8},
    "billing": {"topic": ("billing", 0.7)},
    "review": {"topic.shape": "unsure"},
    "engineering": {"topic": ["bug"], "severity": 0.4},
}


def test_route_first_matching_rule_wins_on_detail_answers() -> None:
    jev = Client(client=FakeJev(tickets_jev), max_workers=1)
    answers = ask(TICKETS, QUESTIONS, detail=True, client=jev)
    assert route(answers, RULES, default="triage") == ["page", "billing", "review", "engineering"]
    assert route(answers[2], RULES, default="triage") == "review"


def test_route_reads_joined_dataframes() -> None:
    jev = Client(client=FakeJev(tickets_jev), max_workers=1)
    df = pd.DataFrame({"text": TICKETS}, index=[7, 8, 9, 10])
    joined = df.join(df["text"].hunch.ask(QUESTIONS, detail=True, client=jev))
    out = joined.hunch.route(RULES, default="triage")
    assert out.name == "route" and list(out.index) == [7, 8, 9, 10]
    assert out.tolist() == ["page", "billing", "review", "engineering"]


def test_route_on_bare_answers_labels_and_bools() -> None:
    assert route(["billing", "bug", "other"], {"money": {"_": "billing"}, "eng": {"_": ["bug"]}}, default="x") == ["money", "eng", "x"]
    assert route({"spam": True, "tier": "enterprise"}, {"block": {"spam": True, "tier": "free"}, "vip": {"tier": "enterprise"}}) == "vip"
    assert route({"spam": None}, {"block": {"spam": True}}, default="keep") == "keep"  # maybe/skipped never matches


def test_route_functions_and_validation() -> None:
    jev = Client(client=FakeJev(tickets_jev), max_workers=1)
    answers = ask(TICKETS[:1], QUESTIONS, detail=True, client=jev)
    assert route(answers, {"hot": {"severity": lambda r: r.level == "high"}}) == ["hot"]
    with pytest.raises(HunchError, match="tuple"):
        route(answers, {"x": {"topic": ("billing", "high")}})
    with pytest.raises(HunchError, match="at least one rule"):
        route(answers, {})


def test_route_works_with_classify_detail_and_enum_labels() -> None:
    from enum import Enum

    class Topic(Enum):
        BILLING = "billing"
        BUG = "bug"
        OTHER = "other"

    jev = Client(client=FakeJev(lambda s, q: response(q=choice_answer("billing", {"billing": 0.9, "bug": 0.05, "other": 0.05}, 0.9))))
    answer = classify("charged twice", Topic, detail=True, client=jev)
    assert route({"topic": answer}, {"money": {"topic": ("billing", 0.8)}}) == "money"


# ----------------------------------------------------------------------------- gateways


def test_openrouter_sends_the_systemone_body_to_the_decisions_endpoint(monkeypatch) -> None:
    sent = {}

    def fake_post(url, headers, body, timeout, retries=3):
        sent.update(url=url, headers=headers, body=body)
        answers = {qid: {"type": "noul", "noul": 0.92} for qid in body["questions"]}
        return {"model": "~typesafe/jev-latest", "answers": answers, "usage": {"input_tokens": 312, "output_tokens": 48}}, {}

    monkeypatch.setattr(gateway, "_post", fake_post)
    jev = Client(gateway="openrouter", api_key="test-key")
    assert jev.jev.__class__.__name__ == "OpenRouterJev"
    from hunch import check

    assert check("Help! payouts failing for 3 days", "conveys urgency", client=jev) is True
    assert sent["url"] == "https://openrouter.ai/api/alpha/decisions"
    assert sent["headers"]["Authorization"] == "Bearer test-key"
    assert sent["body"]["model"] == "~typesafe/jev-latest"
    assert sent["body"]["questions"]["check"]["type"] == "noul"
    assert jev.usage.input_tokens == 312


def test_vercel_translates_boolean_questions_legends_and_confidence(monkeypatch) -> None:
    sent = {}

    def fake_post(url, headers, body, timeout, retries=3):
        sent.update(url=url, headers=headers, body=body)
        return {
            "answers": {
                "urgent": {"type": "boolean", "probability": 0.91},
                "topic": {"type": "choice", "choice": "billing", "probabilities": {"billing": 0.8, "bug": 0.15, "other": 0.05}},
                "severity": {"type": "score", "score": 1.4, "probabilities": {"0": 0.1, "1": 0.6, "2": 0.3}},
            },
            "usage": {"inputTokens": 120, "outputTokens": 3},
            "providerMetadata": {"typesafe": {"confidence": {"topic": 0.72}}},
        }, {}

    monkeypatch.setattr(gateway, "_post", fake_post)
    jev = Client(gateway="vercel", api_key="test-key")
    out = ask("My card was charged twice", {
        "urgent": Check("needs a human within the hour"),
        "topic": Classify(["billing", "bug", "other"]),
        "severity": Rate(LEVELS),
    }, detail=True, client=jev)
    assert sent["url"] == "https://ai-gateway.vercel.sh/v4/ai/evaluation-model"
    assert sent["headers"]["Ai-Model-Id"] == "typesafe-ai/jev"
    assert sent["body"]["questions"]["urgent"]["type"] == "boolean"
    assert out["urgent"].p == pytest.approx(0.91)
    assert out["topic"].label == "billing" and out["topic"].confidence == pytest.approx(0.72)
    assert out["severity"].level == "medium"  # legend rebuilt from the levels we sent
    assert out["severity"].confidence == pytest.approx((3 * 0.6 - 1) / 2)  # Jev's formula when not reported
    assert jev.usage.output_tokens == 3


def test_gateway_needs_a_key(monkeypatch) -> None:
    monkeypatch.delenv("OPENROUTER_API_KEY", raising=False)
    monkeypatch.delenv("AI_GATEWAY_API_KEY", raising=False)
    with pytest.raises(HunchError, match="OPENROUTER_API_KEY"):
        Client(gateway="openrouter")
    with pytest.raises(HunchError, match="AI_GATEWAY_API_KEY"):
        Client(gateway="vercel")
    with pytest.raises(HunchError, match="gateway="):
        Client(gateway="nope")
