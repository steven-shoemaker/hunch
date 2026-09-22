"""0.9: pick(none=), check(uncertain=), pairs, rank(query=), label trees, backoff, extract."""

from __future__ import annotations

import pytest

from hunch import Check, Client, HunchError, Tree, ask, check, classify, extract, pairs, pick, rank, score
from tests.fakes import FakeJev, choice_answer, noul_answer, response, score_answer

pd = pytest.importorskip("pandas")


# ----------------------------------------------------------------------------- pick(none=)


def picker(fits: float):
    def handler(state, questions):
        out = {}
        if "q" in questions:
            ids = list(questions["q"].criteria)
            out["q"] = choice_answer(ids[0], {c: (0.8 if c == ids[0] else 0.2 / (len(ids) - 1)) for c in ids}, 0.7)
        if "fits" in questions:
            assert "candidates" in state["input"]  # the fits question can see what it's judging
            out["fits"] = noul_answer(fits)
        return response(**out)

    return handler


def test_pick_none_returns_none_when_nothing_fits_in_the_same_request() -> None:
    bad = FakeJev(picker(0.2))
    assert pick(["meh", "worse"], "a great first date", none=True, client=Client(client=bad)) is None
    assert len(bad.calls) == 1 and set(bad.calls[0][1]) == {"q", "fits"}
    good = FakeJev(picker(0.9))
    rich = pick(["meh", "worse"], "a great first date", none=True, detail=True, client=Client(client=good))
    assert rich.winner == "meh" and rich.fits == pytest.approx(0.9)
    solo = FakeJev(picker(0.1))
    assert pick(["only"], "x", none=True, client=Client(client=solo)) is None  # even one candidate gets asked


# ----------------------------------------------------------------------------- check(uncertain=)


def test_uncertain_band_gives_maybe_as_none() -> None:
    ps = {"sure yes": 0.9, "borderline": 0.5, "sure no": 0.1}
    jev = Client(client=FakeJev(lambda s, q: response(**{k: noul_answer(ps[s["input"]]) for k in q})), max_workers=1)
    assert check(list(ps), "x", uncertain=(0.3, 0.7), client=jev) == [True, None, False]
    assert check(list(ps), "x", client=jev) == [True, True, False]  # default: forced at 0.5
    detail = check(list(ps), "x", uncertain=(0.3, 0.7), detail=True, client=jev)
    assert [f.verdict for f in detail] == ["yes", "maybe", "no"]
    rows = ask(list(ps), {"flag": Check("x", uncertain=(0.3, 0.7))}, client=jev)
    assert [r["flag"] for r in rows] == [True, None, False]
    with pytest.raises(HunchError, match="uncertain"):
        check("x", "y", uncertain=(0.7, 0.3), client=jev)


# ----------------------------------------------------------------------------- pairs + rank(query=)


def test_pairs_line_up_two_inputs_for_any_verb() -> None:
    crm = pd.DataFrame({"name": ["Acme Inc", "Globex"]}, index=[10, 11])
    vendors = ["ACME Incorporated", "Initech"]
    joined = pairs(crm, vendors)
    assert list(joined.index) == [10, 11] and joined.loc[10] == {"a": {"name": "Acme Inc"}, "b": "ACME Incorporated"}
    fake = FakeJev(lambda s, q: response(score=score_answer(2.0, {2: 1.0}, 1.0, {0: "d", 1: "r", 2: "s"})))
    out = score(joined, ["different", "related", "same company"], client=Client(client=fake, max_workers=1))
    assert list(out.index) == [10, 11]
    assert fake.calls[0][0]["input"]["a"] == {"name": "Acme Inc"}
    assert pairs([1], [2], names=("claim", "source")) == [{"claim": 1, "source": 2}]
    with pytest.raises(HunchError, match="line up"):
        pairs([1, 2], [1])


def test_rank_query_goes_into_every_request() -> None:
    fake = FakeJev(lambda s, q: response(**{k: score_answer(1.0, {1: 1.0}, 1.0, {0: "lo", 1: "hi"}) for k in q}))
    rank(["a", "b"], "relevant to the query?", ["lo", "hi"], query="cheap flights to Denver",
         client=Client(client=fake, max_workers=1))
    assert all(s["query"] == "cheap flights to Denver" for s, _ in fake.calls)


# ----------------------------------------------------------------------------- trees and backoff


TREE = Tree({
    "Electronics": {"Phones": "handsets", "Computers": {"Laptops": None, "Tablets": None}},
    "Home": {"Kitchen": None, "Garden": None},
})


def tree_jev(state, questions):
    q = questions["q"]
    want = {"Electronics": 0.7, "Home": 0.3, "Computers": 0.8, "Phones": 0.2, "Laptops": 0.9, "Tablets": 0.1,
            "Kitchen": 0.6, "Garden": 0.4}
    probs = {o: want[o] for o in q.criteria}
    return response(q=choice_answer(max(probs, key=probs.get), probs, 0.8))


def test_tree_beam_search_returns_the_best_path() -> None:
    fake = FakeJev(tree_jev)
    out = classify(["macbook", "macbook", "thinkpad"], TREE, beam=2, client=Client(client=fake, max_workers=1))
    assert out == ["Electronics > Computers > Laptops"] * 3
    levels = [tuple(q["q"].criteria) for _, q in fake.calls]
    assert ("Electronics", "Home") in levels and ("Laptops", "Tablets") in levels
    top = next(q["q"] for _, q in fake.calls if "Computers" in q["q"].criteria)
    assert top.criteria["Computers"] == {"includes": ["Laptops", "Tablets"]}
    detail = classify("macbook", TREE, detail=True, client=Client(client=FakeJev(tree_jev)))
    assert detail.label == "Electronics > Computers > Laptops"


def test_a_plain_dict_with_object_descriptions_is_not_a_tree() -> None:
    fake = FakeJev(lambda s, q: response(q=choice_answer("a", {"a": 0.9, "b": 0.1}, 0.9)))
    classify("x", {"a": {"what": "alpha", "not_for": "beta"}, "b": {"what": "beta"}}, client=Client(client=fake))
    assert fake.calls[0][1]["q"].criteria["a"] == {"what": "alpha", "not_for": "beta"}


def test_backoff_answers_with_the_parent_when_children_are_close() -> None:
    fake = FakeJev(lambda s, q: response(q=choice_answer("Laptops", {"Laptops": 0.46, "Tablets": 0.44, "Kitchen": 0.10}, 0.2)))
    parents = {"Laptops": "Computers", "Tablets": "Computers", "Kitchen": "Home"}
    jev = Client(client=fake)
    assert classify("device", ["Laptops", "Tablets", "Kitchen"], backoff=parents, client=jev) == "Computers"
    rich = classify("device", ["Laptops", "Tablets", "Kitchen"], backoff=parents, detail=True, client=jev)
    assert rich.by == "parent" and rich.probabilities["Computers"] == pytest.approx(0.9)
    assert classify("device", ["Laptops", "Tablets", "Kitchen"], client=jev) == "Laptops"  # off by default
    frame = classify(pd.Series(["device"]), ["Laptops", "Tablets", "Kitchen"], backoff={"Kitchen": "Home"}, detail=True, client=jev)
    assert frame["label_by"].tolist() == ["jev"]  # column present even when nothing backed off


# ----------------------------------------------------------------------------- extract


def extractor(state, questions):
    out = {}
    for name, q in questions.items():
        ids = {cid: c["value"] for cid, c in q.criteria.items() if cid != "none"}
        if name == "total":
            want = next(cid for cid, v in ids.items() if "1,240" in v)
        elif name == "email":
            want = "none"  # pretend the only email is someone else's
        else:
            want = next(iter(ids))
        probs = {cid: (0.9 if cid == want else 0.1 / max(1, len(q.criteria) - 1)) for cid in q.criteria}
        out[name] = choice_answer(want, probs, 0.85)
    return response(**out)


INVOICE = "Subtotal $1,100.00. Tax $140.00. Total due $1,240.00 by 2026-10-01. Questions: ap@vendor.com"


def test_extract_copies_values_from_the_text_in_one_request() -> None:
    fake = FakeJev(extractor)
    out = extract(INVOICE, {
        "total": ("money", "the amount due, not a subtotal or tax"),
        "due_date": "date",
        "email": "email",
        "po_number": r"PO-\d+",
    }, client=Client(client=fake))
    assert out == {"total": "$1,240.00", "due_date": "2026-10-01", "email": None, "po_number": None}
    assert len(fake.calls) == 1
    q = fake.calls[0][1]
    assert "po_number" not in q  # no candidates, no question
    assert set(q["total"].criteria) == {"c0", "c1", "c2", "none"}
    assert "Total due" in q["total"].criteria["c2"]["in_context"]
    assert q["total"].instructions["field"] == "the amount due, not a subtotal or tax"


def test_extract_on_frames_functions_and_errors() -> None:
    fake = FakeJev(extractor)
    df = pd.DataFrame({"body": [INVOICE, "nothing here", INVOICE]}, index=[5, 6, 7])
    out = extract(df, {"total": "money"}, detail=True, client=Client(client=fake, max_workers=1))
    assert list(out.index) == [5, 6, 7] and out.loc[5, "total"] == "$1,240.00" and pd.isna(out.loc[6, "total"])
    assert "total_p" in out.columns
    assert len(fake.calls) == 1  # duplicate text asked once, the row with no candidates costs nothing
    words = extract("red green blue", {"color": lambda t: t.split()}, client=Client(client=FakeJev(extractor)))
    assert words == {"color": "red"}
    with pytest.raises(HunchError, match="not a known finder"):
        extract("x", {"bad": "("}, client=Client(client=fake))
