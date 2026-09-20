# hunch

Jev as a column primitive.

[Jev](https://docs.typesafe.ai) is TypeSafe’s System One model: you send state and typed questions, and you get labels, scores, and yes/no probabilities back. **hunch** is a small Python layer on that API. Code owns the table and the side effects. Jev only judges. An optional LLM role may *propose* strings; it never closes.

```python
import hunch
from hunch import ask, over, rate

jev = hunch.connect()  # TYPESAFE_API_KEY

@over(prospects, "JOB_TITLE")
def classify(title):
    function = ask(
        title,
        "what business function does this title state?",
        among=functions,
        by=function_rules,
    )
    level = ask(
        title,
        "what level of rank does this title state?",
        among=levels,
        by=seniority_rules,
    )
    seniority = level.on(
        sure=level.top,
        torn=lambda: ask(
            title,
            "which of these two fits better?",
            among=level.top2,
            by=seniority_rules,
        ).top,
        lost=lambda: (
            "Individual Contributor"
            if title.feels("a rank word governing a product, program, or account rather than people")
            else "review"
        ),
    )
    keep = rate(
        title,
        "How important is this account?",
        ["disposable", "nice to have", "should keep", "must keep"],
    )
    return {
        "function": function.top,
        "function_shape": function.shape,
        "seniority": seniority,
        "seniority_shape": level.shape,
        "keep": keep.level,
    }

results = classify.run(jev)
```

Independent `ask` / `rate` / `feels` calls on the same value go in **one** Jev request. `@over` classifies each distinct value once, caches it, and left-joins onto the frame.

## Install

```sh
pip install "hunch @ git+https://github.com/steven-shoemaker/hunch.git"
# or from a clone:
pip install -e ".[dev]"
```

Requires Python 3.10+. Set `TYPESAFE_API_KEY`. Do not put keys in source.

```sh
pytest
```

## Verbs

| Call | Jev primitive | You get |
| --- | --- | --- |
| `ask(state, question, among=..., by=...)` | Choice | `.top`, `.top2`, `.p`, `.confidence`, `.shape` |
| `rate(state, question, levels)` | Score | `.score`, `.level`, `.shape` |
| `value.feels("...")` | Noul | truthy when P(yes) ≥ 0.5 |

`.shape` is **your** policy on the distribution, not a Jev field:

| Shape | Meaning |
| --- | --- |
| `sure` | One option dominates |
| `torn` | Two options are close |
| `lost` | Flat or weak evidence |

Cutoffs live on `ShapePolicy`. Confidence is how peaked the distribution is, not whether the label is true.

`connect(cache="~/.cache/hunch")` persists answers on disk. `jev.usage` reports calls, cache hits, tokens, and the model name.

## LLM roles

A role may only propose text or a list of labels. `ask` still decides.

```python
from hunch import ask, draft, openrouter, role

jev = hunch.connect(llm=openrouter())  # OPENROUTER_API_KEY
taxonomist = role(
    "Propose 8–16 kebab-case folder names. Include junk. No review pile.",
    emit=list[str],
)

with jev.session():
    taxonomy = draft(listing, taxonomist).labels

folder = ask(name, "which folder?", among=taxonomy)
```

`hunch.openai`, `hunch.cerebras`, and `hunch.openrouter` are OpenAI-compatible adapters. `via=` on a role overrides `llm=` on `connect()`. A role stops after `max_loops` (default 5) in one session.

## Example

[`examples/organize_downloads.py`](examples/organize_downloads.py) files a Downloads folder: one LLM taxonomy, then Jev assigns each loose file, then the script moves. Destination folders are skipped on later runs. `--dry-run` prints the plan.

## What this is not

Jev does not invent labels. `among=` is the whole set of allowed answers. Roles invent candidates; they have no tools and do not move files. Open-ended writing and multi-step agents are out of scope.

## License

MIT. Jev and TypeSafe are [typesafe.ai](https://typesafe.ai); this library is not affiliated.
