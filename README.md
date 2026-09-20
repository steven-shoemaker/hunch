# hunch

Plain functions on Jev. Lists in, lists out.

[Jev](https://docs.typesafe.ai) is TypeSafe's System One model: you send state and typed questions, and you get labels, scores, and yes/no probabilities back. **hunch** turns that into six verbs you call like any other function. Code owns the workflow. Jev judges. An optional LLM may *propose* candidates; it never decides.

```python
import hunch

hunch.classify("This product is amazing!", ["positive", "negative", "neutral"])
# 'positive'

hunch.classify(df["JOB_TITLE"], ["Sales", "Engineering", "Marketing"])
# Series of labels, same index

hunch.score("Critical system failure", ["cosmetic", "degraded, workaround exists", "down for everyone"])
# 1.87  (position on the scale, 0 .. n-1)

hunch.check("BUY NOW!!!", "is unsolicited advertising")
# True
```

## Install

```sh
pip install hunch-jev
```

Python 3.10+. Set `TYPESAFE_API_KEY`, or call `hunch.configure(api_key=...)`. Never put keys in source.

## Verbs

| Verb | Jev primitive | Returns |
| --- | --- | --- |
| `classify(data, labels, multi_label=False, instructions=None)` | Choice, or one Noul per label | label, Enum member, or list of labels |
| `score(data, levels, instructions=None)` | Score | float position on the scale, or `{dim: float}` |
| `check(data, statement, criteria=None, threshold=0.5)` | Noul | bool, or `{name: bool}` |
| `pick(candidates, instructions)` | Choice over the candidates | the winning candidate |
| `rank(candidates, dimensions, levels, weights=None)` | Score per dimension | `Ranked` rows, best first |
| `generate(target, n=1, instructions=None)` | your LLM, validated by pydantic | `target` or `list[target]` |

Every verb takes a single item or a list / tuple / pandas Series and returns the same shape. Repeated values are asked once. Items run in parallel across `max_workers` threads. Every verb accepts `context=` (extra state alongside the input) and `client=`. Every verb has an `_async` twin.

`labels` may be a list, an `Enum` class (you get members back), or a mapping of label to description. `instructions` on `score` and `check` may be a mapping of name to question; the dimensions go in one request and you get a dict per item.

### `detail=True`

The bare return is the answer. `detail=True` returns the whole distribution:

| Verb | Detail type | Fields |
| --- | --- | --- |
| `classify` | `Answer` | `.label .p .probabilities .confidence .shape .top2 .on()` |
| `classify(multi_label=True)` | `MultiAnswer` | `.labels .probabilities .threshold` |
| `score` | `Rating` | `.score .level .normalized .probabilities .legend .confidence .shape .on()` |
| `check` | `Feeling` | `.p .threshold`, truthy at threshold |
| `pick` | `Pick` | `.winner .ranked .confidence .shape .on()` |

`.shape` is **your** policy on the distribution, not a Jev field:

| Shape | Meaning |
| --- | --- |
| `sure` | One option dominates |
| `split` | Two options are close |
| `unsure` | Flat or weak evidence |

```python
level = hunch.classify(title, ["IC", "Manager", "Director"], detail=True)
seniority = level.on(
    sure=level.label,
    split=lambda: hunch.classify(title, level.top2),  # rematch the top two
    unsure="review",
)
```

Cutoffs live on `ShapePolicy`. Confidence is how peaked the distribution is, not whether the label is true. Changing the policy never re-runs inference.

## Generate, rank, pick

An LLM drafts. Jev scores and chooses. Code keeps the weights.

```python
import hunch

hunch.configure(llm=hunch.openrouter(), cache="~/.cache/hunch")

tweets = hunch.generate(str, n=20, instructions="Tweets introducing hunch to Python developers", context=README)

ranked = hunch.rank(
    tweets,
    {"hook": "How strong is the first line?", "clarity": "How clearly does it say what hunch does?", "specific": "How concrete, not generic, is it?"},
    levels=["weak", "okay", "strong", "excellent"],
    weights={"hook": 2, "clarity": 1, "specific": 1},
)
finalists = [row.item for row in ranked[:5]]

winner = hunch.pick(finalists, "the tweet most likely to make a Python developer install hunch")
```

`generate` accepts `str`, `int`, dataclasses, `TypedDict`s, pydantic models, `list[str]`, and any other type pydantic can validate. `hunch.openai`, `hunch.cerebras`, and `hunch.openrouter` are OpenAI-compatible adapters; pass `llm=` on `configure()` or on `generate()`.

## Client

```python
jev = hunch.Client(api_key=..., model="jev-latest", cache="~/.cache/hunch", max_workers=8, policy=ShapePolicy(...))
hunch.classify(x, labels, client=jev)
jev.usage   # calls, cache hits, tokens, model
```

`hunch.configure(...)` takes the same arguments and sets the default used when `client=` is omitted. `cache=` persists raw Jev answers on disk keyed by state and question.

## Examples

Each one is a single file with the data inline. The first three need only `TYPESAFE_API_KEY`.

| File | Shows |
| --- | --- |
| [`find_angry_reviews.py`](examples/find_angry_reviews.py) | `check` over a column as a boolean mask, ranking by probability, several checks in one request |
| [`classify_job_titles.py`](examples/classify_job_titles.py) | `classify` with Enums, `detail=True`, and `.on()` routing sure / split / unsure with a rematch |
| [`triage_tickets.py`](examples/triage_tickets.py) | `score` on two scales in one request, multi-label `classify`, paging policy kept in code |
| [`introduce_hunch.py`](examples/introduce_hunch.py) | `generate` 20 tweets with an LLM, `rank` them on weighted dimensions, `pick` the winner |
| [`organize_downloads.py`](examples/organize_downloads.py) | An LLM proposes a folder taxonomy, `classify` assigns every file, the script moves them. `--dry-run` prints the plan |

## What this is not

Jev does not invent labels. `labels` is the whole set of allowed answers. `generate` invents candidates; it has no tools and takes no actions. Open-ended writing and multi-step agents are out of scope.

## License

MIT. Jev and TypeSafe are [typesafe.ai](https://typesafe.ai); this library is not affiliated.
