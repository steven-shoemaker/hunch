# hunch

![hunch — lists in, lists out](https://raw.githubusercontent.com/steven-shoemaker/hunch/main/docs/banner.png)

Plain functions on Jev. Lists in, lists out.

[Jev](https://docs.typesafe.ai) is TypeSafe's System One model. You send it some state and a typed question, and it sends back a label, a score, or a yes/no probability instead of a paragraph. I think it's the most useful thing to happen to "AI in a for loop" in a while. Calling it raw is fiddly, though: build a state object, build a question object, dig the answer out of the response. hunch is the version I wanted, where each of those is one function call and you can hand it a list or a pandas column instead of one thing at a time.

The rule the whole library follows: Jev decides, your code owns the workflow, and if an LLM is involved at all it only gets to propose candidates.

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

drafts = hunch.generate(str, n=20, instructions="tweets introducing hunch")   # an LLM writes
hunch.pick(drafts, "most likely to make a Python developer install it")      # Jev chooses
# 'Most of my "AI" code was a for loop around a prompt and a JSON parser. ...'
```

## Install

```sh
pip install hunch-jev
```

Python 3.10+. Set `TYPESAFE_API_KEY` in your environment, or call `hunch.configure(api_key=...)` at startup. Keys don't belong in source files.

## Verbs

| Verb | Jev primitive | Returns |
| --- | --- | --- |
| `classify(data, labels, multi_label=False, instructions=None)` | Choice, or one Noul per label | label, Enum member, or list of labels |
| `score(data, levels, instructions=None)` | Score | float position on the scale, or `{dim: float}` |
| `check(data, statement, criteria=None, threshold=0.5)` | Noul | bool, or `{name: bool}` |
| `pick(candidates, instructions)` | Choice over the candidates | the winning candidate |
| `rank(candidates, dimensions, levels, weights=None)` | Score per dimension | `Ranked` rows, best first |
| `generate(target, n=1, instructions=None)` | your LLM, validated by pydantic | `target` or `list[target]` |

Hand any verb one item and you get one answer back. Hand it a list, a tuple, or a pandas Series and you get the same container back, same length, same index. Duplicate values are only asked once, and the distinct ones run in parallel across `max_workers` threads. All of them take `context=` for extra state that should ride along with the input, and `client=` if you don't want the default. There's an `_async` twin of each, too.

`labels` can be a plain list, an `Enum` class (you get members back, not strings), or a dict of label to description when the names alone are ambiguous. On `score` and `check`, `instructions` can be a dict of name to question. Those go out as one request per item and you get a dict back per item, which is how you score five dimensions without five round trips.

### `detail=True`

The bare return is the answer. `detail=True` returns the whole distribution:

| Verb | Detail type | Fields |
| --- | --- | --- |
| `classify` | `Answer` | `.label .p .probabilities .confidence .shape .top2 .on()` |
| `classify(multi_label=True)` | `MultiAnswer` | `.labels .probabilities .threshold` |
| `score` | `Rating` | `.score .level .normalized .probabilities .legend .confidence .shape .on()` |
| `check` | `Feeling` | `.p .threshold`, truthy at threshold |
| `pick` | `Pick` | `.winner .ranked .confidence .shape .on()` |

`.shape` is a judgment about the distribution, and the cutoffs are yours, not Jev's:

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

Cutoffs live on `ShapePolicy`. One thing worth internalizing: confidence measures how peaked the distribution is, not whether the label is correct. A confidently wrong answer is still confident. Changing the policy never re-runs inference, because the cache stores the raw distribution and the shape is computed on the way out.

## Generate, rank, pick

This is the part where an LLM is allowed in the room. It writes the candidates. Jev scores them and picks. The weights stay in your code, so re-ranking after you change your mind costs nothing.

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

`hunch.configure(...)` takes the same arguments and sets the default used when `client=` is omitted. `cache=` writes raw Jev answers to disk keyed by state and question, so re-running a script over the same data is free.

## Examples

Each is a single file with the data inline, so you can run it as-is. The first three need only `TYPESAFE_API_KEY`. The last two also draft with an LLM, so they want an OpenRouter key.

| File | Shows |
| --- | --- |
| [`find_angry_reviews.py`](examples/find_angry_reviews.py) | `check` over a column as a boolean mask, ranking by probability, several checks in one request |
| [`classify_job_titles.py`](examples/classify_job_titles.py) | `classify` with Enums, `detail=True`, and `.on()` routing sure / split / unsure with a rematch |
| [`triage_tickets.py`](examples/triage_tickets.py) | `score` on two scales in one request, multi-label `classify`, paging policy kept in code |
| [`introduce_hunch.py`](examples/introduce_hunch.py) | `generate` 20 tweets with an LLM, `rank` them on weighted dimensions, `pick` the winner |
| [`organize_downloads.py`](examples/organize_downloads.py) | An LLM proposes a folder taxonomy, `classify` assigns every file, the script moves them. `--dry-run` prints the plan |

## What this is not

Jev does not invent labels. Whatever you pass as `labels` is the entire set of allowed answers, and that constraint is the point. `generate` is the one place invention happens, and it has no tools and takes no actions. If you want open-ended writing or a multi-step agent, this is the wrong library, on purpose.

## License

MIT. Jev and TypeSafe are [typesafe.ai](https://typesafe.ai); this library is not affiliated.
