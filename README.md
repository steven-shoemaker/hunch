# hunch

![hunch — lists in, lists out](https://raw.githubusercontent.com/steven-shoemaker/hunch/main/docs/banner.png)

**Judgment as a Python function.** Hand hunch a column, ask it a question, get an answer for every row that your code can branch on.

```python
import hunch

themes = reviews.hunch.discover(6)                    # an LLM reads a sample and names the categories
reviews["theme"] = reviews.hunch.classify(themes)     # Jev files all 40,000 rows into them
angry = reviews.hunch.where("the customer is angry")  # a WHERE clause that understands English
```

That's three lines. Duplicates are asked once, requests run in parallel, and every answer comes back with its probabilities. Nobody writes a prompt or parses JSON.

`pip install hunch-jev`

**Using a coding agent?** Give it the hunch skill so it reaches for these verbs instead of writing prompt-and-parse code:

```bash
npx skills add steven-shoemaker/hunch --skill hunch        # Cursor, Codex, Claude Code, and other agents
claude plugin marketplace add steven-shoemaker/hunch && claude plugin install hunch@hunch   # Claude Code plugin
```

## Why this exists

[Jev](https://docs.typesafe.ai) is TypeSafe's System One model: a classifier with frontier-level intelligence that needs no fine-tuning. You give it some state and a typed question, and it answers with a label, a score on your rubric, or a yes/no probability. It doesn't write paragraphs. That makes it the right tool for the enormous number of jobs where you'd otherwise prompt an LLM and hope the answer parses.

Calling Jev directly means building state objects and question objects and digging answers out of responses, one row at a time, in your own loop. hunch is the version I wanted. Every judgment is one function call, and it works the same on a string, a list, or a DataFrame.

The rule hunch follows everywhere: **Jev decides, your code owns the workflow, and an LLM only ever proposes.** LLMs are great at writing and brainstorming, and unreliable as the final decision-maker. So they draft the tweets, name the categories, and take the rows Jev wasn't sure about. Jev makes the calls, and your code keeps the thresholds and weights.

## A tour

### Label anything

```python
hunch.classify("This product is amazing!", ["positive", "negative", "neutral"])
# 'positive'

df["function"] = df["title"].hunch.classify(["Sales", "Engineering", "Marketing", "Finance"])
df["severity"] = df["ticket"].hunch.score(["cosmetic", "degraded, workaround exists", "blocked", "outage"])
df["spam"]     = df["email"].hunch.check("is unsolicited advertising")
```

`classify` picks a label, `score` places each row on an ordered scale you describe, and `check` answers yes or no. Labels can be a list, an `Enum`, or a dict of label to description for when the names alone are ambiguous. A Series comes back as a Series on the same index.

### Filter by meaning

```python
df.hunch.where("might yell at a waiter for getting their order wrong")
df.hunch.where("is an economic buyer for a product like ours", columns=["title", "company"], threshold=0.7)
```

`where` keeps the rows a statement holds for, strongest match first, and returns them whole. `columns=` limits what Jev reads.

### Ask five things at once

```python
from hunch import Classify, Rate, Check

df = df.join(df.hunch.ask({
    "kind":     Classify(["bug", "feature request", "question"]),
    "severity": Rate(["cosmetic", "degraded", "blocked", "outage"]),
    "angry":    Check("the customer is frustrated"),
    "refund":   Check("asks for money back"),
}))
```

Every question about a row goes in one Jev request, and the answers come back as columns ready to `join`.

### Know when it's unsure

Every answer carries its distribution, and hunch sorts each one into a **shape**: `sure` when one label dominates, `split` when two are close, `unsure` when the evidence is flat. Tell it what to do with the shaky ones:

```python
seniority = df["title"].hunch.classify(["IC", "Manager", "Director", "Executive"],
                                       split="rematch", unsure="review")
```

`split="rematch"` re-asks between just the top two, and only for the rows that were split. `unsure="review"` flags the rest for a person.

### Hand the hard ones to an LLM

```python
seniority = df["title"].hunch.classify(LEVELS, unsure=hunch.anthropic())
```

Jev answers every row. The few it's unsure about go to the LLM, which has to choose from the same labels. So you pay for Jev on most rows and for a reasoning model only where it earns its cost. With `detail=True`, a `label_by` column shows which rows the LLM decided.

### Find the categories you didn't know you had

```python
themes = hunch.discover(tickets["body"], 8, instructions="by what the customer needs")
# {'Billing error': 'charged wrongly or twice...', 'Login trouble': '...', ..., 'other': '...'}
tickets["theme"] = tickets["body"].hunch.classify(themes)
```

An LLM reads a sample and proposes categories with descriptions. The result plugs straight into `classify`, and Jev files every row. An `other` bucket is included, so nothing gets forced into a category it doesn't fit.

### Draft, check, fix

```python
hunch.refine("hunch is a cool new library for AI stuff, check it out!!", {
    "accurate": "describes the library correctly according to the readme in context",
    "concrete": "names one specific thing the library does",
    "install":  "includes the command pip install hunch-jev",
    "calm":     "uses no exclamation marks or hype words",
}, context={"readme": README})
```

The LLM rewrites, and Jev checks every draft against your rules. Only the drafts that failed go back, each told which checks it missed, until everything passes or the rounds run out. One warning from testing this: checks only test what you ask. Without the `accurate` check and the README as context, the model happily wrote a fluent, calm, install-command-bearing tweet describing hunch as an embeddings library.

### Catch an agent making things up

```python
claims = ["skips None values", "adds a cache", "renames the function"]
hunch.verify(claims, diff)
# [True, False, False]
```

`verify` checks whether each claim is supported by a source: one document for all claims, or one per row. Use it as the last step of anything an LLM or agent produced, like review-bot comments, extracted fields, or summaries.

### Generate, rank, pick

```python
drafts = hunch.generate(str, n=20, instructions="tweets introducing hunch", context=README)
ranked = hunch.rank(drafts, {"hook": "How strong is the first line?", "clarity": "How clear is it?"},
                    levels=["weak", "okay", "strong", "excellent"], weights={"hook": 2, "clarity": 1})
winner = hunch.pick([row.item for row in ranked[:5]], "most likely to make a developer install it")
```

`generate` makes typed data: `str`, dataclasses, pydantic models, anything pydantic validates. `rank` scores every candidate on weighted dimensions, and `pick` puts the finalists head to head. Weights live in your code, so re-ranking costs nothing.

### Measure before you trust

```python
pred = sample["title"].hunch.classify(LEVELS, detail=True)
hunch.evaluate(pred, sample["true_level"])
# Evaluation(accuracy=91.0% on 100 rows, by shape: sure: 98% of 71, split: 79% of 19, unsure: 60% of 10)

cut = hunch.tune_threshold(sample.hunch.check("is a buyer", detail=True), sample["is_buyer"], precision=0.9)
```

Label 50 to 100 rows by hand. `evaluate` shows whether `sure` really means right on your data. `tune_threshold` picks the `check` / `where` cutoff that hits the precision or recall you need, instead of a number that felt right. The numbers in that comment are illustrative, since yours will depend on your data.

## Where it fits

Anywhere a person is reading rows and making a call.

**GTM: ICP fit.** Pass the ideal customer profile as context and let Jev read every account against it.

```python
ICP = "B2B SaaS, 200 to 2,000 employees, sells to mid-market, has a RevOps function, US or UK."
prospects = prospects.join(prospects.hunch.ask({
    "fit":   Rate(["not our buyer", "partial fit", "good fit", "textbook ICP"], "How well does this account match the ICP?"),
    "buyer": Check("this person could sign or sponsor a purchase for their team"),
}, context={"icp": ICP}))
outreach = prospects[prospects.buyer].nlargest(50, "fit")
```

**Engineering: triage, and a verifier for a code review agent.**

```python
tickets = tickets.join(tickets.hunch.ask({
    "kind":     Classify(["bug", "feature request", "question"]),
    "severity": Rate(["cosmetic", "degraded, workaround exists", "blocked", "outage"]),
}))
real = bot_comments[hunch.verify(bot_comments["comment"], diff)]   # drop comments the diff doesn't support
```

**SEO: intent, thin pages, and a title tag Jev picks from LLM drafts.**

```python
pages = pages.join(pages.hunch.ask({
    "intent": Classify(["informational", "commercial", "transactional", "navigational"]),
    "thin":   Check("adds nothing over the top results for its query"),
}))
best = hunch.pick(hunch.generate(str, n=10, instructions="title tags", context=page), "most likely to earn the click")
```

**Finance: anomalies and categorization.**

```python
suspect = txns.hunch.where("looks like a duplicate or erroneous charge", columns=["merchant", "amount", "date", "memo"])
txns["account"] = txns.hunch.classify(GLAccount, columns=["merchant", "memo"])   # your Enum of GL accounts
```

## Bring your own model

Jev is always the judge. The LLM for `generate`, `discover`, `refine`, and escalation can be whatever your team already pays for:

```python
hunch.configure(llm=hunch.anthropic())                          # Claude, via the anthropic SDK
hunch.configure(llm=hunch.openai(model="gpt-5-mini"))
hunch.configure(llm=hunch.azure(deployment="my-gpt"))           # Azure OpenAI
hunch.configure(llm=hunch.openrouter(model="z-ai/glm-5.3-flash"))
hunch.configure(llm=hunch.ollama("qwen3"))                      # local, nothing leaves your machine
hunch.configure(llm=lambda system, user: my_gateway(system, user))  # anything else
```

Keys come from the usual environment variables (`ANTHROPIC_API_KEY`, `OPENAI_API_KEY`, `AZURE_OPENAI_API_KEY`, `OPENROUTER_API_KEY`) or `api_key=`. `hunch.anthropic()` needs `pip install anthropic` and defaults to `claude-opus-5`; pass `model=` for another Claude. Any verb that uses an LLM also takes `llm=` to override the default for one call.

## Built for real columns

- **Dedupe and cache.** A value is asked once. With `cache="~/.cache/hunch"`, answers persist on disk, so re-running a notebook is free.
- **Parallel.** `max_workers` threads for normal calls. The `_async` twins run requests on your event loop through the async SDK client, up to `max_concurrency` at a time.
- **Survives failures.** With `errors="skip"`, a request that fails after the SDK's retries comes back `None` with a warning, instead of sinking the other 49,999. Running the same call again re-sends only the failures.
- **Rate limits.** `max_rps=20` caps requests per second.
- **Know the cost first.** Code inside `with hunch.dry_run() as plan:` sends nothing and tells you how many requests it would have made.
- **Progress.** Calls needing 10 or more requests show a bar that counts requests, so dedupe and cache hits are already reflected. `generate` shows a timer.

```python
hunch.configure(api_key=..., llm=hunch.anthropic(), cache="~/.cache/hunch",
                max_workers=8, max_concurrency=64, max_rps=None, errors="skip", progress="auto")
```

## Reference

| Verb | What it does | Returns |
| --- | --- | --- |
| `classify(data, labels, multi_label=False, split=, unsure=)` | Pick one label (or several) | label, Enum member, or list |
| `score(data, levels, instructions=)` | Place on an ordered scale, 2 to 10 levels | float from 0 to n-1, or `{dim: float}` |
| `check(data, statement, criteria=, threshold=0.5)` | Yes or no | bool, or `{name: bool}` |
| `where(data, statement, threshold=0.5)` | Keep rows the statement holds for | the matching rows, strongest first |
| `ask(data, {name: Classify \| Rate \| Check})` | Several questions, one request per row | dict per item, or a DataFrame |
| `pick(candidates, instructions)` | Choose the best candidate | the winner, or its index label |
| `rank(candidates, dimensions, levels, weights=)` | Weighted multi-dimension ranking | `Ranked` rows, or a sorted DataFrame |
| `generate(target, n=1, instructions=)` | LLM makes typed data | `target` or `list[target]` |
| `discover(data, n=8, instructions=)` | LLM proposes categories | `{name: description}` for `classify` |
| `refine(data, checks, rounds=3)` | LLM rewrites until Jev's checks pass | text in the same container |
| `verify(claims, source)` | Is each claim supported by the source? | bool, or a Series named `supported` |
| `evaluate(predicted, truth)` | Accuracy, by shape, misses, confusion matrix | `Evaluation` |
| `tune_threshold(p, truth, precision= \| recall=)` | Pick a cutoff from labeled rows | `Threshold` |

Every verb takes a string, a list, a Series, or a DataFrame, and returns the same shape. With a DataFrame, each row is the thing being judged. The Jev verbs take `context=` for state that rides along with each row, `columns=` to narrow a DataFrame, and `client=`. Most have an `_async` twin, and all are on the `df.hunch` / `series.hunch` accessor.

**`detail=True`** returns the full distribution. On pandas it spreads into columns, like `label`, `label_p`, `label_confidence`, `label_shape` for classify, `score_level` for score, and `check_p` for check. On lists you get `Answer`, `Rating`, `Feeling`, `Pick`, or `Refined` objects. `Answer.on(sure=, split=, unsure=)` branches on shape for custom logic.

**Context per question.** In `ask`, `Classify`, `Rate`, and `Check` take their own `context=`, so something only one question should see doesn't leak into the others.

**Shapes** come from `ShapePolicy(sure_peak=0.8, unsure_peak=0.5, split_margin=0.15, split_mass=0.75)`, computed from the probabilities alone. Jev's `confidence` is derived from the top probability, so it adds nothing. None of these say whether a label is correct, which is what `evaluate` is for. Changing the policy never re-runs inference.

## Examples

Single files with the data inline or generated. The first three need only `TYPESAFE_API_KEY`, and the rest also use an LLM key.

| File | Shows |
| --- | --- |
| [`find_angry_reviews.py`](examples/find_angry_reviews.py) | `check` as a boolean mask, ranking by probability, several checks in one request |
| [`classify_job_titles.py`](examples/classify_job_titles.py) | `classify` with Enums, shapes, and a rematch for split rows |
| [`triage_tickets.py`](examples/triage_tickets.py) | Two `score` scales in one request, multi-label tags, paging policy in code |
| [`review_themes.py`](examples/review_themes.py) | `discover` themes in generated reviews, `classify` with LLM escalation, `where` for churn risk |
| [`dating_profiles.py`](examples/dating_profiles.py) | Typed `generate`, `ask`, `score` against a described person, `pick`, and `where` for cat people and waiter-yellers |
| [`introduce_hunch.py`](examples/introduce_hunch.py) | `generate` 20 tweets, `rank`, `pick` the winner, then `refine` it against the README |
| [`organize_downloads.py`](examples/organize_downloads.py) | An LLM proposes folders, `classify` assigns every file, the script moves them. `--dry-run` prints the plan |

## What this is not

Jev never invents labels. Whatever you pass as `labels` is the entire set of allowed answers, and that constraint is the point. LLMs are only in the building to propose: drafts, categories, rewrites, second opinions on hard rows. They have no tools and take no actions. If you want open-ended writing or a multi-step agent, this is the wrong library, on purpose.

See [CHANGELOG.md](CHANGELOG.md) for what changed between versions.

## License

MIT. Jev and TypeSafe are [typesafe.ai](https://typesafe.ai); this library is not affiliated.
