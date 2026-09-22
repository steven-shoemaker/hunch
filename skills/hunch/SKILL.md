---
name: hunch
description: >
  Use the hunch Python library (pip install hunch-jev) to put judgment into code: label,
  score, filter, rank, or verify strings, lists, or pandas columns with Jev (TypeSafe's
  classifier), with an optional LLM that only proposes. Use when a task needs semantic
  judgment over data, such as categorizing rows, filtering by meaning ("angry reviews",
  "ICP fit"), triage, picking the best of several drafts, finding themes, checking an
  LLM's output against a source, or replacing prompt-then-parse LLM code with typed answers.
---

# Using hunch

hunch turns Jev judgments into plain functions. Hand a verb a string, list, Series, or
DataFrame; get the same shape back. Jev decides. An LLM, if configured, only proposes
(drafts, categories, rewrites, second opinions on hard rows). Code owns thresholds,
weights, and actions.

## Setup

```python
import hunch                                   # pip install hunch-jev
hunch.configure(api_key=TYPESAFE_API_KEY)      # or TYPESAFE_API_KEY in the environment
hunch.configure(api_key=..., llm=hunch.anthropic(), cache="~/.cache/hunch")  # with an LLM + disk cache
```

LLM adapters: `hunch.anthropic()`, `hunch.openai()`, `hunch.azure(deployment=...)`,
`hunch.openrouter()`, `hunch.ollama("model")`, or any function `(system, user) -> str`.
Only `generate`, `discover`, `refine`, and escalation need an LLM. Never hardcode keys.

## Pick the verb

| The task | Verb | Jev primitive |
| --- | --- | --- |
| Put each row in one of known categories | `classify(data, labels)` | Choice |
| Several labels can apply | `classify(data, labels, multi_label=True)` | one Noul per label |
| Place on an ordered scale (severity, fit) | `score(data, levels)` | Score |
| Yes/no about each row | `check(data, statement)` | Noul |
| Keep only rows that match a description | `where(data, statement)` | Noul, filtered |
| Several questions about the same rows | `ask(data, {name: Classify/Rate/Check})` | one request per row |
| Best of N candidates | `pick(candidates, instructions)` | one Choice |
| Order candidates on weighted criteria | `rank(candidates, {dim: question}, levels, weights=)` | Score per dim |
| Categories unknown | `discover(data, n)` then `classify(data, result)` | LLM proposes |
| Draft text that must meet rules | `refine(text, checks)` | LLM writes, Noul checks |
| Is an LLM/agent claim supported? | `verify(claims, source)` | Noul |
| Make fake or seed data | `generate(type, n, instructions=)` | LLM |
| How accurate is it on my data? | `evaluate(pred, truth)`, `tune_threshold(p, truth, precision=)` | none |

Every verb is also on `df.hunch.<verb>(...)` and `series.hunch.<verb>(...)`.

## Core patterns

```python
df["theme"] = df["review"].hunch.classify({"billing": "charges, refunds", "shipping": "late or damaged"})
df = df.join(df.hunch.ask({
    "kind":  hunch.Classify(["bug", "feature request", "question"]),
    "sev":   hunch.Rate(["cosmetic", "degraded", "blocked", "outage"]),
    "angry": hunch.Check("the customer is frustrated"),
}, columns=["subject", "body"]))
buyers = prospects.hunch.where("could sign a purchase for their team", columns=["title", "company"])
best = hunch.pick(drafts, "most likely to get a reply from a busy CFO")
```

- A DataFrame row is the state Jev sees; use `columns=` to limit what it reads.
- `context=` adds shared state (an ICP, a policy, a diff). In `ask`, put context on the one
  question that needs it (`Rate([...], "...", context={...})`), or it colors every answer.
- `detail=True` returns probabilities. On pandas it spreads into columns:
  `label/label_p/label_shape`, `score/score_level`, `check/check_p`.

## Handling uncertainty

Each answer has a shape: `sure`, `split` (two labels close), or `unsure` (flat).

```python
df["level"] = df["title"].hunch.classify(LEVELS, split="rematch", unsure="review")
df["level"] = df["title"].hunch.classify(LEVELS, unsure=hunch.anthropic())  # escalate hard rows to an LLM
```

- Confidence measures how peaked the distribution is, not correctness. Before trusting a
  pipeline, label 50 to 100 rows and run `hunch.evaluate(pred_detail, truth)`.
- For `check`/`where` thresholds, use `hunch.tune_threshold(p_detail, truth, precision=0.9)`
  instead of guessing a number.

## Writing good questions

- Labels: give descriptions when names are ambiguous (`{"Manager": "manages people, not accounts"}`).
  Include an `other` label when inputs may fit nothing.
- Score levels: 2 to 10 concrete situations, worst first ("broken, workaround exists"),
  not degrees ("moderately bad").
- `where`/`check`: statements about evidence in the row filter well ("mentions a refund").
  Predictions about behavior ("might cancel") sit near 0.3 to 0.4 when the row says nothing;
  rank them (`detail=True`, sort by `match_p`) instead of thresholding.
- `refine` checks only test what you list. Add an accuracy check plus source context
  ("describes the product correctly according to the readme in context"), or the LLM
  may write fluent, rule-passing text that is false.

## Scale and cost

```python
hunch.configure(api_key=..., cache="~/.cache/hunch", max_workers=8, max_rps=20, errors="skip")
with hunch.dry_run() as plan:
    df.hunch.ask({...})
print(plan)  # requests it would send; nothing is sent or cached
```

- Duplicate values are asked once; cached answers cost nothing on rerun.
- `errors="skip"`: failed rows come back None with a warning, and a rerun re-sends only those.
- Async twins (`classify_async`, `ask_async`, ...) run requests on the event loop; use them in services.
- `hunch.default().usage` reports calls, cache hits, and tokens.

## Don'ts

- Don't loop over rows calling a verb per row; pass the whole list/Series/DataFrame.
- Don't prompt an LLM and parse JSON for a closed-set decision; use `classify`/`check`/`score`.
- Don't let an LLM make the final call; route it through `pick`, `verify`, or escalation.
- Don't pass a DataFrame when only one column matters; pass the Series or use `columns=`.

Full signatures, return types, and detail columns: [references/api.md](references/api.md).
Jev primitives and prompting guidance: https://docs.typesafe.ai/llms.txt
