# hunch API reference

Signatures and docstrings from the package. The Jev verbs have `_async` twins and are on `df.hunch` / `series.hunch`.

## classify

```python
hunch.classify(data: Any, labels: Any, *, columns: Sequence[str] | None = None, multi_label: bool = False, instructions: str | None = None, context: Any = None, threshold: float = 0.5, split: Any = KEEP, unsure: Any = KEEP, detail: bool = False, client: Client | None = None) -> Any
```

Assign data to one of the labels (or several with multi_label=True).

labels: a sequence of strings, an Enum class, or a mapping label -> description.
Returns the label (an Enum member when labels is an Enum). detail=True returns
Answer / MultiAnswer with the full distribution. columns= narrows a DataFrame.

split= and unsure= are policies for shaky answers. split="rematch" re-asks between the
top two options for rows where two labels were close. An LLM (hunch.anthropic(), any
adapter, or a function (system, user) -> str) escalates those rows: it picks from the
same labels, and Answer.by says "llm". Any other value is returned as the label for
those rows. unsure= does the same for flat distributions. Both default to keeping the
first answer.

## score

```python
hunch.score(data: Any, levels: Sequence[str], *, columns: Sequence[str] | None = None, instructions: str | Mapping[str, str] | None = None, context: Any = None, detail: bool = False, client: Client | None = None) -> Any
```

Place data on an ordered scale. Returns the position 0 .. len(levels)-1.

levels: 2–10 concrete situations, worst first. instructions may be a mapping of
dimension name -> question to score several dimensions in one request; the result
is then a dict per item. detail=True returns Rating(s). columns= narrows a DataFrame.

## check

```python
hunch.check(data: Any, statement: str | Mapping[str, str], *, columns: Sequence[str] | None = None, criteria: Mapping[str, str | None] | None = None, context: Any = None, threshold: float = 0.5, detail: bool = False, client: Client | None = None) -> Any
```

Does the statement hold for the data? Returns bool (P(yes) >= threshold).

statement may be a mapping name -> statement to check several in one request.
criteria={"true": ..., "false": ...} sharpens the boundary. detail=True returns Feeling(s).
columns= narrows a DataFrame.

## where

```python
hunch.where(data: Any, statement: str, *, columns: Sequence[str] | None = None, criteria: Mapping[str, str | None] | None = None, context: Any = None, threshold: float = 0.5, detail: bool = False, client: Client | None = None) -> Any
```

Semantic filter: keep the rows (or items) for which the statement holds.

Results come back sorted by how strongly they match. columns= limits what Jev
reads from a DataFrame; the whole row is still returned. detail=True returns every
row with `match` and `match_p` columns instead of filtering. Skipped requests don't match.

## ask

```python
hunch.ask(data: Any, questions: Mapping[str, Spec], *, columns: Sequence[str] | None = None, context: Any = None, detail: bool = False, client: Client | None = None) -> Any
```

Ask several questions about the same data, one Jev request per item.

questions maps a name to Classify(...), Rate(...), or Check(...). One item returns a
dict of name -> answer; a list returns a list of dicts; a Series or DataFrame returns
a DataFrame with one column per question on the same index (detail=True spreads each
answer into several columns). A question's own context= is merged over context= here;
questions whose merged context differs go in separate requests.

## pick

```python
hunch.pick(candidates: Any, instructions: str, *, columns: Sequence[str] | None = None, context: Any = None, detail: bool = False, client: Client | None = None) -> Any
```

Choose the single best candidate. Jev compares them head to head in one Choice.

A list returns the winning item. A Series or DataFrame returns the winner's index
label, so df.loc[winner] is the row; columns= limits what Jev reads. More than 255
candidates run as a tournament. detail=True returns Pick with every candidate's probability.

## rank

```python
hunch.rank(candidates: Any, dimensions: str | Mapping[str, str], levels: Sequence[str], *, columns: Sequence[str] | None = None, weights: Mapping[str, float] | None = None, context: Any = None, client: Client | None = None) -> Any
```

Score every candidate on each dimension, weight, and sort best first.

Composite = weighted mean of normalized (0–1) dimension scores. Weights default to 1.
A list returns Ranked rows. A Series or DataFrame returns a DataFrame on the same
index with `composite` and one column per dimension, sorted best first; columns=
limits what Jev reads. Skipped requests get a NaN composite and sort last.

## generate

```python
hunch.generate(target: Any = <class str>, n: int = 1, *, instructions: str | None = None, context: Any = None, llm: LanguageModel | None = None, fresh: bool = False, client: Client | None = None) -> Any
```

Create n examples of target: str, int, a dataclass, a TypedDict, list[str], ...

Returns one value when n == 1, else a list of exactly n distinct items. Uses llm= here,
else the client's llm= from configure(). Output is validated against the target type.
n above 25 is drawn in batches that avoid repeating earlier items. Results are cached
on the client (and on disk with cache=), so re-running the same call returns the same
items; fresh=True draws new ones.

## discover

```python
hunch.discover(data: Any, n: int = 8, *, instructions: str | None = None, columns: Sequence[str] | None = None, sample: int = 100, other: bool = True, llm: Any = None, fresh: bool = False, client: Client | None = None) -> dict[str, str]
```

Let an LLM read a sample and propose n categories, as {name: description}.

The result plugs straight into classify(data, labels=...), so Jev files every row into
categories nobody had to write by hand. instructions= steers what to group by
("by the customer's complaint", "by buying intent"). other=True adds an "other"
category so rows that fit nothing aren't forced into one. Cached like generate().

## refine

```python
hunch.refine(data: Any, checks: Sequence[str] | Mapping[str, str], *, instructions: str | None = None, rounds: int = 3, threshold: float = 0.5, context: Any = None, llm: Any = None, detail: bool = False, client: Client | None = None) -> Any
```

Rewrite text with an LLM until Jev agrees every check holds, or rounds run out.

checks are statements a good version makes true ("has no hype words", "shows the
install command"), as a list or {name: statement}. Each round, Jev checks every draft
at once; only the failing drafts go back to the LLM, told which checks failed.
Returns the final text in the caller's container. detail=True returns Refined objects,
or for pandas a DataFrame with text / passed / rounds / failed.

## verify

```python
hunch.verify(claims: Any, source: Any, *, threshold: float = 0.5, context: Any = None, detail: bool = False, client: Client | None = None) -> Any
```

Is each claim supported by its source? For checking what an LLM or agent produced.

source is one document for all claims, or a sequence / Series aligned with claims
(one source per claim). Returns bools in the caller's container (a Series named
`supported` for pandas); detail=True gives probabilities too. Only what the source
states counts: missing, contradicted, or partly supported claims are False.

## evaluate

```python
hunch.evaluate(predicted: Any, truth: Any) -> Evaluation
```

Compare classify() output with true labels.

predicted: labels, Answer objects, or the DataFrame from classify(..., detail=True).
truth: the correct labels, aligned by position (or by index when both are pandas).
With shapes available, by_shape shows whether "sure" rows really are more accurate.

## tune_threshold

```python
hunch.tune_threshold(probabilities: Any, truth: Any, *, precision: float | None = None, recall: float | None = None) -> Threshold
```

Find the cutoff for check() / where() from labeled rows.

probabilities: P(yes) per row, as floats, Feelings, or the DataFrame from
check(..., detail=True) / where(..., detail=True). truth: True/False per row.
precision=0.9 returns the lowest cutoff that keeps at least 90% of matches correct,
which lets through as many true matches as possible. recall=0.9 returns the highest
cutoff that still catches 90% of true matches. With neither, it maximizes F1.

## configure / Client

```python
hunch.configure(api_key: str | None = None, *, model: str | None = None, client: Any | None = None, async_client: Any | None = None, llm: Any | None = None, cache: str | Path | None = None, policy: ShapePolicy | None = None, max_workers: int = 8, max_concurrency: int = 64, max_rps: float | None = None, errors: str = raise, progress: bool | str = auto) -> None
```

Open a Jev client. Reads TYPESAFE_API_KEY and TYPESAFE_DEFAULT_MODEL when omitted.

client= / async_client= inject objects with a system_one(state=, questions=) method.
llm= is a LanguageModel for generate(). cache= persists answers on disk.
max_workers= threads for sync calls; max_concurrency= requests in flight for _async calls.
max_rps= caps requests per second across both. errors="skip" turns a failed request
into None for its rows (with a warning) instead of raising.
progress= shows a progress bar per verb call: True, False, or "auto" (10+ requests).

## dry_run

```python
with hunch.dry_run() as plan:
    ...
plan.requests, plan.questions, plan.items
```

Count what the enclosed hunch calls would send to Jev, without sending anything.

Verbs return placeholder answers inside the block (first label, score 0, check False),
so code after them keeps running. Nothing is cached.

## Specs for ask

```python
hunch.Classify(labels, instructions=None, split=KEEP, unsure=KEEP, context=None)
hunch.Rate(levels, instructions=None, context=None)
hunch.Check(statement, criteria=None, threshold=0.5, context=None)
```

## detail=True columns on pandas

| Verb | Columns |
| --- | --- |
| classify | `label`, `label_p`, `label_confidence`, `label_shape`, plus `label_by` when an LLM policy is set |
| classify(multi_label=True) | `labels`, `labels_p` (dict of label to P) |
| score | `score`, `score_level`, `score_confidence`, `score_shape` (or per dimension name) |
| check | `check`, `check_p` (or per statement name) |
| where | the input columns plus `match`, `match_p` |
| ask | per question name, as above |
| verify | `supported`, `supported_p` |
| refine | `text`, `passed`, `rounds`, `failed` |
