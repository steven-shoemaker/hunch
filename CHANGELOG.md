# Changelog

## 0.7.0

- `errors="skip"` on the client: a request that still fails after the SDK's retries returns `None` for its rows, with a warning, instead of raising. Good answers are cached, so a rerun only re-sends failures.
- `max_rps=` caps requests per second across sync and async calls.
- `_async` verbs now run requests as coroutines on your event loop through the async SDK client, `max_concurrency=` at a time (default 64). They used to wrap the sync call in a thread.
- `columns=` on every verb, not just `where`.
- `Classify`, `Rate`, and `Check` take `context=` for that question only. `ask` groups questions by context.
- `hunch.dry_run()` counts the requests a block would send, without sending anything.
- `hunch.evaluate(predicted, truth)` reports accuracy, accuracy by shape, the misses, and a confusion matrix.
- `hunch.tune_threshold(p, truth, precision= | recall=)` picks a `check` / `where` cutoff from labeled rows.
- `generate` draws large `n` in batches of 25 that avoid repeats, and caches results. `fresh=True` draws new ones.
- **Changed:** `ShapePolicy` works on probabilities only: `sure_peak`, `unsure_peak`, `split_margin`, `split_mass`. Jev's confidence is derived from the top probability, so it added nothing. The old `sure_confidence` / `unsure_confidence` arguments are gone, and a few borderline rows may change shape.
- Type overloads: a Series or DataFrame in is typed as pandas out, a list as a list.

## 0.6.0

- `classify(split="rematch", unsure="review")`: policies for shaky answers, with batched rematches. Also on `Classify()` in `ask`.

## 0.5.1

- tqdm is a regular dependency. Plain text bars, no ipywidgets.

## 0.5.0

- `where(data, statement)`: semantic row filter, strongest match first.
- `df.hunch` / `series.hunch` accessor for every verb.

## 0.4.1

- Progress bars redraw per request and stay when done. The `generate` timer ticks.

## 0.4.0

- DataFrames as input: each row is the state.
- `detail=True` on pandas spreads answers into columns.
- `pick` returns the winner's index label; `rank` returns a sorted DataFrame.
- **Changed:** single-question `score` / `check` name their question `score` / `check` instead of `q`.

## 0.3.2

- `generate` shows an elapsed timer.

## 0.3.1

- Progress bars for calls needing 10+ requests.

## 0.3.0

- `ask(data, {name: Classify | Rate | Check})`: several questions, one request per item.

## 0.2.1

- README refresh.

## 0.2.0

- **Changed:** replaced the session / `@over` API with plain verbs: `classify`, `score`, `check`, `pick`, `rank`, `generate`.
