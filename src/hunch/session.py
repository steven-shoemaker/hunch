from __future__ import annotations

import json
import threading
from contextlib import contextmanager
from contextvars import ContextVar
from dataclasses import dataclass
from typing import Any, Iterator, Mapping, Sequence

from typesafe_sdk import Choice, Noul, Score

from hunch.answer import Answer, Feeling, Rating
from hunch.exceptions import HunchError, NoSessionError
from hunch.role import Draft, Role
from hunch.shapes import ShapePolicy
from hunch.subject import encode_state
from hunch.usage import Meter

_SESSION: ContextVar[Session | None] = ContextVar("hunch_session", default=None)


def current_session() -> Session:
    session = _SESSION.get()
    if session is None:
        raise NoSessionError()
    return session


def ask(
    state: Any,
    question: str,
    *,
    among: Sequence[str],
    by: str | Mapping[str, Any] | None = None,
) -> LazyAnswer:
    return current_session().choice(state, question, among=among, by=by)


def rate(
    state: Any,
    question: str,
    levels: Sequence[str],
    *,
    by: str | None = None,
) -> LazyRating:
    return current_session().score(state, question, levels=levels, by=by)


@dataclass
class Pending:
    qid: str
    kind: str
    state: Any
    state_key: str
    cache_key: str
    question: Choice | Noul | Score
    resolved: Answer | Feeling | Rating | None = None


class Session:
    def __init__(
        self,
        client: Any,
        policy: ShapePolicy,
        cache: dict[str, Any],
        lock: threading.Lock,
        llm: Any | None = None,
        meter: Meter | None = None,
    ) -> None:
        self.client = client
        self.policy = policy
        self.cache = cache
        self.lock = lock
        self.llm = llm
        self.meter = meter or Meter()
        self.pending: list[Pending] = []
        self._seq = 0
        self._role_drafts: dict[int, int] = {}

    def choice(
        self,
        state: Any,
        question: str,
        *,
        among: Sequence[str],
        by: str | Mapping[str, Any] | None,
    ) -> LazyAnswer:
        options = _unique_options(among)
        built = _choice_question(question, options, by)
        pending = self._enqueue("choice", state, built)
        return LazyAnswer(self, pending)

    def noul(
        self,
        state: Any,
        description: str,
        *,
        true: str | None,
        false: str | None,
    ) -> LazyFeeling:
        criteria = None
        if true is not None or false is not None:
            criteria = {"true": true, "false": false}
        built = Noul(instructions=description, criteria=criteria)
        pending = self._enqueue("noul", state, built)
        return LazyFeeling(self, pending)

    def score(
        self,
        state: Any,
        question: str,
        *,
        levels: Sequence[str],
        by: str | None,
    ) -> LazyRating:
        if not levels:
            raise HunchError("rate() needs at least one level.")
        instructions: Any = {"question": question, "rules": by} if by and by.strip() else question
        built = Score(instructions=instructions, criteria=list(levels))
        pending = self._enqueue("score", state, built)
        return LazyRating(self, pending)

    def write(self, state: Any, who: Role) -> Draft:
        from hunch.role import parse_draft, render_system

        via = who.via or self.llm
        if via is None:
            raise HunchError(
                "draft() needs a language model. Pass via= on the role or llm= on connect()."
            )
        encoded = encode_state(state)
        cache_key = _dump(
            {
                "kind": "draft",
                "state": encoded,
                "instructions": who.instructions,
                "emit": who.emit,
                "model": getattr(via, "name", None),
            }
        )
        with self.lock:
            cached = self.cache.get(cache_key)
        if isinstance(cached, Draft):
            self.meter.hit()
            return cached
        used = self._role_drafts.get(id(who), 0)
        if used >= who.max_loops:
            raise HunchError(f"draft() stopped at the role loop limit ({who.max_loops}).")
        text = via.complete(system=render_system(who), user=_dump(encoded))
        result = parse_draft(text, who)
        self._role_drafts[id(who)] = used + 1
        with self.lock:
            self.cache.set(cache_key, result)
        return result

    def _enqueue(self, kind: str, state: Any, question: Choice | Noul | Score) -> Pending:
        encoded = encode_state(state)
        state_key = _dump(encoded)
        cache_key = _dump(
            {
                "kind": kind,
                "state": encoded,
                "instructions": getattr(question, "instructions", None),
                "criteria": getattr(question, "criteria", None),
            }
        )
        with self.lock:
            cached = self.cache.get(cache_key)
        self._seq += 1
        pending = Pending(
            qid=f"q{self._seq}",
            kind=kind,
            state=encoded,
            state_key=state_key,
            cache_key=cache_key,
            question=question,
            resolved=cached,
        )
        if cached is None:
            self.pending.append(pending)
        else:
            self.meter.hit()
        return pending

    def flush(self) -> None:
        groups: dict[str, list[Pending]] = {}
        for item in self.pending:
            if item.resolved is None:
                groups.setdefault(item.state_key, []).append(item)
        self.pending = []
        for batch in groups.values():
            self._dispatch(batch)

    def _dispatch(self, batch: list[Pending]) -> None:
        questions = {item.qid: item.question for item in batch}
        with self.lock:
            raw = self.client.system_one(state=batch[0].state, questions=questions)
        self.meter.call(raw)
        for item in batch:
            item.resolved = self._read(item, raw)
            with self.lock:
                self.cache.set(item.cache_key, item.resolved)

    def _read(self, item: Pending, raw: Any) -> Answer | Feeling | Rating:
        if item.kind == "noul":
            noul = _noul_value(raw, item.qid)
            return Feeling(p=float(noul))
        if item.kind == "score":
            score, confidence, legend, probabilities = _score_value(raw, item.qid)
            shaped = {legend[level]: mass for level, mass in probabilities.items()}
            return Rating(
                score=score,
                confidence=confidence,
                legend=legend,
                probabilities=probabilities,
                shape=self.policy.classify(shaped, confidence),
            )
        choice, confidence, probabilities = _choice_value(raw, item.qid)
        return Answer(
            top=str(choice),
            probabilities={str(k): float(v) for k, v in probabilities.items()},
            confidence=float(confidence),
            shape=self.policy.classify(probabilities, float(confidence)),
        )


class LazyAnswer:
    def __init__(self, session: Session, pending: Pending) -> None:
        self._session = session
        self._pending = pending

    def _get(self) -> Answer:
        if self._pending.resolved is None:
            self._session.flush()
        resolved = self._pending.resolved
        if not isinstance(resolved, Answer):
            raise HunchError("Internal error: expected a Choice answer.")
        return resolved

    @property
    def top(self) -> str:
        return self._get().top

    @property
    def top2(self) -> list[str]:
        return self._get().top2

    @property
    def p(self) -> float:
        return self._get().p

    @property
    def confidence(self) -> float:
        return self._get().confidence

    @property
    def probabilities(self) -> Mapping[str, float]:
        return self._get().probabilities

    @property
    def shape(self) -> str:
        return self._get().shape

    def on(self, *, sure: Any, torn: Any, lost: Any) -> Any:
        return self._get().on(sure=sure, torn=torn, lost=lost)


class LazyFeeling:
    def __init__(self, session: Session, pending: Pending) -> None:
        self._session = session
        self._pending = pending

    def _get(self) -> Feeling:
        if self._pending.resolved is None:
            self._session.flush()
        resolved = self._pending.resolved
        if not isinstance(resolved, Feeling):
            raise HunchError("Internal error: expected a Noul answer.")
        return resolved

    @property
    def p(self) -> float:
        return self._get().p

    def __bool__(self) -> bool:
        return bool(self._get())


class LazyRating:
    def __init__(self, session: Session, pending: Pending) -> None:
        self._session = session
        self._pending = pending

    def _get(self) -> Rating:
        if self._pending.resolved is None:
            self._session.flush()
        resolved = self._pending.resolved
        if not isinstance(resolved, Rating):
            raise HunchError("Internal error: expected a Score answer.")
        return resolved

    @property
    def score(self) -> float:
        return self._get().score

    @property
    def level(self) -> str:
        return self._get().level

    @property
    def top(self) -> str:
        return self._get().top

    @property
    def confidence(self) -> float:
        return self._get().confidence

    @property
    def probabilities(self) -> Mapping[int, float]:
        return self._get().probabilities

    @property
    def legend(self) -> Mapping[int, str]:
        return self._get().legend

    @property
    def shape(self) -> str:
        return self._get().shape

    def on(self, *, sure: Any, torn: Any, lost: Any) -> Any:
        return self._get().on(sure=sure, torn=torn, lost=lost)


@contextmanager
def activate(
    client: Any,
    policy: ShapePolicy,
    cache: Any,
    lock: threading.Lock,
    llm: Any | None = None,
    meter: Meter | None = None,
) -> Iterator[Session]:
    session = Session(client, policy, cache, lock, llm=llm, meter=meter)
    token = _SESSION.set(session)
    try:
        yield session
    finally:
        _SESSION.reset(token)


def _unique_options(among: Sequence[str]) -> list[str]:
    options = [str(option) for option in among]
    if not options:
        raise HunchError("ask() needs at least one option in among=.")
    if len(set(options)) != len(options):
        raise HunchError("among= options must be unique.")
    return options


def _choice_question(
    question: str,
    options: Sequence[str],
    by: str | Mapping[str, Any] | None,
) -> Choice:
    if isinstance(by, Mapping):
        criteria = {label: by.get(label) for label in options}
        instructions: Any = question
    elif isinstance(by, str) and by.strip():
        criteria = {label: None for label in options}
        instructions = {"question": question, "rules": by}
    else:
        criteria = {label: None for label in options}
        instructions = question
    return Choice(instructions=instructions, criteria=criteria)


def _dump(value: Any) -> str:
    return json.dumps(value, sort_keys=True, default=str)


def _choice_value(raw: Any, qid: str) -> tuple[str, float, Mapping[str, float]]:
    answers = getattr(raw, "answers", None)
    choices = getattr(raw, "choices", None)
    payload = None
    if answers is not None:
        payload = _lookup(answers, qid)
    if payload is None and choices is not None:
        payload = _lookup(choices, qid)
    if payload is None and isinstance(raw, Mapping):
        nested = raw.get("answers") or raw.get("choices") or raw
        payload = nested.get(qid) if isinstance(nested, Mapping) else None
    if payload is None:
        raise HunchError(f"Jev response is missing choice {qid!r}.")
    choice = getattr(payload, "choice", None)
    if choice is None and isinstance(payload, Mapping):
        choice = payload.get("choice")
    confidence = getattr(payload, "confidence", None)
    if confidence is None and isinstance(payload, Mapping):
        confidence = payload.get("confidence", 0.0)
    probabilities = getattr(payload, "probabilities", None)
    if probabilities is None and isinstance(payload, Mapping):
        probabilities = payload.get("probabilities")
    if choice is None or probabilities is None:
        raise HunchError(f"Jev choice {qid!r} is missing a label or probabilities.")
    return str(choice), float(confidence or 0.0), probabilities


def _noul_value(raw: Any, qid: str) -> float:
    answers = getattr(raw, "answers", None)
    nouls = getattr(raw, "nouls", None)
    payload = None
    if answers is not None:
        payload = _lookup(answers, qid)
    if payload is None and nouls is not None:
        payload = _lookup(nouls, qid)
    if payload is None and isinstance(raw, Mapping):
        nested = raw.get("answers") or raw.get("nouls") or raw
        payload = nested.get(qid) if isinstance(nested, Mapping) else None
    if payload is None:
        raise HunchError(f"Jev response is missing noul {qid!r}.")
    value = getattr(payload, "noul", None)
    if value is None and isinstance(payload, Mapping):
        value = payload.get("noul")
    if value is None:
        raise HunchError(f"Jev noul {qid!r} is missing a probability.")
    return float(value)


def _score_value(
    raw: Any, qid: str
) -> tuple[float, float, dict[int, str], dict[int, float]]:
    answers = getattr(raw, "answers", None)
    scores = getattr(raw, "scores", None)
    payload = None
    if answers is not None:
        payload = _lookup(answers, qid)
    if payload is None and scores is not None:
        payload = _lookup(scores, qid)
    if payload is None and isinstance(raw, Mapping):
        nested = raw.get("answers") or raw.get("scores") or raw
        payload = nested.get(qid) if isinstance(nested, Mapping) else None
    if payload is None:
        raise HunchError(f"Jev response is missing score {qid!r}.")
    score = getattr(payload, "score", None)
    if score is None and isinstance(payload, Mapping):
        score = payload.get("score")
    confidence = getattr(payload, "confidence", None)
    if confidence is None and isinstance(payload, Mapping):
        confidence = payload.get("confidence", 0.0)
    legend = getattr(payload, "legend", None)
    if legend is None and isinstance(payload, Mapping):
        legend = payload.get("legend")
    probabilities = getattr(payload, "probabilities", None)
    if probabilities is None and isinstance(payload, Mapping):
        probabilities = payload.get("probabilities")
    if score is None or legend is None or probabilities is None:
        raise HunchError(f"Jev score {qid!r} is missing a value, legend, or probabilities.")
    return (
        float(score),
        float(confidence or 0.0),
        {int(k): str(v) for k, v in dict(legend).items()},
        {int(k): float(v) for k, v in dict(probabilities).items()},
    )


def _lookup(container: Any, qid: str) -> Any:
    if container is None:
        return None
    if isinstance(container, Mapping):
        return container.get(qid)
    return getattr(container, qid, None)
