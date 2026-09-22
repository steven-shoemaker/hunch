"""Reach Jev through a gateway instead of TypeSafe directly: OpenRouter or Vercel AI Gateway.

Each adapter has the SDK's system_one(state=, questions=) shape, so the rest of hunch doesn't
know which route answered. Retries 408, 429, 5xx, and 529 with backoff.
"""

from __future__ import annotations

import json
import os
import random
import time
import urllib.error
import urllib.request
from types import SimpleNamespace
from typing import Any, Mapping

from hunch.exceptions import HunchError

RETRY = {408, 429, 500, 502, 503, 504, 529}


def _post(url: str, headers: Mapping[str, str], body: Any, timeout: float, retries: int = 3) -> tuple[dict, Mapping[str, str]]:
    data = json.dumps(body).encode()
    for attempt in range(retries + 1):
        request = urllib.request.Request(url, data=data, headers={"Content-Type": "application/json", "User-Agent": "hunch", **headers}, method="POST")
        try:
            with urllib.request.urlopen(request, timeout=timeout) as response:
                return json.loads(response.read().decode()), dict(response.headers)
        except urllib.error.HTTPError as error:
            if error.code not in RETRY or attempt == retries:
                detail = error.read().decode("utf-8", errors="replace")[:400]
                raise HunchError(f"Jev gateway request failed ({error.code}). {detail}".strip()) from error
        except urllib.error.URLError as error:
            if attempt == retries:
                raise HunchError(f"Jev gateway request failed: {error.reason}") from error
        time.sleep(min(8.0, 0.5 * 2**attempt) * (1 - 0.25 * random.random()))
    raise HunchError("unreachable")  # pragma: no cover


def _dump(questions: Mapping[str, Any]) -> dict[str, Any]:
    return {qid: q.model_dump(mode="json") if hasattr(q, "model_dump") else dict(q) for qid, q in questions.items()}


def _usage(u: Mapping[str, Any] | None) -> SimpleNamespace:
    u = u or {}
    return SimpleNamespace(input_tokens=u.get("input_tokens", u.get("inputTokens")), output_tokens=u.get("output_tokens", u.get("outputTokens")))


class OpenRouterJev:
    """Jev through OpenRouter's decisions endpoint (alpha). Same body as TypeSafe's API."""

    url = "https://openrouter.ai/api/alpha/decisions"

    def __init__(self, api_key: str | None = None, model: str = "~typesafe/jev-latest", timeout: float = 30.0) -> None:
        key = api_key or os.environ.get("OPENROUTER_API_KEY")
        if not key:
            raise HunchError('gateway="openrouter" needs OPENROUTER_API_KEY or api_key=.')
        self.key, self.model, self.timeout = key, model, timeout

    def system_one(self, state: Any = None, questions: Mapping[str, Any] | None = None, **_: Any) -> SimpleNamespace:
        body, _headers = _post(
            self.url,
            {"Authorization": f"Bearer {self.key}", "HTTP-Referer": "https://github.com/steven-shoemaker/hunch", "X-Title": "hunch"},
            {"model": self.model, "state": state, "questions": _dump(questions or {})},
            self.timeout,
        )
        return SimpleNamespace(model=body.get("model", self.model), answers=body.get("answers", {}), usage=_usage(body.get("usage")))


class VercelJev:
    """Jev through Vercel AI Gateway's evaluation-model endpoint.

    The gateway calls a noul question "boolean", drops score legends, and reports confidence
    separately; this adapter translates both ways.
    """

    url = "https://ai-gateway.vercel.sh/v4/ai/evaluation-model"

    def __init__(self, api_key: str | None = None, model: str = "typesafe-ai/jev", timeout: float = 30.0) -> None:
        key = api_key or os.environ.get("AI_GATEWAY_API_KEY")
        if not key:
            raise HunchError('gateway="vercel" needs AI_GATEWAY_API_KEY or api_key=.')
        self.key, self.model, self.timeout = key, model, timeout

    def system_one(self, state: Any = None, questions: Mapping[str, Any] | None = None, **_: Any) -> SimpleNamespace:
        dumped = _dump(questions or {})
        wire = {qid: ({**q, "type": "boolean"} if q.get("type") == "noul" else q) for qid, q in dumped.items()}
        body, _headers = _post(
            self.url,
            {
                "Authorization": f"Bearer {self.key}",
                "Ai-Gateway-Protocol-Version": "0.0.1",
                "Ai-Gateway-Auth-Method": "api-key",
                "Ai-Evaluation-Model-Specification-Version": "4",
                "Ai-Model-Id": self.model,
            },
            {"state": state, "questions": wire},
            self.timeout,
        )
        reported = ((body.get("providerMetadata") or {}).get("typesafe") or {}).get("confidence") or {}
        answers: dict[str, Any] = {}
        for qid, raw in (body.get("answers") or {}).items():
            kind = raw.get("type")
            if kind == "boolean":
                answers[qid] = {"type": "noul", "noul": float(raw["probability"])}
                continue
            probs = raw.get("probabilities") or {}
            confidence = reported.get(qid, _confidence(probs))
            if kind == "score":
                legend = {str(i): str(c) for i, c in enumerate(dumped[qid].get("criteria", []))}
                answers[qid] = {"type": "score", "score": raw["score"], "probabilities": probs, "legend": legend, "confidence": confidence}
            else:
                answers[qid] = {"type": "choice", "choice": raw["choice"], "probabilities": probs, "confidence": confidence}
        return SimpleNamespace(model=self.model, answers=answers, usage=_usage(body.get("usage")))


def _confidence(probabilities: Mapping[str, float]) -> float:
    """Jev's own formula: the top probability rescaled from uniform (0) to certain (1)."""
    values = [float(v) for v in probabilities.values()]
    n = len(values)
    return 0.0 if n < 2 else max(0.0, (n * max(values) - 1) / (n - 1))


def make(gateway: str, api_key: str | None, model: str | None) -> Any:
    if gateway == "openrouter":
        return OpenRouterJev(api_key, **({"model": model} if model else {}))
    if gateway == "vercel":
        return VercelJev(api_key, **({"model": model} if model else {}))
    raise HunchError('gateway= must be "typesafe", "openrouter", or "vercel".')
