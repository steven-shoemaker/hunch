from __future__ import annotations

import json
import os
import urllib.error
import urllib.request
from typing import Any, Protocol

from hunch.exceptions import HunchError


class LanguageModel(Protocol):
    name: str

    def complete(self, *, system: str, user: str) -> str: ...


class OpenAICompat:
    """Chat Completions client for OpenAI-compatible hosts (OpenAI, Cerebras, …)."""

    def __init__(
        self,
        *,
        api_key: str,
        model: str,
        base_url: str = "https://api.openai.com/v1",
        extra_body: dict[str, Any] | None = None,
        extra_headers: dict[str, str] | None = None,
        timeout: float = 30.0,
    ) -> None:
        self.api_key = api_key
        self.model = model
        self.base_url = base_url.rstrip("/")
        self.extra_body = extra_body or {}
        self.extra_headers = extra_headers or {}
        self.timeout = timeout
        self.name = model

    def complete(self, *, system: str, user: str) -> str:
        payload = {
            "model": self.model,
            "messages": [
                {"role": "system", "content": system},
                {"role": "user", "content": user},
            ],
            **self.extra_body,
        }
        request = urllib.request.Request(
            f"{self.base_url}/chat/completions",
            data=json.dumps(payload).encode(),
            headers={
                "Authorization": f"Bearer {self.api_key}",
                "Content-Type": "application/json",
                "User-Agent": "hunch/0.1",
                **self.extra_headers,
            },
            method="POST",
        )
        try:
            with urllib.request.urlopen(request, timeout=self.timeout) as response:
                body = json.loads(response.read().decode())
        except urllib.error.HTTPError as error:
            detail = error.read().decode("utf-8", errors="replace")[:400]
            raise HunchError(
                f"Language model request failed ({error.code}). {detail}".strip()
            ) from error
        except urllib.error.URLError as error:
            raise HunchError("Language model request failed.") from error
        text = (
            body.get("choices", [{}])[0]
            .get("message", {})
            .get("content")
        )
        if not isinstance(text, str) or not text.strip():
            raise HunchError("Language model returned no text.")
        return text.strip()


def openai(
    api_key: str | None = None,
    *,
    model: str = "gpt-4.1-mini",
    base_url: str = "https://api.openai.com/v1",
    extra_body: dict[str, Any] | None = None,
    extra_headers: dict[str, str] | None = None,
    timeout: float = 30.0,
) -> OpenAICompat:
    key = api_key if api_key is not None else os.environ.get("OPENAI_API_KEY")
    if not key:
        raise HunchError("Set OPENAI_API_KEY or pass api_key= to openai().")
    return OpenAICompat(
        api_key=key,
        model=model,
        base_url=base_url,
        extra_body=extra_body,
        extra_headers=extra_headers,
        timeout=timeout,
    )


def openrouter(
    api_key: str | None = None,
    *,
    model: str = "z-ai/glm-5.3-flash",
    timeout: float = 60.0,
) -> OpenAICompat:
    key = api_key if api_key is not None else os.environ.get("OPENROUTER_API_KEY")
    if not key:
        raise HunchError("Set OPENROUTER_API_KEY or pass api_key= to openrouter().")
    return OpenAICompat(
        api_key=key,
        model=model,
        base_url="https://openrouter.ai/api/v1",
        extra_headers={
            "HTTP-Referer": "https://github.com/hunch",
            "X-Title": "hunch",
        },
        timeout=timeout,
    )


def cerebras(
    api_key: str | None = None,
    *,
    model: str = "gpt-oss-120b",
    reasoning_effort: str = "low",
    timeout: float = 30.0,
) -> OpenAICompat:
    key = api_key if api_key is not None else os.environ.get("CEREBRAS_API_KEY")
    if not key:
        raise HunchError("Set CEREBRAS_API_KEY or pass api_key= to cerebras().")
    return OpenAICompat(
        api_key=key,
        model=model,
        base_url="https://api.cerebras.ai/v1",
        extra_body={"reasoning_effort": reasoning_effort},
        timeout=timeout,
    )
