"""Configurable LLM provider abstraction.

Providers are optional.  The deterministic core never calls them.  Three
implementations:

- ``EchoProvider``: deterministic local stub used for tests and offline demos.
  It returns a fixed structured answer and NEVER pretends to be a real model.
- ``OpenAIProvider``: uses the ``openai`` package when installed and
  ``OPENAI_API_KEY`` is set.
- ``DisabledProvider``: raises a clear error when an LLM is requested but not
  configured.

No API keys are stored in Failsafe config or code; they are read from the
environment only.
"""

from __future__ import annotations

import json
import os
from typing import Any, Protocol


class LLMProviderError(Exception):
    """Raised when an LLM call cannot be made (missing config, budget, etc.)."""


class LLMProvider(Protocol):
    name: str

    def complete(self, prompt: str, *, system: str | None = None,
                 response_schema: dict[str, Any] | None = None) -> str:  # pragma: no cover
        ...


class DisabledProvider:
    name = "disabled"

    def complete(self, prompt: str, *, system: str | None = None,
                 response_schema: dict[str, Any] | None = None) -> str:
        raise LLMProviderError(
            "LLM features are disabled. Set FAILSAFE_LLM_ENABLED=true and configure "
            "a provider (e.g. OPENAI_API_KEY) to use semantic evaluation. "
            "The deterministic core works without any LLM."
        )


class EchoProvider:
    """Deterministic offline provider for tests.

    It inspects the prompt for a JSON instruction block and returns a canned,
    schema-shaped response.  It is NOT a language model: it exists so the
    optional pipeline can be tested end-to-end without credentials, and its
    outputs must never be presented as model judgments.
    """

    name = "echo"

    def __init__(self, canned: dict[str, Any] | None = None):
        self.canned = canned or {}
        self.calls: list[dict[str, Any]] = []

    def complete(self, prompt: str, *, system: str | None = None,
                 response_schema: dict[str, Any] | None = None) -> str:
        self.calls.append({"prompt": prompt, "system": system})
        if self.canned:
            return json.dumps(self.canned)
        # default: suspicious-but-cautious structured answer citing nothing
        return json.dumps({
            "verdict": "uncertain",
            "self_reported_confidence": 0.0,
            "summary": "EchoProvider is a test stub; no semantic judgment was made.",
            "cited_step_ids": [],
            "alternative_explanations": ["This response came from a deterministic test stub."],
            "needs_human_review": True,
        })


class OpenAIProvider:
    """OpenAI chat-completions provider (requires ``openai`` and ``OPENAI_API_KEY``)."""

    name = "openai"

    def __init__(self, model: str = "gpt-4o-mini", api_key_env: str = "OPENAI_API_KEY"):
        self.model = model
        self.api_key_env = api_key_env
        self.calls: list[dict[str, Any]] = []
        self.total_tokens = 0
        self.estimated_cost_usd = 0.0

    def complete(self, prompt: str, *, system: str | None = None,
                 response_schema: dict[str, Any] | None = None) -> str:
        api_key = os.environ.get(self.api_key_env)
        if not api_key:
            raise LLMProviderError(
                f"No API key in environment variable {self.api_key_env}. "
                "Set it (never commit it) to enable LLM features."
            )
        try:
            from openai import OpenAI
        except ImportError as exc:
            raise LLMProviderError("The 'openai' package is not installed. pip install 'failsafe[llm]'.") from exc

        client = OpenAI(api_key=api_key)
        messages: list[dict[str, str]] = []
        if system:
            messages.append({"role": "system", "content": system})
        messages.append({"role": "user", "content": prompt})
        kwargs: dict[str, Any] = {"model": self.model, "messages": messages}
        if response_schema:
            kwargs["response_format"] = {"type": "json_object"}
        response = client.chat.completions.create(**kwargs)
        usage = getattr(response, "usage", None)
        if usage is not None:
            self.total_tokens += int(getattr(usage, "total_tokens", 0) or 0)
            # conservative public pricing estimate; overwritten by providers when known
            self.estimated_cost_usd += self.total_tokens * 0.000002
        self.calls.append({"model": self.model, "tokens": getattr(usage, "total_tokens", None)})
        return response.choices[0].message.content or ""


def get_provider(name: str = "echo", **kwargs: Any) -> Any:
    if name == "disabled":
        return DisabledProvider()
    if name == "echo":
        return EchoProvider(**kwargs)
    if name == "openai":
        return OpenAIProvider(**kwargs)
    raise LLMProviderError(f"Unknown LLM provider '{name}'. Known: disabled, echo, openai.")
