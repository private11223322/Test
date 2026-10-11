"""Optional LLM-based semantic evaluation (Layer B).

Everything in this package is optional: with no credentials the deterministic
core remains fully functional and judge calls are skipped explicitly.
"""

from __future__ import annotations

from failsafe.judges.llm_judge import JudgeError, JudgeResult, LLMJudge, human_review
from failsafe.judges.providers import (
    DisabledProvider,
    EchoProvider,
    LLMProvider,
    LLMProviderError,
    OpenAIProvider,
    get_provider,
)

__all__ = [
    "LLMJudge",
    "JudgeResult",
    "JudgeError",
    "human_review",
    "LLMProvider",
    "LLMProviderError",
    "EchoProvider",
    "OpenAIProvider",
    "DisabledProvider",
    "get_provider",
]
