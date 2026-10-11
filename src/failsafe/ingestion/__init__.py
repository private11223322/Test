"""Secret redaction and size limits for untrusted trace data."""

from __future__ import annotations

import re
from typing import Any

SECRET_KEY_PATTERN = re.compile(
    r"(?i)(api[_-]?key|apikey|secret|token|password|passwd|authorization|auth|credential|private[_-]?key|access[_-]?key|session[_-]?id|cookie)"
)

SECRET_VALUE_PATTERNS: list[tuple[re.Pattern[str], str]] = [
    # OpenAI-style keys
    (re.compile(r"sk-[A-Za-z0-9_\-]{16,}"), "sk-***REDACTED***"),
    # Bearer tokens
    (re.compile(r"(?i)bearer\s+[A-Za-z0-9._\-]{8,}"), "Bearer ***REDACTED***"),
    # AWS access keys
    (re.compile(r"AKIA[0-9A-Z]{12,}"), "***REDACTED-AWS-KEY***"),
    # Generic key=value secrets
    (
        re.compile(r"(?i)\b(api[_-]?key|secret|token|password)\b\s*[=:]\s*['\"]?([^\s'\",}]{6,})"),
        r"\1=***REDACTED***",
    ),
    # JWT-like
    (re.compile(r"eyJ[A-Za-z0-9_\-]{10,}\.[A-Za-z0-9_\-]{10,}\.[A-Za-z0-9_\-]{5,}"), "***REDACTED-JWT***"),
]

REDACTED = "***REDACTED***"


def redact_text(text: str) -> str:
    if not text:
        return text
    out = text
    for pattern, repl in SECRET_VALUE_PATTERNS:
        out = pattern.sub(repl, out)
    return out


def redact_value(value: Any, enabled: bool = True) -> Any:
    """Recursively redact secret-looking keys and values."""
    if not enabled:
        return value
    if isinstance(value, str):
        return redact_text(value)
    if isinstance(value, dict):
        out: dict[str, Any] = {}
        for k, v in value.items():
            if isinstance(k, str) and SECRET_KEY_PATTERN.search(k):
                out[k] = REDACTED
            else:
                out[k] = redact_value(v, enabled=True)
        return out
    if isinstance(value, list):
        return [redact_value(v, enabled=True) for v in value]
    return value


class TraceSizeLimitExceeded(Exception):
    """Raised when a trace exceeds configured size limits."""


def enforce_limits(
    raw: dict[str, Any],
    max_bytes: int = 5_000_000,
    max_steps: int = 5_000,
) -> None:
    import json

    size = len(json.dumps(raw, default=str).encode("utf-8"))
    if size > max_bytes:
        raise TraceSizeLimitExceeded(
            f"Trace size {size} bytes exceeds limit of {max_bytes} bytes."
        )
    steps = raw.get("steps")
    if isinstance(steps, list) and len(steps) > max_steps:
        raise TraceSizeLimitExceeded(
            f"Trace has {len(steps)} steps, exceeding limit of {max_steps}."
        )
