"""Record-and-replay support for repeatable CI tests.

Recorded tool interactions (inputs -> outputs, including injected faults) are
stored as JSON "cassettes".  A replayer reproduces the recorded behavior
deterministically, keeping live model evaluations separate from deterministic
replay evaluations.
"""

from __future__ import annotations

import hashlib
import json
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any


def _key(tool_name: str, arguments: dict[str, Any]) -> str:
    payload = json.dumps({"tool": tool_name, "arguments": arguments}, sort_keys=True, default=str)
    return hashlib.sha256(payload.encode("utf-8")).hexdigest()[:24]


@dataclass
class RecordedCall:
    tool: str
    arguments: dict[str, Any]
    ok: bool
    value: Any = None
    error: str | None = None
    error_kind: str | None = None


@dataclass
class Cassette:
    name: str
    calls: list[RecordedCall] = field(default_factory=list)

    def save(self, path: str | Path) -> Path:
        p = Path(path)
        p.parent.mkdir(parents=True, exist_ok=True)
        payload = {
            "name": self.name,
            "calls": [
                {
                    "tool": c.tool,
                    "arguments": c.arguments,
                    "ok": c.ok,
                    "value": c.value,
                    "error": c.error,
                    "error_kind": c.error_kind,
                }
                for c in self.calls
            ],
        }
        p.write_text(json.dumps(payload, indent=2, default=str), encoding="utf-8")
        return p

    @staticmethod
    def load(path: str | Path) -> Cassette:
        payload = json.loads(Path(path).read_text(encoding="utf-8"))
        calls = [RecordedCall(**c) for c in payload.get("calls", [])]
        return Cassette(name=payload.get("name", Path(path).stem), calls=calls)


class Recorder:
    """Records tool calls into a cassette."""

    def __init__(self, name: str = "cassette") -> None:
        self.cassette = Cassette(name=name)

    def wrap(self, fn: Any, tool_name: str | None = None) -> Any:
        name = tool_name or fn.__name__
        recorder = self

        def wrapper(*args: Any, **kwargs: Any) -> Any:
            arguments = dict(kwargs)
            if args:
                arguments["_positional"] = list(args)
            try:
                value = fn(*args, **kwargs)
                recorder.cassette.calls.append(RecordedCall(tool=name, arguments=arguments, ok=True, value=value))
                return value
            except Exception as exc:  # noqa: BLE001 - record then re-raise
                recorder.cassette.calls.append(
                    RecordedCall(
                        tool=name, arguments=arguments, ok=False,
                        error=str(exc), error_kind=type(exc).__name__,
                    )
                )
                raise

        wrapper.__name__ = name
        return wrapper


class Replayer:
    """Replays recorded tool calls deterministically."""

    def __init__(self, cassette: Cassette) -> None:
        self.cassette = cassette
        self._index: dict[str, list[RecordedCall]] = {}
        for call in cassette.calls:
            self._index.setdefault(_key(call.tool, call.arguments), []).append(call)
        self._used: dict[str, int] = {}

    def wrap(self, fn: Any, tool_name: str | None = None) -> Any:
        name = tool_name or fn.__name__
        replayer = self

        def wrapper(*args: Any, **kwargs: Any) -> Any:
            arguments = dict(kwargs)
            if args:
                arguments["_positional"] = list(args)
            key = _key(name, arguments)
            queue = replayer._index.get(key, [])
            used = replayer._used.get(key, 0)
            if used >= len(queue):
                raise LookupError(f"No recorded call for tool={name} args={arguments!r}")
            replayer._used[key] = used + 1
            call = queue[used]
            if call.ok:
                return call.value
            raise RuntimeError(call.error or "recorded error")

        wrapper.__name__ = name
        return wrapper
