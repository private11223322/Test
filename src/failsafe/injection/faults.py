"""Fault primitives for controlled tool-fault injection."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

from failsafe.schemas import FaultKind


class InjectedFault(Exception):
    """Exception raised by the injector to simulate a tool failure."""

    def __init__(self, kind: FaultKind, message: str, metadata: dict[str, Any] | None = None):
        super().__init__(message)
        self.kind = kind
        self.metadata = metadata or {}


@dataclass
class FaultSpec:
    """Specification of one fault to inject at a scheduled occurrence."""

    kind: FaultKind
    tool: str | None = None  # None = any tool
    occurrence: int = 1  # inject on the Nth matching call (1-based)
    params: dict[str, Any] = field(default_factory=dict)

    def matches(self, tool_name: str, call_index: int) -> bool:
        if self.tool is not None and self.tool != tool_name:
            return False
        return call_index == self.occurrence


@dataclass
class FaultSchedule:
    """A deterministic, seeded schedule of faults.

    A fixed seed reproduces the *schedule* (which faults are injected where).
    It does NOT guarantee identical live LLM behaviour.
    """

    seed: int
    specs: list[FaultSpec] = field(default_factory=list)
    max_injections: int = 5

    @staticmethod
    def from_dicts(items: list[dict[str, Any]], seed: int = 1337, max_injections: int = 5) -> FaultSchedule:
        specs = []
        for item in items:
            specs.append(
                FaultSpec(
                    kind=FaultKind(item["kind"]),
                    tool=item.get("tool"),
                    occurrence=int(item.get("occurrence", 1)),
                    params=dict(item.get("params", {})),
                )
            )
        return FaultSchedule(seed=seed, specs=specs, max_injections=max_injections)

    @staticmethod
    def generate(
        tools: list[str],
        seed: int = 1337,
        count: int = 3,
        kinds: list[FaultKind] | None = None,
        max_injections: int = 5,
    ) -> FaultSchedule:
        """Generate a reproducible random schedule over the given tools."""
        import random

        rng = random.Random(seed)
        kinds = kinds or list(FaultKind)
        specs: list[FaultSpec] = []
        if not tools:
            return FaultSchedule(seed=seed, specs=[], max_injections=max_injections)
        for _ in range(min(count, max_injections)):
            tool = rng.choice(tools)
            kind = rng.choice(kinds)
            occurrence = rng.randint(1, 3)
            params: dict[str, Any] = {}
            if kind == FaultKind.RATE_LIMIT:
                params["retry_after"] = rng.choice([0.1, 0.5, 1.0, 2.0])
            if kind == FaultKind.STALE_RESPONSE:
                params["age_seconds"] = rng.choice([60, 3600, 86400])
            specs.append(FaultSpec(kind=kind, tool=tool, occurrence=occurrence, params=params))
        return FaultSchedule(seed=seed, specs=specs, max_injections=max_injections)

    def to_json(self) -> list[dict[str, Any]]:
        return [
            {"kind": s.kind.value, "tool": s.tool, "occurrence": s.occurrence, "params": s.params}
            for s in self.specs
        ]
