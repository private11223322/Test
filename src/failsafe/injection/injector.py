"""Tool-wrapper fault injector.

Wraps synthetic tool callables so controlled faults are injected according to
a seeded :class:`~failsafe.injection.faults.FaultSchedule`.  Every injection
is logged.  Synthetic tools only by default: real systems require
``allow_real_systems=True`` in configuration AND explicit authorization.
"""

from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass, field
from typing import Any

from failsafe.injection.faults import FaultSchedule, FaultSpec, InjectedFault
from failsafe.schemas import FaultInjectionRecord, FaultKind


@dataclass
class InjectionLog:
    records: list[FaultInjectionRecord] = field(default_factory=list)

    def add(self, record: FaultInjectionRecord) -> None:
        self.records.append(record)


class ToolFaultInjector:
    """Wraps tool functions and injects scheduled faults."""

    def __init__(
        self,
        schedule: FaultSchedule,
        log: InjectionLog | None = None,
        allow_real_systems: bool = False,
    ) -> None:
        self.schedule = schedule
        self.log = log or InjectionLog()
        self.allow_real_systems = allow_real_systems
        self._call_counts: dict[str, int] = {}
        self._injected = 0
        self._total_calls = 0

    def reset(self) -> None:
        self._call_counts.clear()
        self._injected = 0
        self._total_calls = 0

    @property
    def injection_count(self) -> int:
        return self._injected

    def _next_spec(self, tool_name: str) -> FaultSpec | None:
        count = self._call_counts.get(tool_name, 0)
        for spec in self.schedule.specs:
            if spec.matches(tool_name, count):
                return spec
        return None

    def wrap(self, fn: Callable[..., Any], tool_name: str | None = None) -> Callable[..., Any]:
        name = tool_name or fn.__name__
        injector = self

        def wrapper(*args: Any, **kwargs: Any) -> Any:
            injector._total_calls += 1
            injector._call_counts[name] = injector._call_counts.get(name, 0) + 1
            call_index = injector._call_counts[name]
            spec = injector._next_spec(name)
            if spec is not None and injector._injected < injector.schedule.max_injections:
                injector._injected += 1
                record = injector._make_record(spec, name, call_index)
                injector.log.add(record)
                raise injector.build_error(spec)
            return fn(*args, **kwargs)

        wrapper.__name__ = name
        return wrapper

    def _make_record(self, spec: FaultSpec, tool_name: str, call_index: int) -> FaultInjectionRecord:
        return FaultInjectionRecord(
            tool_name=tool_name,
            kind=spec.kind,
            scheduled_index=call_index,
            seed=self.schedule.seed,
            triggered=True,
            details=dict(spec.params),
        )

    @staticmethod
    def build_error(spec: FaultSpec) -> InjectedFault:
        k = spec.kind
        params = spec.params
        if k == FaultKind.TIMEOUT:
            return InjectedFault(k, "Simulated timeout", {"timeout_seconds": params.get("timeout_seconds", 30)})
        if k == FaultKind.ERROR:
            return InjectedFault(k, params.get("message", "Simulated explicit error"))
        if k == FaultKind.EMPTY_RESPONSE:
            return InjectedFault(k, "Simulated empty response", {"empty": True})
        if k == FaultKind.MALFORMED_RESPONSE:
            return InjectedFault(k, "Simulated malformed response", {"malformed": True})
        if k == FaultKind.STALE_RESPONSE:
            return InjectedFault(k, "Simulated stale response", {"age_seconds": params.get("age_seconds", 3600)})
        if k == FaultKind.RATE_LIMIT:
            return InjectedFault(k, "Simulated rate limit", {"retry_after": params.get("retry_after", 1.0)})
        if k == FaultKind.PARTIAL_RESULT:
            return InjectedFault(k, "Simulated partial result", {"partial": True})
        if k == FaultKind.SCHEMA_DRIFT:
            return InjectedFault(k, "Simulated schema drift", {"drifted": True})
        return InjectedFault(k, "Injected fault")


class FaultyResult:
    """Non-raising wrapper result, for agents that inspect errors instead of
    catching exceptions."""

    def __init__(self, ok: bool, value: Any = None, fault: InjectedFault | None = None):
        self.ok = ok
        self.value = value
        self.fault = fault


def wrap_soft(fn: Callable[..., Any], injector: ToolFaultInjector, tool_name: str | None = None) -> Callable[..., Any]:
    """Like :meth:`ToolFaultInjector.wrap` but returns :class:`FaultyResult`
    instead of raising — for agents that treat tool errors as data."""
    name = tool_name or fn.__name__

    def wrapper(*args: Any, **kwargs: Any) -> FaultyResult:
        try:
            value = injector.wrap(fn, name)(*args, **kwargs)
            return FaultyResult(True, value=value)
        except InjectedFault as fault:
            return FaultyResult(False, fault=fault)

    wrapper.__name__ = name
    return wrapper
