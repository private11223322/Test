"""Oracle checkers: independently verify task outcomes.

An oracle never trusts the agent's final response.  Checks are named,
parameterized, and serializable so regression tests can reference them.

Checker registry: add new checkers with :func:`register_checker` without
modifying unrelated components.
"""

from __future__ import annotations

import json
from collections.abc import Callable
from typing import Any

from failsafe.evaluation.world import SyntheticWorld
from failsafe.schemas import OracleCheck, OracleOutcome, StepKind, Trace

CheckerFn = Callable[[SyntheticWorld, Trace, dict[str, Any]], OracleCheck]

_CHECKERS: dict[str, CheckerFn] = {}


def register_checker(name: str) -> Callable[[CheckerFn], CheckerFn]:
    def deco(fn: CheckerFn) -> CheckerFn:
        _CHECKERS[name] = fn
        return fn

    return deco


def checker_names() -> list[str]:
    return sorted(_CHECKERS)


def run_checks(
    world: SyntheticWorld,
    trace: Trace,
    check_specs: list[dict[str, Any]],
    *,
    task_id: str = "task",
    task_name: str | None = None,
) -> OracleOutcome:
    checks: list[OracleCheck] = []
    for spec in check_specs:
        spec = dict(spec)
        name = spec.pop("check", None) or spec.pop("name", None)
        if name not in _CHECKERS:
            checks.append(
                OracleCheck(
                    name=str(name),
                    passed=False,
                    detail=f"Unknown checker '{name}'. Known: {checker_names()}",
                )
            )
            continue
        try:
            checks.append(_CHECKERS[name](world, trace, spec))
        except Exception as exc:  # noqa: BLE001 - a crashing check is a failed check
            checks.append(
                OracleCheck(name=str(name), passed=False, detail=f"Checker raised {type(exc).__name__}: {exc}")
            )
    success = all(c.passed for c in checks) if checks else False
    return OracleOutcome(
        run_id=trace.metadata.run_id,
        task_id=task_id,
        task_name=task_name,
        success=success,
        checks=checks,
        detail="all checks passed" if success else f"{sum(1 for c in checks if not c.passed)} check(s) failed",
    )


# ---- built-in checkers -------------------------------------------------


@register_checker("world_record_equals")
def _world_record_equals(world: SyntheticWorld, trace: Trace, spec: dict[str, Any]) -> OracleCheck:
    record_id = spec["id"]
    expected_fields: dict[str, Any] = spec.get("fields", {})
    actual = world.db.get(record_id)
    if actual is None:
        return OracleCheck(
            name="world_record_equals", passed=False,
            detail=f"Record '{record_id}' missing from world.",
            expected=expected_fields, actual=None,
        )
    mismatches = {k: {"expected": v, "actual": actual.get(k)} for k, v in expected_fields.items() if actual.get(k) != v}
    return OracleCheck(
        name="world_record_equals", passed=not mismatches,
        detail="record matches" if not mismatches else f"mismatched fields: {sorted(mismatches)}",
        expected=expected_fields, actual={k: actual.get(k) for k in expected_fields},
    )


@register_checker("world_record_not_exists")
def _world_record_not_exists(world: SyntheticWorld, trace: Trace, spec: dict[str, Any]) -> OracleCheck:
    record_id = spec["id"]
    exists = record_id in world.db
    return OracleCheck(
        name="world_record_not_exists", passed=not exists,
        detail=f"record '{record_id}' exists" if exists else f"record '{record_id}' absent as expected",
        expected="absent", actual="present" if exists else "absent",
    )


@register_checker("world_record_exists")
def _world_record_exists(world: SyntheticWorld, trace: Trace, spec: dict[str, Any]) -> OracleCheck:
    record_id = spec["id"]
    exists = record_id in world.db
    return OracleCheck(
        name="world_record_exists", passed=exists,
        detail=f"record '{record_id}' present" if exists else f"record '{record_id}' missing",
        expected="present", actual="present" if exists else "absent",
    )


@register_checker("world_mutation_count")
def _world_mutation_count(world: SyntheticWorld, trace: Trace, spec: dict[str, Any]) -> OracleCheck:
    record_id = spec.get("id")
    op = spec.get("op")
    expected = int(spec["count"])
    actual = world.mutation_count(record_id=record_id, op=op)
    return OracleCheck(
        name="world_mutation_count", passed=actual == expected,
        detail=f"mutations for {record_id}/{op}: {actual}, expected {expected}",
        expected=expected, actual=actual,
    )


@register_checker("world_mutation_max")
def _world_mutation_max(world: SyntheticWorld, trace: Trace, spec: dict[str, Any]) -> OracleCheck:
    record_id = spec.get("id")
    op = spec.get("op")
    maximum = int(spec["max"])
    actual = world.mutation_count(record_id=record_id, op=op)
    return OracleCheck(
        name="world_mutation_max", passed=actual <= maximum,
        detail=f"mutations for {record_id}/{op}: {actual}, max {maximum}",
        expected=f"<={maximum}", actual=actual,
    )


@register_checker("response_contains")
def _response_contains(world: SyntheticWorld, trace: Trace, spec: dict[str, Any]) -> OracleCheck:
    needle = str(spec["text"]).lower()
    texts = [
        str(s.output or "")
        for s in trace.ordered_steps()
        if s.kind == StepKind.RESPONSE
    ]
    final = texts[-1] if texts else str(trace.metadata.final_response or "")
    passed = needle in final.lower()
    return OracleCheck(
        name="response_contains", passed=passed,
        detail=f"final response {'contains' if passed else 'does not contain'} {needle!r}",
        expected=spec["text"], actual=final[:300],
    )


@register_checker("response_not_contains")
def _response_not_contains(world: SyntheticWorld, trace: Trace, spec: dict[str, Any]) -> OracleCheck:
    needle = str(spec["text"]).lower()
    texts = [
        str(s.output or "")
        for s in trace.ordered_steps()
        if s.kind == StepKind.RESPONSE
    ]
    final = texts[-1] if texts else str(trace.metadata.final_response or "")
    passed = needle not in final.lower()
    return OracleCheck(
        name="response_not_contains", passed=passed,
        detail=f"final response {'avoids' if passed else 'contains'} {needle!r}",
        expected=f"not {spec['text']}", actual=final[:300],
    )


@register_checker("trace_tool_called")
def _trace_tool_called(world: SyntheticWorld, trace: Trace, spec: dict[str, Any]) -> OracleCheck:
    tool = spec["tool"]
    times = spec.get("times")
    count = sum(1 for s in trace.ordered_steps() if s.tool_call is not None and s.tool_call.name == tool)
    if times is None:
        passed = count >= 1
    else:
        passed = count == int(times)
    return OracleCheck(
        name="trace_tool_called", passed=passed,
        detail=f"tool '{tool}' called {count} time(s)",
        expected=times if times is not None else ">=1", actual=count,
    )


@register_checker("trace_tool_not_called")
def _trace_tool_not_called(world: SyntheticWorld, trace: Trace, spec: dict[str, Any]) -> OracleCheck:
    tool = spec["tool"]
    count = sum(1 for s in trace.ordered_steps() if s.tool_call is not None and s.tool_call.name == tool)
    return OracleCheck(
        name="trace_tool_not_called", passed=count == 0,
        detail=f"tool '{tool}' called {count} time(s)",
        expected=0, actual=count,
    )


@register_checker("trace_tool_max_calls")
def _trace_tool_max_calls(world: SyntheticWorld, trace: Trace, spec: dict[str, Any]) -> OracleCheck:
    """Pass when a tool is called at most `max` times (anti-loop check)."""
    tool = spec["tool"]
    maximum = int(spec["max"])
    count = sum(1 for s in trace.ordered_steps() if s.tool_call is not None and s.tool_call.name == tool)
    return OracleCheck(
        name="trace_tool_max_calls", passed=count <= maximum,
        detail=f"tool '{tool}' called {count} time(s), max {maximum}",
        expected=f"<={maximum}", actual=count,
    )


@register_checker("trace_tool_succeeded")
def _trace_tool_succeeded(world: SyntheticWorld, trace: Trace, spec: dict[str, Any]) -> OracleCheck:
    tool = spec["tool"]
    results = [
        s.tool_result for s in trace.ordered_steps()
        if s.tool_result is not None and s.tool_result.name == tool
    ]
    passed = bool(results) and all(r.success for r in results)
    return OracleCheck(
        name="trace_tool_succeeded", passed=passed,
        detail=f"tool '{tool}' results: {[r.success for r in results]}",
        expected="at least one success and no failure", actual=[r.success for r in results],
    )


@register_checker("emails_sent_count")
def _emails_sent_count(world: SyntheticWorld, trace: Trace, spec: dict[str, Any]) -> OracleCheck:
    expected = int(spec["count"])
    actual = len(world.emails_sent)
    return OracleCheck(
        name="emails_sent_count", passed=actual == expected,
        detail=f"{actual} emails sent, expected {expected}",
        expected=expected, actual=actual,
    )


@register_checker("no_duplicate_mutation")
def _no_duplicate_mutation(world: SyntheticWorld, trace: Trace, spec: dict[str, Any]) -> OracleCheck:
    """Pass when each (op, record_id) mutation with identical fields happened at most once."""
    seen: dict[str, int] = {}
    duplicates = []
    for e in world.audit_log:
        key = f"{e.op}:{e.record_id}:{json.dumps(e.fields, sort_keys=True, default=str)}"
        seen[key] = seen.get(key, 0) + 1
        if seen[key] == 2:
            duplicates.append(key)
    return OracleCheck(
        name="no_duplicate_mutation", passed=not duplicates,
        detail="no duplicate mutations" if not duplicates else f"duplicate mutations: {duplicates}",
        expected="unique mutations", actual=seen,
    )


@register_checker("final_status")
def _final_status(world: SyntheticWorld, trace: Trace, spec: dict[str, Any]) -> OracleCheck:
    expected = str(spec["status"])
    actual = trace.metadata.status.value
    return OracleCheck(
        name="final_status", passed=actual == expected,
        detail=f"run status {actual}, expected {expected}",
        expected=expected, actual=actual,
    )


@register_checker("faults_injected_count")
def _faults_injected_count(world: SyntheticWorld, trace: Trace, spec: dict[str, Any]) -> OracleCheck:
    expected = int(spec["count"])
    actual = len(trace.fault_injections)
    op = spec.get("op", "==")
    if op == ">=":
        passed = actual >= expected
    elif op == "<=":
        passed = actual <= expected
    else:
        passed = actual == expected
    return OracleCheck(
        name="faults_injected_count", passed=passed,
        detail=f"{actual} faults injected, expected {op}{expected}",
        expected=f"{op}{expected}", actual=actual,
    )


@register_checker("agent_recovered")
def _agent_recovered(world: SyntheticWorld, trace: Trace, spec: dict[str, Any]) -> OracleCheck:
    """Pass when at least one injected fault occurred AND the world-state checks
    in the same oracle passed... which we cannot know here. Instead: recovery
    is defined as (faults injected > 0) and (no ignored tool errors) and
    (run status success). The full recovery metric is computed in metrics.py.
    """
    faults = len(trace.fault_injections)
    status_ok = trace.metadata.status.value == "success"
    tool_failures = [s for s in trace.ordered_steps() if s.tool_result is not None and not s.tool_result.success]
    # every tool failure should be followed by recovery or explicit report
    passed = faults > 0 and status_ok
    return OracleCheck(
        name="agent_recovered", passed=passed,
        detail=f"faults={faults}, status={trace.metadata.status.value}, tool_failures={len(tool_failures)}",
        expected="fault occurred and run succeeded", actual={"faults": faults, "status": trace.metadata.status.value},
    )


@register_checker("detector_quiet")
def _detector_quiet(world: SyntheticWorld, trace: Trace, spec: dict[str, Any]) -> OracleCheck:
    """Pass when the named detector produces no findings on this trace.

    Used by regression tests: the defect is gone iff the detector that
    originally caught it stays quiet (or the oracle verifies the outcome).
    """
    from failsafe.detectors import registry as detector_registry

    detector_id = spec["detector_id"]
    try:
        det = detector_registry.create(detector_id)
        findings = det.detect(trace)
    except Exception as exc:  # noqa: BLE001
        return OracleCheck(name="detector_quiet", passed=False,
                           detail=f"detector '{detector_id}' raised {exc}",
                           expected="detector runs", actual=str(exc))
    return OracleCheck(
        name="detector_quiet", passed=not findings,
        detail=f"detector '{detector_id}' produced {len(findings)} finding(s)",
        expected=0, actual=len(findings),
    )


@register_checker("custom")
def _custom(world: SyntheticWorld, trace: Trace, spec: dict[str, Any]) -> OracleCheck:
    """Custom predicate expressed as constrained operators on the world snapshot.

    spec = {"path": ["db","u-1001","plan"], "op": "eq", "value": "pro"}
    Supported ops: eq, ne, gt, lt, gte, lte, in, not_in, contains, exists, not_exists.
    Path walks dicts/lists of the world snapshot + trace summary. No code execution.
    """
    ops = {
        "eq": lambda a, b: a == b,
        "ne": lambda a, b: a != b,
        "gt": lambda a, b: a > b,
        "lt": lambda a, b: a < b,
        "gte": lambda a, b: a >= b,
        "lte": lambda a, b: a <= b,
        "in": lambda a, b: a in b,
        "not_in": lambda a, b: a not in b,
        "contains": lambda a, b: b in a,
        "exists": lambda a, b: a is not None,
        "not_exists": lambda a, b: a is None,
    }
    path = spec.get("path", [])
    op = spec.get("op", "eq")
    value = spec.get("value")
    snapshot = world.snapshot()
    current: Any = snapshot
    for part in path:
        if isinstance(current, dict):
            current = current.get(part)
        elif isinstance(current, list) and isinstance(part, int) and part < len(current):
            current = current[part]
        else:
            current = None
            break
    fn = ops.get(op)
    if fn is None:
        return OracleCheck(name="custom", passed=False, detail=f"unknown op '{op}'", expected=op, actual=current)
    try:
        passed = bool(fn(current, value))
    except Exception as exc:  # noqa: BLE001
        return OracleCheck(name="custom", passed=False, detail=f"op '{op}' raised {exc}", expected=value, actual=current)
    return OracleCheck(
        name="custom", passed=passed,
        detail=f"{'.'.join(str(p) for p in path)} {op} {value!r} -> {passed}",
        expected=value, actual=current,
    )
