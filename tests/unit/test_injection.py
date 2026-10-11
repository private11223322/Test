"""Unit tests for fault injection and record-and-replay."""

from __future__ import annotations

import pytest

from failsafe.injection import (
    Cassette,
    FaultSchedule,
    FaultSpec,
    InjectedFault,
    Recorder,
    Replayer,
    ToolFaultInjector,
    wrap_soft,
)
from failsafe.schemas import FaultKind


def _tool(x: int) -> str:
    return f"ok-{x}"


def test_schedule_from_dicts_deterministic():
    specs = [{"kind": "timeout", "tool": "db_get", "occurrence": 1}]
    s1 = FaultSchedule.from_dicts(specs, seed=42)
    s2 = FaultSchedule.from_dicts(specs, seed=42)
    assert s1.to_json() == s2.to_json()


def test_generate_schedule_reproducible():
    a = FaultSchedule.generate(["db_get", "search"], seed=7, count=3)
    b = FaultSchedule.generate(["db_get", "search"], seed=7, count=3)
    assert a.to_json() == b.to_json()
    c = FaultSchedule.generate(["db_get", "search"], seed=8, count=3)
    assert c.to_json() != a.to_json()  # different seed, different schedule


def test_generate_schedule_empty_tools():
    s = FaultSchedule.generate([], seed=1, count=3)
    assert s.specs == []


def test_injector_injects_on_schedule():
    schedule = FaultSchedule.from_dicts([{"kind": "error", "tool": "t", "occurrence": 2, "params": {"message": "boom"}}])
    injector = ToolFaultInjector(schedule)
    wrapped = injector.wrap(_tool, "t")
    assert wrapped(1) == "ok-1"
    with pytest.raises(InjectedFault) as exc:
        wrapped(2)
    assert exc.value.kind == FaultKind.ERROR
    assert wrapped(3) == "ok-3"
    assert injector.injection_count == 1
    assert len(injector.log.records) == 1
    assert injector.log.records[0].tool_name == "t"


def test_injector_respects_max_injections():
    schedule = FaultSchedule.from_dicts(
        [{"kind": "error", "tool": "t", "occurrence": n} for n in (1, 2, 3, 4)],
        max_injections=2,
    )
    injector = ToolFaultInjector(schedule)
    wrapped = injector.wrap(_tool, "t")
    errors = 0
    for i in range(4):
        try:
            wrapped(i)
        except InjectedFault:
            errors += 1
    assert errors == 2
    assert injector.injection_count == 2


def test_injector_only_matching_tool():
    schedule = FaultSchedule.from_dicts([{"kind": "timeout", "tool": "other", "occurrence": 1}])
    injector = ToolFaultInjector(schedule)
    wrapped = injector.wrap(_tool, "t")
    assert wrapped(1) == "ok-1"  # no injection for wrong tool


def test_all_fault_kinds_build_errors():
    for kind in FaultKind:
        spec = FaultSpec(kind=kind, params={"retry_after": 0.5, "age_seconds": 60})
        err = ToolFaultInjector.build_error(spec)
        assert isinstance(err, InjectedFault)
        assert err.kind == kind


def test_wrap_soft_returns_result():
    schedule = FaultSchedule.from_dicts([{"kind": "error", "tool": "t", "occurrence": 2}])
    injector = ToolFaultInjector(schedule)
    wrapped = wrap_soft(_tool, injector, "t")
    r1 = wrapped(1)
    assert r1.ok and r1.value == "ok-1"
    r2 = wrapped(2)
    assert not r2.ok and r2.fault is not None


def test_injection_log_records_seed():
    schedule = FaultSchedule.from_dicts([{"kind": "rate_limit", "tool": "t", "occurrence": 1}], seed=99)
    injector = ToolFaultInjector(schedule)
    wrapped = injector.wrap(_tool, "t")
    with pytest.raises(InjectedFault):
        wrapped(1)
    rec = injector.log.records[0]
    assert rec.seed == 99
    assert rec.triggered is True
    assert rec.kind == FaultKind.RATE_LIMIT


def test_cassette_record_and_replay(tmp_path):
    recorder = Recorder("demo")
    calls = []

    def tool(x: int) -> str:
        calls.append(x)
        return f"val-{x}"

    wrapped = recorder.wrap(tool, "mytool")
    wrapped(1)
    wrapped(2)
    path = recorder.cassette.save(tmp_path / "cassette.json")
    assert path.is_file()

    cassette = Cassette.load(path)
    replayer = Replayer(cassette)
    replayed = replayer.wrap(tool, "mytool")
    assert replayed(1) == "val-1"
    assert replayed(2) == "val-2"
    with pytest.raises(LookupError):
        replayed(3)


def test_replayer_replays_errors():
    recorder = Recorder()

    def tool() -> str:
        raise RuntimeError("recorded-boom")

    wrapped = recorder.wrap(tool, "t")
    with pytest.raises(RuntimeError):
        wrapped()
    replayer = Replayer(recorder.cassette)
    replayed = replayer.wrap(tool, "t")
    with pytest.raises(RuntimeError) as exc:
        replayed()
    assert "recorded-boom" in str(exc.value)


def test_record_and_replay_deterministic(tmp_path):
    def run_once() -> list:
        recorder = Recorder()

        def tool(x: int) -> str:
            return f"v{x * 2}"

        wrapped = recorder.wrap(tool, "t")
        return [wrapped(i) for i in range(3)], [c.value for c in recorder.cassette.calls]

    (out1, rec1) = run_once()
    (out2, rec2) = run_once()
    assert out1 == out2
    assert rec1 == rec2
