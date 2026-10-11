"""Task runner: executes an agent against oracle tasks with fault injection.

An *agent* is any callable implementing :class:`Agent` — typically a sample
agent from ``examples/agents`` or a wrapper around a user's agent.  The
runner owns the world, tools, fault injection, recording, oracle checks, and
detector execution so every run is measured the same way.
"""

from __future__ import annotations

import time
from collections.abc import Callable, Sequence
from dataclasses import dataclass
from typing import Any, Protocol

from failsafe.config import DetectorConfig
from failsafe.detectors import run_detectors
from failsafe.evaluation.metrics import RunResult
from failsafe.evaluation.oracles import run_checks
from failsafe.evaluation.tasks import OracleTask
from failsafe.evaluation.world import SyntheticWorld
from failsafe.injection.faults import FaultSchedule
from failsafe.injection.injector import InjectionLog, ToolFaultInjector
from failsafe.recording import TraceRecorder
from failsafe.schemas import RunStatus, StepKind


class Agent(Protocol):
    """Protocol for agents that can be evaluated.

    The agent receives the task spec, a dict of tools (already wrapped with
    fault injection), and the trace recorder.  It returns its final response
    text.  Agents must not inspect the world directly — only the tools.
    """

    name: str

    def run(self, task: Any, tools: dict[str, Callable[..., Any]], recorder: TraceRecorder) -> str:  # pragma: no cover
        ...


@dataclass
class RunnerConfig:
    seed: int = 1337
    inject_faults: bool = True
    detector_config: DetectorConfig | None = None
    run_detectors: bool = True
    timeout_seconds: float = 60.0


def _default_reported_failure(text: str) -> bool:
    lowered = text.lower()
    markers = ("fail", "unable", "cannot", "could not", "couldn't", "error", "unable to", "not possible")
    return any(m in lowered for m in markers)


def execute_task(
    agent: Agent,
    task: OracleTask,
    config: RunnerConfig | None = None,
    *,
    run_id: str | None = None,
    world: SyntheticWorld | None = None,
) -> RunResult:
    """Execute one agent on one task and produce a fully measured result."""
    cfg = config or RunnerConfig()
    world = world or task.make_world()
    log = InjectionLog()
    schedule = FaultSchedule.from_dicts(task.fault_schedule, seed=cfg.seed) if task.fault_schedule else FaultSchedule(seed=cfg.seed, specs=[])
    injector = ToolFaultInjector(schedule, log=log)
    tools = world.tool_map(injector if cfg.inject_faults else None)

    recorder = TraceRecorder(
        agent_name=getattr(agent, "name", "agent"),
        agent_version=getattr(agent, "version", None),
        task=task.spec,
        run_id=run_id,
        seed=cfg.seed,
    )

    start = time.monotonic()
    error: str | None = None
    response_text = ""
    try:
        response_text = agent.run(task.spec, tools, recorder) or ""
    except Exception as exc:  # noqa: BLE001 - recorded as run error
        error = f"{type(exc).__name__}: {exc}"
        recorder.record_error(error)
        response_text = ""
    runtime_ms = (time.monotonic() - start) * 1000.0

    # record fault injection log into the trace
    for rec in log.records:
        rec.tool_name = rec.tool_name
        recorder.record_fault_injection(rec)

    # ensure the final response appears in the trace timeline
    has_response_step = any(s.kind == StepKind.RESPONSE for s in recorder.trace.steps)
    if response_text and not has_response_step:
        recorder.record_response(response_text)

    # finalize trace
    agent_reported_failure = _default_reported_failure(response_text)
    if error is not None:
        status = RunStatus.ERROR
    elif agent_reported_failure:
        status = RunStatus.FAILURE
    else:
        status = RunStatus.SUCCESS
    trace = recorder.finish(
        status=status,
        final_response=response_text,
        error=error,
    )

    # oracle verification (never trusts the response)
    oracle = run_checks(
        world,
        trace,
        task.oracle_checks,
        task_id=task.spec.id,
        task_name=task.spec.name,
    )

    findings = run_detectors(trace, cfg.detector_config) if cfg.run_detectors else []

    return RunResult(
        run_id=trace.metadata.run_id,
        task_id=task.spec.id,
        task_name=task.spec.name,
        trace=trace,
        oracle=oracle,
        findings=findings,
        faults_scheduled=len(schedule.specs),
        faults_injected=len(log.records),
        runtime_ms=runtime_ms,
        llm_usage=dict(trace.metadata.llm_usage),
        agent_reported_failure=agent_reported_failure,
        error=error,
    )


def execute_suite(
    agent: Agent,
    tasks: Sequence[OracleTask],
    config: RunnerConfig | None = None,
    *,
    seed: int | None = None,
) -> list[RunResult]:
    """Execute a suite of tasks with a fixed seed for reproducibility."""
    cfg = config or RunnerConfig()
    if seed is not None:
        cfg = RunnerConfig(
            seed=seed,
            inject_faults=cfg.inject_faults,
            detector_config=cfg.detector_config,
            run_detectors=cfg.run_detectors,
            timeout_seconds=cfg.timeout_seconds,
        )
    results = []
    for task in tasks:
        results.append(execute_task(agent, task, cfg))
    return results
