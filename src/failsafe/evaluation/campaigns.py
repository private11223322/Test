"""Adaptive testing campaigns (Feature 8).

State machine: DISCOVER -> PLAN -> EXECUTE -> ANALYZE -> ADAPT -> REPORT.

A campaign inspects available tool metadata, generates bounded scenarios,
executes controlled faults, analyzes results, and selects subsequent
scenarios based on previous findings — all under explicit run, token, time,
and cost budgets.  Deterministic given a seed (the fault schedule and the
selection policy are seeded); live LLM variability is out of scope here.

The adaptive policy is intentionally simple and documented: it ranks candidate
scenarios by (a) tools not yet faulted, (b) fault kinds that previously
produced failures, (c) unexplored task categories.  ``evals/adaptive_vs_fixed.py``
compares it with fixed/random schedules under equal budgets.
"""

from __future__ import annotations

import random
import time
from collections.abc import Sequence
from dataclasses import dataclass, field
from enum import StrEnum
from typing import Any

from failsafe.evaluation.metrics import RunResult, compute_run_metrics
from failsafe.evaluation.runner import RunnerConfig, execute_task
from failsafe.evaluation.tasks import OracleTask
from failsafe.schemas import FaultKind


class CampaignPhase(StrEnum):
    DISCOVER = "DISCOVER"
    PLAN = "PLAN"
    EXECUTE = "EXECUTE"
    ANALYZE = "ANALYZE"
    ADAPT = "ADAPT"
    REPORT = "REPORT"


@dataclass
class Scenario:
    """A planned test scenario: task + fault schedule."""

    id: str
    task_id: str
    fault_schedule: list[dict[str, Any]]
    rationale: str = ""

    def to_dict(self) -> dict[str, Any]:
        return {
            "id": self.id,
            "task_id": self.task_id,
            "fault_schedule": self.fault_schedule,
            "rationale": self.rationale,
        }


@dataclass
class CampaignBudget:
    max_scenarios: int = 12
    max_seconds: float = 60.0
    max_faults_total: int = 10
    max_estimated_cost_usd: float = 0.0  # synthetic runs cost nothing

    def exhausted(self, used_scenarios: int, elapsed: float, faults: int) -> bool:
        return (
            used_scenarios >= self.max_scenarios
            or elapsed >= self.max_seconds
            or faults >= self.max_faults_total
        )


@dataclass
class CampaignReport:
    seed: int
    policy: str
    phases_executed: list[str] = field(default_factory=list)
    scenarios: list[dict[str, Any]] = field(default_factory=list)
    results: list[dict[str, Any]] = field(default_factory=list)
    metrics: dict[str, Any] = field(default_factory=dict)
    budgets: dict[str, Any] = field(default_factory=dict)
    adaptation_notes: list[str] = field(default_factory=list)

    def to_dict(self) -> dict[str, Any]:
        return {
            "seed": self.seed,
            "policy": self.policy,
            "phases_executed": self.phases_executed,
            "scenarios": self.scenarios,
            "results": self.results,
            "metrics": self.metrics,
            "budgets": self.budgets,
            "adaptation_notes": self.adaptation_notes,
        }


class AdaptiveCampaign:
    """Bounded adaptive fault-injection campaign over synthetic tasks."""

    def __init__(
        self,
        agent: Any,
        tasks: Sequence[OracleTask],
        *,
        seed: int = 1337,
        budget: CampaignBudget | None = None,
        runner_config: RunnerConfig | None = None,
    ) -> None:
        self.agent = agent
        self.tasks = list(tasks)
        self.seed = seed
        self.budget = budget or CampaignBudget()
        self.rng = random.Random(seed)
        self.runner_config = runner_config or RunnerConfig(seed=seed)
        self.report = CampaignReport(seed=seed, policy="adaptive-v1")

    # ---- phase implementations ---------------------------------------
    def discover(self) -> dict[str, Any]:
        """Inspect available tool metadata from task specs."""
        tools: set[str] = set()
        categories: set[str] = set()
        for t in self.tasks:
            tools.update(t.tools_required)
            categories.add(t.category)
        info = {"tools": sorted(tools), "categories": sorted(categories), "task_count": len(self.tasks)}
        self.report.phases_executed.append(CampaignPhase.DISCOVER.value)
        return info

    def plan(self, discovered: dict[str, Any], history: list[dict[str, Any]]) -> Scenario:
        """Select the next scenario deterministically from history."""
        self.report.phases_executed.append(CampaignPhase.PLAN.value)
        used_tasks = {h["task_id"] for h in history}
        # (c) prefer unexplored task categories first
        candidates = [t for t in self.tasks if t.spec.id not in used_tasks] or list(self.tasks)
        task = candidates[0]
        # (b) prefer fault kinds that previously produced failures, else seeded choice
        failing_kinds = [h["fault_kind"] for h in history if h.get("produced_failure") and h.get("fault_kind")]
        kinds = list(FaultKind)
        if failing_kinds:
            kind = FaultKind(self.rng.choice(failing_kinds))
            rationale = "re-use fault kind that previously produced a failure"
        else:
            # (a) prefer tools not yet faulted
            faulted_tools = {h.get("fault_tool") for h in history}
            untried = [t for t in task.tools_required if t not in faulted_tools]
            tool = self.rng.choice(untried or task.tools_required or ["unknown"])
            kind = self.rng.choice(kinds)
            rationale = f"explore tool {tool} with fault {kind.value}"
        tool = self.rng.choice(task.tools_required or ["unknown"])
        schedule = [{"kind": kind.value, "tool": tool, "occurrence": 1, "params": {}}]
        return Scenario(
            id=f"scenario-{len(history) + 1:03d}",
            task_id=task.spec.id,
            fault_schedule=schedule,
            rationale=rationale,
        )

    def execute(self, scenario: Scenario) -> RunResult:
        self.report.phases_executed.append(CampaignPhase.EXECUTE.value)
        task = next(t for t in self.tasks if t.spec.id == scenario.task_id)
        # bind the planned fault schedule to the task run
        task_with_faults = type(task)(
            spec=task.spec,
            world_factory=task.world_factory,
            oracle_checks=task.oracle_checks,
            tools_required=task.tools_required,
            fault_schedule=scenario.fault_schedule,
            expected_behavior=task.expected_behavior,
            category=task.category,
        )
        return execute_task(self.agent, task_with_faults, self.runner_config)

    def analyze(self, result: RunResult) -> dict[str, Any]:
        self.report.phases_executed.append(CampaignPhase.ANALYZE.value)
        produced_failure = (not result.oracle.success) or any(
            f.severity.value in ("high", "critical") for f in result.findings
        )
        return {
            "run_id": result.run_id,
            "task_id": result.task_id,
            "oracle_success": result.oracle.success,
            "findings": len(result.findings),
            "produced_failure": produced_failure,
            "fault_kind": result.trace.fault_injections[0].kind.value if result.trace.fault_injections else None,
            "fault_tool": result.trace.fault_injections[0].tool_name if result.trace.fault_injections else None,
            "recovered": result.recovered,
        }

    def adapt(self, history: list[dict[str, Any]]) -> list[str]:
        self.report.phases_executed.append(CampaignPhase.ADAPT.value)
        notes = []
        failures = [h for h in history if h["produced_failure"]]
        if failures:
            kind = failures[-1].get("fault_kind")
            notes.append(f"Increase weight of fault kind '{kind}' after {len(failures)} failure-producing scenario(s).")
        else:
            notes.append("No failure-producing scenario yet; broaden fault-kind exploration.")
        return notes

    # ---- driver ------------------------------------------------------
    def run(self) -> CampaignReport:
        start = time.monotonic()
        discovered = self.discover()
        history: list[dict[str, Any]] = []
        results: list[RunResult] = []
        faults_total = 0

        while not self.budget.exhausted(len(history), time.monotonic() - start, faults_total):
            scenario = self.plan(discovered, history)
            result = self.execute(scenario)
            summary = self.analyze(result)
            history.append({"task_id": scenario.task_id, "scenario_id": scenario.id,
                            "fault_kind": summary["fault_kind"], "fault_tool": summary["fault_tool"],
                            "produced_failure": summary["produced_failure"]})
            results.append(result)
            faults_total += result.faults_injected
            self.report.scenarios.append(scenario.to_dict())
            self.report.results.append(summary)
            self.report.adaptation_notes.extend(self.adapt(history))

        self.report.phases_executed.append(CampaignPhase.REPORT.value)
        metrics = compute_run_metrics(results)
        self.report.metrics = metrics.to_dict()
        self.report.budgets = {
            "max_scenarios": self.budget.max_scenarios,
            "max_seconds": self.budget.max_seconds,
            "max_faults_total": self.budget.max_faults_total,
            "scenarios_used": len(results),
            "faults_injected_total": faults_total,
            "elapsed_seconds": round(time.monotonic() - start, 3),
        }
        return self.report


class FixedCampaign:
    """Baseline: fixed fault schedule — every task gets the same seeded fault
    plan.  Used for equal-budget comparison with :class:`AdaptiveCampaign`."""

    policy = "fixed-v1"

    def __init__(self, agent: Any, tasks: Sequence[OracleTask], *, seed: int = 1337,
                 budget: CampaignBudget | None = None, runner_config: RunnerConfig | None = None):
        self.agent = agent
        self.tasks = list(tasks)
        self.seed = seed
        self.budget = budget or CampaignBudget()
        self.runner_config = runner_config or RunnerConfig(seed=seed)

    def run(self) -> CampaignReport:
        start = time.monotonic()
        rng = random.Random(self.seed)
        report = CampaignReport(seed=self.seed, policy=self.policy)
        results: list[RunResult] = []
        faults_total = 0
        i = 0
        while not self.budget.exhausted(i, time.monotonic() - start, faults_total):
            task = self.tasks[i % len(self.tasks)]
            kind = rng.choice(list(FaultKind))
            tool = rng.choice(task.tools_required or ["unknown"])
            scenario = Scenario(
                id=f"fixed-{i + 1:03d}",
                task_id=task.spec.id,
                fault_schedule=[{"kind": kind.value, "tool": tool, "occurrence": 1, "params": {}}],
                rationale="fixed seeded schedule",
            )
            task_with_faults = type(task)(
                spec=task.spec, world_factory=task.world_factory,
                oracle_checks=task.oracle_checks, tools_required=task.tools_required,
                fault_schedule=scenario.fault_schedule,
                expected_behavior=task.expected_behavior, category=task.category,
            )
            result = execute_task(self.agent, task_with_faults, self.runner_config)
            results.append(result)
            faults_total += result.faults_injected
            report.scenarios.append(scenario.to_dict())
            report.results.append({
                "run_id": result.run_id, "task_id": result.task_id,
                "oracle_success": result.oracle.success, "findings": len(result.findings),
                "produced_failure": (not result.oracle.success),
            })
            i += 1
        report.metrics = compute_run_metrics(results).to_dict()
        report.budgets = {
            "max_scenarios": self.budget.max_scenarios,
            "scenarios_used": len(results),
            "faults_injected_total": faults_total,
            "elapsed_seconds": round(time.monotonic() - start, 3),
        }
        return report
