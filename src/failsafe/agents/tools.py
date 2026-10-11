"""Allowlisted tools for the investigation agent.

All tools are pure-Python, read-mostly, and operate only on Failsafe's own
data structures and controlled runners.  There is no shell access, no
arbitrary code execution, no filesystem writes outside the repository layer,
and no network access.  Every tool call is logged.
"""

from __future__ import annotations

import json
from collections.abc import Callable
from dataclasses import dataclass, field
from typing import Any

from failsafe.agents.regression import (
    generate_regression_test,
    validate_test,
)
from failsafe.schemas import DetectorFinding, RegressionTest, Trace


@dataclass
class ToolLogEntry:
    tool: str
    arguments: dict[str, Any]
    result_summary: str


@dataclass
class InvestigationTools:
    """The tool surface exposed to the investigation agent."""

    trace: Trace
    findings: list[DetectorFinding]
    repository: Any = None
    compare_trace: Trace | None = None
    authorized_replay: bool = False
    max_steps_returned: int = 20
    log: list[ToolLogEntry] = field(default_factory=list)

    # ---- tools ------------------------------------------------------
    def inspect_trace(self) -> dict[str, Any]:
        """Summarize the run under investigation."""
        self._log("inspect_trace", {}, f"{len(self.trace.steps)} steps")
        return {
            "run_id": self.trace.metadata.run_id,
            "agent": self.trace.metadata.agent_name,
            "status": self.trace.metadata.status.value,
            "task": self.trace.metadata.task_name,
            "step_count": len(self.trace.steps),
            "fault_injection_count": len(self.trace.fault_injections),
            "final_response": (self.trace.metadata.final_response or "")[:500],
        }

    def retrieve_steps(self, step_ids: list[str]) -> dict[str, Any]:
        """Retrieve specific steps by id (evidence)."""
        out = []
        for sid in step_ids:
            s = self.trace.step_by_id(sid)
            if s is None:
                out.append({"id": sid, "found": False})
                continue
            out.append({
                "id": s.id,
                "found": True,
                "kind": s.kind.value,
                "name": s.name,
                "status": s.status.value,
                "input": s.input,
                "output": s.output,
                "error": s.error_message,
                "tool_call": s.tool_call.model_dump(mode="json") if s.tool_call else None,
                "tool_result": s.tool_result.model_dump(mode="json") if s.tool_result else None,
            })
        self._log("retrieve_steps", {"step_ids": step_ids}, f"{len(out)} steps")
        return {"steps": out}

    def inspect_tool_errors(self) -> dict[str, Any]:
        """List every failed tool result with its step id."""
        errors = []
        for s in self.trace.ordered_steps():
            tr = s.tool_result
            if tr is not None and not tr.success:
                errors.append({
                    "step_id": s.id,
                    "tool": tr.name,
                    "error_type": tr.error_type,
                    "error_message": tr.error_message,
                    "tool_call_id": tr.tool_call_id,
                })
        self._log("inspect_tool_errors", {}, f"{len(errors)} errors")
        return {"errors": errors}

    def compare_runs(self) -> dict[str, Any]:
        """Compare the investigated run with a reference run when available."""
        if self.compare_trace is None:
            self._log("compare_runs", {}, "no reference run")
            return {"available": False, "note": "No comparison run is available."}
        a, b = self.trace, self.compare_trace
        a_tools = [tc.name for tc in a.tool_calls()]
        b_tools = [tc.name for tc in b.tool_calls()]
        self._log("compare_runs", {}, f"compared {a.metadata.run_id} vs {b.metadata.run_id}")
        return {
            "available": True,
            "run_a": a.metadata.run_id,
            "run_b": b.metadata.run_id,
            "tool_sequence_a": a_tools,
            "tool_sequence_b": b_tools,
            "status_a": a.metadata.status.value,
            "status_b": b.metadata.status.value,
            "steps_a": len(a.steps),
            "steps_b": len(b.steps),
        }

    def read_task_spec(self) -> dict[str, Any]:
        """Read the task requirements for the run."""
        task = self.trace.task
        self._log("read_task_spec", {}, "task" if task else "none")
        if task is None:
            return {"available": False}
        return {
            "available": True,
            "id": task.id,
            "name": task.name,
            "instruction": task.instruction,
            "required_outcomes": task.required_outcomes,
            "tool_policy": task.tool_policy,
        }

    def search_regression_tests(self, query: str) -> dict[str, Any]:
        """Search existing regression tests (repository when attached)."""
        results = []
        if self.repository is not None:
            for rec in self.repository.list_regression_tests():
                text = f"{rec.scenario} {json.dumps(rec.payload, default=str)}".lower()
                if query.lower() in text:
                    results.append({"id": rec.id, "state": rec.state, "scenario": rec.scenario})
        self._log("search_regression_tests", {"query": query}, f"{len(results)} hits")
        return {"matches": results[:10]}

    def propose_regression_test(self, finding_id: str) -> dict[str, Any]:
        """Propose a regression test for a specific finding (candidate state)."""
        finding = next((f for f in self.findings if f.id == finding_id), None)
        if finding is None:
            self._log("propose_regression_test", {"finding_id": finding_id}, "finding not found")
            return {"ok": False, "error": f"unknown finding id {finding_id}"}
        test = generate_regression_test(self.trace, finding)
        self._log("propose_regression_test", {"finding_id": finding_id}, f"test {test.id}")
        return {"ok": True, "test": test.model_dump(mode="json")}

    def validate_test(self, test_json: dict[str, Any]) -> dict[str, Any]:
        """Validate a proposed test schema and oracle definition."""
        try:
            test = RegressionTest.model_validate(test_json)
        except Exception as exc:  # noqa: BLE001
            self._log("validate_test", {}, "invalid")
            return {"ok": False, "errors": [f"schema error: {exc}"]}
        validated, errors = validate_test(test)
        self._log("validate_test", {}, f"errors={len(errors)}")
        return {"ok": not errors, "errors": errors, "test": validated.model_dump(mode="json")}

    def request_sandbox_replay(self, test_json: dict[str, Any]) -> dict[str, Any]:
        """Execute a validated test ONLY when replay is authorized and safe.

        Safety: the test runs against the local synthetic world through the
        controlled runner — never against real systems, and never executing
        arbitrary code.
        """
        if not self.authorized_replay:
            self._log("request_sandbox_replay", {}, "denied: not authorized")
            return {
                "ok": False,
                "authorized": False,
                "note": "Sandbox replay is not authorized in this investigation. "
                        "Set authorized_replay=True explicitly to allow it.",
            }
        try:
            from failsafe.evaluation.runner import RunnerConfig, execute_task
            from failsafe.evaluation.tasks import OracleTask

            test = RegressionTest.model_validate(test_json)
            task = OracleTask(
                spec=test.task,
                world_factory=self._world_factory(),
                oracle_checks=list(test.oracle.get("checks", [])),
                tools_required=list(test.tools_required),
                fault_schedule=list(test.fault_schedule),
            )
            from failsafe.agents.sample import GoodAgent

            res = execute_task(GoodAgent(), task, RunnerConfig(seed=1337))
            self._log("request_sandbox_replay", {}, f"oracle={res.oracle.success}")
            return {
                "ok": True,
                "authorized": True,
                "run_id": res.run_id,
                "oracle_success": res.oracle.success,
                "checks": [c.model_dump(mode="json") for c in res.oracle.checks],
                "note": "Replay executed against the local synthetic world only.",
            }
        except Exception as exc:  # noqa: BLE001
            self._log("request_sandbox_replay", {}, "error")
            return {"ok": False, "error": f"{type(exc).__name__}: {exc}"}

    def summarize_evidence(self, step_ids: list[str], note: str = "") -> dict[str, Any]:
        """Attach a structured evidence summary to the investigation record."""
        found = [sid for sid in step_ids if self.trace.step_by_id(sid) is not None]
        self._log("summarize_evidence", {"step_ids": step_ids, "note": note}, f"{len(found)} valid")
        return {"valid_step_ids": found, "invalid_step_ids": [s for s in step_ids if s not in found], "note": note}

    def findings_overview(self) -> dict[str, Any]:
        """List the deterministic findings for this run."""
        self._log("findings_overview", {}, f"{len(self.findings)} findings")
        return {
            "findings": [
                {"id": f.id, "category": f.category, "severity": f.severity.value,
                 "status": f.status.value, "summary": f.summary, "step_ids": f.step_ids}
                for f in self.findings
            ]
        }

    # ---- helpers ----------------------------------------------------
    def _world_factory(self) -> Callable[[], Any]:
        from failsafe.evaluation.world import make_demo_world

        return make_demo_world

    def _log(self, tool: str, arguments: dict[str, Any], result_summary: str) -> None:
        self.log.append(ToolLogEntry(tool=tool, arguments=arguments, result_summary=result_summary))

    def as_map(self) -> dict[str, Callable[..., Any]]:
        return {
            "inspect_trace": self.inspect_trace,
            "retrieve_steps": self.retrieve_steps,
            "inspect_tool_errors": self.inspect_tool_errors,
            "compare_runs": self.compare_runs,
            "read_task_spec": self.read_task_spec,
            "search_regression_tests": self.search_regression_tests,
            "propose_regression_test": self.propose_regression_test,
            "validate_test": self.validate_test,
            "request_sandbox_replay": self.request_sandbox_replay,
            "summarize_evidence": self.summarize_evidence,
            "findings_overview": self.findings_overview,
        }
