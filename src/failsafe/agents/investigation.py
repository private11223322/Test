"""LangGraph investigation agent (Layer C).

A real tool-using investigation workflow built on LangGraph's StateGraph with
explicit state, bounded loops, iteration limits, and cost limits.  The agent:

1. Receives a suspicious run.
2. Inspects relevant evidence (allowlisted tools only).
3. Retrieves task requirements.
4. Compares relevant executions where available.
5. Forms a failure hypothesis.
6. Considers alternative explanations.
7. Proposes a regression test.
8. Validates the test schema and oracle.
9. Executes the test ONLY when authorized and safe.
10. Records the actual result.
11. Produces an evidence-backed investigation report.

The investigation agent is never the sole authority for a release decision:
its report feeds the deterministic CI gate as advisory material only.

Planner policy: the graph structure and tool use are identical regardless of
planner; ``HeuristicPlanner`` is deterministic (used for tests and offline
operation) and ``LLMPlanner`` uses a configured provider when available.
"""

from __future__ import annotations

import json
import time
from dataclasses import dataclass
from typing import Any, Literal, TypedDict

from failsafe.agents.tools import InvestigationTools
from failsafe.config import BudgetConfig
from failsafe.schemas import DetectorFinding, RegressionTest, Trace

try:
    from langgraph.graph import END, StateGraph

    LANGGRAPH_AVAILABLE = True
except ImportError:  # pragma: no cover - exercised when the extra is absent
    LANGGRAPH_AVAILABLE = False

    class StateGraph:  # type: ignore[no-redef]
        def __init__(self, *args, **kwargs):
            raise RuntimeError("langgraph is not installed. pip install 'failsafe[agents]'.")

    END = "__end__"  # type: ignore[assignment]


class InvestigationState(TypedDict, total=False):
    """Explicit graph state — everything the workflow knows."""

    run_id: str
    # evidence
    trace_summary: dict[str, Any]
    tool_errors: dict[str, Any]
    task_spec: dict[str, Any]
    comparison: dict[str, Any]
    relevant_steps: dict[str, Any]
    findings_overview: dict[str, Any]
    # reasoning
    hypothesis: str
    alternatives: list[str]
    evidence_step_ids: list[str]
    evidence_notes: list[str]
    # regression test
    proposed_test: dict[str, Any]
    validation: dict[str, Any]
    replay_result: dict[str, Any]
    # control
    iteration: int
    tool_calls: int
    llm_calls: int
    elapsed_seconds: float
    status: Literal["running", "done", "budget_exhausted", "failed"]
    done: list[str]
    log: list[str]
    report: dict[str, Any]


# ---- planners -----------------------------------------------------------


class HeuristicPlanner:
    """Deterministic planner implementing the documented investigation order.

    This is not an LLM: it selects the next action from explicit state.  It
    exists so the workflow is testable offline and reproducible in CI.

    Progress is tracked via ``state["done"]`` (completed action names) so that
    empty tool results (e.g. no findings) do not cause the planner to repeat
    an action forever.
    """

    name = "heuristic"
    ORDER = [
        "inspect_trace", "findings_overview", "inspect_tool_errors", "read_task_spec",
        "compare_runs", "retrieve_steps", "form_hypothesis", "propose_test",
        "validate_test", "replay_or_skip", "report",
    ]

    def next_action(self, state: InvestigationState) -> str:
        done = state.get("done", [])
        for action in self.ORDER:
            if action not in done:
                return action
        return "report"


class LLMPlanner:
    """Planner backed by a configured LLM provider (optional).

    The provider proposes the next action from a compact state summary; the
    action is restricted to the same allowlisted set as the heuristic planner.
    """

    name = "llm"
    ALLOWED = [
        "inspect_trace", "findings_overview", "inspect_tool_errors", "read_task_spec",
        "compare_runs", "retrieve_steps", "form_hypothesis", "propose_test",
        "validate_test", "replay_or_skip", "report",
    ]

    def __init__(self, provider: Any, model_note: str = ""):
        self.provider = provider
        self.model_note = model_note

    def next_action(self, state: InvestigationState) -> str:
        summary = {
            "has_trace": bool(state.get("trace_summary")),
            "has_findings": bool(state.get("findings_overview")),
            "has_errors": bool(state.get("tool_errors")),
            "has_task": bool(state.get("task_spec")),
            "has_comparison": bool(state.get("comparison")),
            "has_steps": bool(state.get("relevant_steps")),
            "has_hypothesis": bool(state.get("hypothesis")),
            "has_test": bool(state.get("proposed_test")),
            "has_validation": bool(state.get("validation")),
            "has_replay": bool(state.get("replay_result")),
            "iteration": state.get("iteration"),
        }
        prompt = (
            "You are planning an investigation of an AI agent execution. "
            "Choose the next action from this allowlist only: "
            f"{self.ALLOWED}. State: {json.dumps(summary)}. "
            "Respond with JSON: {\"action\": \"...\"}"
        )
        try:
            raw = self.provider.complete(prompt, system="Choose one action. JSON only.")
            data = json.loads(raw)
            action = data.get("action")
            if action in self.ALLOWED:
                return action
        except Exception:  # noqa: BLE001 - fall back to the deterministic planner
            pass
        return HeuristicPlanner().next_action(state)


# ---- workflow -----------------------------------------------------------


@dataclass
class InvestigationReport:
    run_id: str
    hypothesis: str
    alternatives: list[str]
    evidence_step_ids: list[str]
    proposed_test: dict[str, Any] | None
    replay_result: dict[str, Any] | None
    status: str
    tool_log: list[dict[str, Any]]
    budgets: dict[str, Any]
    note: str = (
        "This investigation is advisory. Release decisions are made by the "
        "deterministic CI gate; the investigation agent never controls the gate."
    )

    def to_dict(self) -> dict[str, Any]:
        return {
            "run_id": self.run_id,
            "hypothesis": self.hypothesis,
            "alternatives": self.alternatives,
            "evidence_step_ids": self.evidence_step_ids,
            "proposed_test": self.proposed_test,
            "replay_result": self.replay_result,
            "status": self.status,
            "tool_log": self.tool_log,
            "budgets": self.budgets,
            "note": self.note,
        }


class InvestigationAgent:
    """LangGraph-based bounded investigation workflow."""

    def __init__(
        self,
        tools: InvestigationTools,
        planner: Any | None = None,
        budgets: BudgetConfig | None = None,
    ) -> None:
        self.tools = tools
        self.planner = planner or HeuristicPlanner()
        self.budgets = budgets or BudgetConfig()
        self._start = 0.0

    # -- node implementations ----------------------------------------
    def _budget_ok(self, state: InvestigationState) -> bool:
        return (
            state.get("iteration", 0) < self.budgets.max_iterations
            and state.get("tool_calls", 0) < self.budgets.max_tool_calls
            and state.get("llm_calls", 0) < self.budgets.max_llm_calls
            and (time.monotonic() - self._start) < self.budgets.max_seconds
        )

    def _bump(self, state: InvestigationState, tool: bool = False, llm: bool = False) -> None:
        state["iteration"] = state.get("iteration", 0) + 1
        if tool:
            state["tool_calls"] = state.get("tool_calls", 0) + 1
        if llm:
            state["llm_calls"] = state.get("llm_calls", 0) + 1
        state["elapsed_seconds"] = time.monotonic() - self._start
        state.setdefault("log", []).append(f"iter={state['iteration']}")

    def node_dispatch(self, state: InvestigationState) -> InvestigationState:
        """Planner-driven dispatch node: chooses and executes one action."""
        if not self._budget_ok(state):
            state["status"] = "budget_exhausted"
            return state
        action = self.planner.next_action(state)
        state.setdefault("log", []).append(f"action={action}")
        handler = getattr(self, f"_act_{action}", None)
        if handler is None:
            state["status"] = "failed"
            state.setdefault("log", []).append(f"unknown action {action}")
            return state
        handler(state)
        state.setdefault("done", []).append(action)
        self._bump(state, tool=action not in ("form_hypothesis", "report", "replay_or_skip"),
                   llm=self.planner.name == "llm" and action in ("form_hypothesis",))
        if action == "report":
            state["status"] = "done"
        return state

    def _act_inspect_trace(self, state: InvestigationState) -> None:
        state["trace_summary"] = self.tools.inspect_trace()

    def _act_findings_overview(self, state: InvestigationState) -> None:
        state["findings_overview"] = self.tools.findings_overview()

    def _act_inspect_tool_errors(self, state: InvestigationState) -> None:
        state["tool_errors"] = self.tools.inspect_tool_errors()

    def _act_read_task_spec(self, state: InvestigationState) -> None:
        state["task_spec"] = self.tools.read_task_spec()

    def _act_compare_runs(self, state: InvestigationState) -> None:
        state["comparison"] = self.tools.compare_runs()

    def _act_retrieve_steps(self, state: InvestigationState) -> None:
        step_ids: list[str] = []
        for f in self.tools.findings:
            step_ids.extend(f.step_ids)
        state["relevant_steps"] = self.tools.retrieve_steps(step_ids[: self.tools.max_steps_returned])

    def _act_form_hypothesis(self, state: InvestigationState) -> None:
        if self.planner.name == "llm":
            self._act_form_hypothesis_llm(state)
            return
        findings = self.tools.findings
        errors = state.get("tool_errors", {}).get("errors", [])
        if findings:
            top = max(findings, key=lambda f: ("critical", "high", "medium", "low", "info").index(f.severity.value) * -1)
            state["hypothesis"] = (
                f"The run exhibits '{top.category}' (finding {top.id}): {top.summary}"
            )
            state["evidence_step_ids"] = top.step_ids
            state["alternatives"] = [
                "The behavior may be an intentional retry strategy rather than a defect.",
                "The task specification may be ambiguous about required outcomes.",
            ]
        elif errors:
            e = errors[0]
            state["hypothesis"] = f"Unhandled tool error from '{e['tool']}' may have corrupted the outcome."
            state["evidence_step_ids"] = [e["step_id"]]
            state["alternatives"] = ["The error may have been externally injected for testing."]
        else:
            state["hypothesis"] = "No deterministic failure evidence found; the suspicion may be semantic."
            state["evidence_step_ids"] = []
            state["alternatives"] = ["The suspicion may be a false positive."]
        summary = self.tools.summarize_evidence(state["evidence_step_ids"], "hypothesis evidence")
        state.setdefault("evidence_notes", []).append(json.dumps(summary))

    def _act_form_hypothesis_llm(self, state: InvestigationState) -> None:
        prompt = (
            "Investigate this agent run. Trace: "
            + json.dumps(state.get("trace_summary", {}), default=str)
            + " Findings: " + json.dumps(state.get("findings_overview", {}), default=str)
            + " Errors: " + json.dumps(state.get("tool_errors", {}), default=str)
            + ' Respond JSON: {"hypothesis": str, "evidence_step_ids": [str], "alternatives": [str]}'
        )
        try:
            raw = self.planner.provider.complete(prompt, system="Evidence-based analysis. JSON only.")
            data = json.loads(raw)
            state["hypothesis"] = str(data.get("hypothesis", ""))[:500]
            state["evidence_step_ids"] = [str(s) for s in data.get("evidence_step_ids", [])][:20]
            state["alternatives"] = [str(a) for a in data.get("alternatives", [])][:5]
        except Exception:  # noqa: BLE001
            state["hypothesis"] = "LLM planner failed to produce a hypothesis; falling back to evidence review."
            state["alternatives"] = ["Planner output was malformed."]

    def _act_propose_test(self, state: InvestigationState) -> None:
        findings = self.tools.findings
        target = next((f for f in findings if f.id in (state.get("hypothesis") or "")), None)
        if target is None and findings:
            target = findings[0]
        if target is None:
            state["proposed_test"] = {}
            return
        result = self.tools.propose_regression_test(target.id)
        state["proposed_test"] = result.get("test", {}) if result.get("ok") else {}

    def _act_validate_test(self, state: InvestigationState) -> None:
        if not state.get("proposed_test"):
            state["validation"] = {"ok": False, "errors": ["no test proposed"]}
            return
        state["validation"] = self.tools.validate_test(state["proposed_test"])

    def _act_replay_or_skip(self, state: InvestigationState) -> None:
        if not state.get("proposed_test") or not state.get("validation", {}).get("ok"):
            state["replay_result"] = {"ok": False, "note": "No validated test to replay."}
            return
        state["replay_result"] = self.tools.request_sandbox_replay(state["proposed_test"])

    def _act_report(self, state: InvestigationState) -> None:
        test = state.get("proposed_test") or None
        if test:
            try:
                test = RegressionTest.model_validate(test).model_dump(mode="json")
            except Exception:  # noqa: BLE001
                pass
        report = InvestigationReport(
            run_id=self.tools.trace.metadata.run_id,
            hypothesis=state.get("hypothesis", ""),
            alternatives=state.get("alternatives", []),
            evidence_step_ids=state.get("evidence_step_ids", []),
            proposed_test=test,
            replay_result=state.get("replay_result"),
            status=state.get("status", "done"),
            tool_log=[{"tool": e.tool, "args": e.arguments, "result": e.result_summary} for e in self.tools.log],
            budgets={
                "max_iterations": self.budgets.max_iterations,
                "max_tool_calls": self.budgets.max_tool_calls,
                "max_seconds": self.budgets.max_seconds,
                "iterations_used": state.get("iteration", 0),
                "tool_calls_used": state.get("tool_calls", 0),
                "elapsed_seconds": round(state.get("elapsed_seconds", 0.0), 3),
            },
        )
        state["report"] = report.to_dict()

    # -- graph construction -------------------------------------------
    def build_graph(self) -> Any:
        if not LANGGRAPH_AVAILABLE:
            raise RuntimeError("langgraph is not installed. pip install 'failsafe[agents]'.")
        g = StateGraph(InvestigationState)
        g.add_node("dispatch", self.node_dispatch)

        def route(state: InvestigationState) -> str:
            if state.get("status") in ("done", "budget_exhausted", "failed"):
                return END
            return "dispatch"

        g.set_entry_point("dispatch")
        g.add_conditional_edges("dispatch", route, {"dispatch": "dispatch", END: END})
        return g.compile()

    def investigate(self) -> InvestigationReport:
        self._start = time.monotonic()
        state: InvestigationState = {
            "run_id": self.tools.trace.metadata.run_id,
            "iteration": 0,
            "tool_calls": 0,
            "llm_calls": 0,
            "elapsed_seconds": 0.0,
            "status": "running",
            "log": [],
        }
        app = self.build_graph()
        final_state = app.invoke(
            state,
            config={"recursion_limit": self.budgets.max_iterations * 3 + 10},
        )
        report_dict = final_state.get("report") or {}
        if not report_dict:
            # budget exhausted or failed before the report node: still record
            # what happened, including budgets actually used.
            self._act_report(final_state)
            report_dict = final_state.get("report") or {}
        return InvestigationReport(
            run_id=report_dict.get("run_id", state["run_id"]),
            hypothesis=report_dict.get("hypothesis", ""),
            alternatives=report_dict.get("alternatives", []),
            evidence_step_ids=report_dict.get("evidence_step_ids", []),
            proposed_test=report_dict.get("proposed_test"),
            replay_result=report_dict.get("replay_result"),
            status=final_state.get("status", "failed"),
            tool_log=report_dict.get("tool_log", []),
            budgets=report_dict.get("budgets", {}),
        )


def investigate_run(
    trace: Trace,
    findings: list[DetectorFinding],
    *,
    repository: Any = None,
    compare_trace: Trace | None = None,
    authorized_replay: bool = False,
    budgets: BudgetConfig | None = None,
    planner: Any | None = None,
) -> InvestigationReport:
    """Convenience entry point: investigate one suspicious run."""
    tools = InvestigationTools(
        trace=trace,
        findings=findings,
        repository=repository,
        compare_trace=compare_trace,
        authorized_replay=authorized_replay,
    )
    agent = InvestigationAgent(tools, planner=planner, budgets=budgets)
    return agent.investigate()
