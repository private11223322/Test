"""Detector D: premature termination.

Compares the execution with an explicit task specification or oracle: if the
run ends (final response or run end) while required outcomes from
``task.required_outcomes`` have no corresponding evidence, the termination
is premature.  Structural checks only; semantic judging is optional elsewhere.
"""

from __future__ import annotations

import json

from failsafe.detectors.base import Detector, registry
from failsafe.schemas import Evidence, Severity, StepKind, Trace


def _evidence_satisfied(trace: Trace, outcome: dict) -> tuple[bool, list[str]]:
    """Check a required outcome against trace structure.

    Supported outcome forms (deterministic):
      {"type": "tool_called", "tool": "db.query"}
      {"type": "tool_called_with", "tool": "db.query", "arguments_subset": {...}}
      {"type": "tool_success", "tool": "db.update"}
      {"type": "response_contains", "text": "..."}
      {"type": "step_count_min", "min": 2}
    """
    kind = outcome.get("type")
    step_ids: list[str] = []
    if kind == "tool_called":
        for s in trace.ordered_steps():
            if s.tool_call is not None and s.tool_call.name == outcome.get("tool"):
                step_ids.append(s.id)
        return (bool(step_ids), step_ids)
    if kind == "tool_called_with":
        subset = outcome.get("arguments_subset") or {}
        for s in trace.ordered_steps():
            if s.tool_call is None or s.tool_call.name != outcome.get("tool"):
                continue
            args = s.tool_call.arguments
            if all(json.dumps(args.get(k), sort_keys=True, default=str) == json.dumps(v, sort_keys=True, default=str)
                   for k, v in subset.items()):
                step_ids.append(s.id)
        return (bool(step_ids), step_ids)
    if kind == "tool_success":
        for s in trace.ordered_steps():
            if s.tool_result is not None and s.tool_result.name == outcome.get("tool") and s.tool_result.success:
                step_ids.append(s.id)
        return (bool(step_ids), step_ids)
    if kind == "response_contains":
        text = str(outcome.get("text", ""))
        for s in trace.ordered_steps():
            if s.kind == StepKind.RESPONSE and text.lower() in str(s.output or "").lower():
                step_ids.append(s.id)
        return (bool(step_ids), step_ids)
    if kind == "step_count_min":
        return (len(trace.steps) >= int(outcome.get("min", 1)), [s.id for s in trace.steps[:1]])
    # Unknown outcome type: cannot verify deterministically -> not satisfied
    return (False, [])


@registry.register
class PrematureTerminationDetector(Detector):
    id = "premature_termination"
    category = "premature_termination"
    description = (
        "Detects runs that terminate before every required task outcome has "
        "structural evidence in the trace. Requires an explicit task spec."
    )
    known_limitations = [
        "Only structural outcome types are checked deterministically; semantic "
        "completion requires an optional LLM judge.",
        "Without a task specification this detector produces no findings "
        "(configurable via premature_require_task_spec).",
    ]

    def detect(self, trace: Trace) -> list:
        task = trace.task
        if task is None or not task.required_outcomes:
            if self.config.premature_require_task_spec:
                return []
            return []

        findings = []
        unmet = []
        for outcome in task.required_outcomes:
            ok, ids = _evidence_satisfied(trace, outcome)
            if not ok:
                unmet.append((outcome, ids))

        if unmet:
            # the terminal step is evidence of where the run ended
            ordered = trace.ordered_steps()
            terminal_ids = [ordered[-1].id] if ordered else []
            for outcome, _ in unmet:
                findings.append(
                    self.make_finding(
                        trace,
                        summary=(
                            f"Run terminated before required outcome was satisfied: "
                            f"{json.dumps(outcome, sort_keys=True, default=str)[:120]}"
                        ),
                        evidence=[
                            Evidence(
                                kind="unmet_required_outcome",
                                step_ids=terminal_ids,
                                detail={"required_outcome": outcome, "task_id": task.id},
                                quote=str(outcome),
                            )
                        ],
                        severity=Severity.HIGH,
                        metrics={"unmet_count": len(unmet), "required_count": len(task.required_outcomes)},
                        description=self.description,
                    )
                )
        return findings
