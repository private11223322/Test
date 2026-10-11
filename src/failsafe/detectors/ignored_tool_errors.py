"""Detector C: ignored tool errors.

Identifies failed tool results without an appropriate response: no retry, no
alternative action, no escalation, and no explicit failure report to the
user, according to the task policy (``task.tool_policy`` or the configured
allow-list of response actions).
"""

from __future__ import annotations

from failsafe.detectors.base import Detector, registry
from failsafe.schemas import Evidence, Severity, StepKind, Trace


@registry.register
class IgnoredToolErrorsDetector(Detector):
    id = "ignored_tool_errors"
    category = "ignored_tool_errors"
    description = (
        "Flags failed tool results that were not followed by a retry, fallback, "
        "escalation, or explicit failure report within the remaining steps."
    )
    known_limitations = [
        "The detector uses heuristics to recognize a fallback or escalation; "
        "an unusual but valid recovery strategy may be flagged.",
        "Policy comes from task.tool_policy.require_failure_report / allow_actions "
        "when present, otherwise from detector configuration.",
    ]

    def detect(self, trace: Trace) -> list:
        policy = (trace.task.tool_policy if trace.task else {}) or {}
        allow_actions = policy.get("allow_actions") or self.config.ignored_error_allow_actions
        require_report = bool(policy.get("require_failure_report", self.config.require_failure_report))
        ordered = trace.ordered_steps()
        findings = []

        for idx, step in enumerate(ordered):
            tr = step.tool_result
            if tr is None or tr.success:
                continue
            # what happens after this failed tool result?
            subsequent = ordered[idx + 1 :]
            handled = False
            handle_kind: str | None = None
            handle_step_id: str | None = None

            for later in subsequent:
                # retry: another tool call to same tool
                if later.tool_call is not None and later.tool_call.name == (tr.name or ""):
                    if "retry" in allow_actions:
                        handled, handle_kind, handle_step_id = True, "retry", later.id
                        break
                # fallback: a different tool call after the failure
                if later.tool_call is not None and later.tool_call.name != (tr.name or ""):
                    if "fallback" in allow_actions:
                        handled, handle_kind, handle_step_id = True, "fallback", later.id
                        break
                # escalation / explicit failure report in a response
                if later.kind in (StepKind.RESPONSE, StepKind.ERROR):
                    text = str(later.output or later.error_message or "").lower()
                    failure_markers = (
                        "fail", "error", "unable", "cannot", "could not", "couldn't",
                        "unavailable", "issue", "problem", "sorry", "unable to",
                    )
                    if any(m in text for m in failure_markers):
                        if "report_failure" in allow_actions or "escalate" in allow_actions:
                            handled, handle_kind, handle_step_id = True, "report_failure", later.id
                            break
                if later.kind == StepKind.ERROR and "escalate" in allow_actions:
                    handled, handle_kind, handle_step_id = True, "escalate", later.id
                    break

            if not handled and require_report:
                findings.append(
                    self.make_finding(
                        trace,
                        summary=(
                            f"Tool error from '{tr.name or 'unknown'}' was not followed by a retry, "
                            f"fallback, escalation, or explicit failure report."
                        ),
                        evidence=[
                            Evidence(
                                kind="ignored_tool_error",
                                step_ids=[step.id],
                                detail={
                                    "tool": tr.name,
                                    "error_type": tr.error_type,
                                    "error_message": tr.error_message,
                                    "tool_call_id": tr.tool_call_id,
                                },
                                quote=(tr.error_message or "")[:200],
                            )
                        ],
                        severity=Severity.HIGH,
                        metrics={"require_failure_report": require_report},
                        description=self.description,
                    )
                )
            elif handled and handle_step_id is not None:
                # Not a failure: record nothing (handled correctly).
                _ = (handle_kind, handle_step_id)
        return findings
