"""Detector E: unsupported success claims.

Deterministic where possible: a final response claiming success while the
trace contains no evidence for the task's required outcomes (or while tool
results contradict the claim) is flagged.  Ambiguous cases are marked for
optional semantic judging rather than asserted as failure.
"""

from __future__ import annotations

import re

from failsafe.detectors.base import Detector, registry
from failsafe.detectors.premature_termination import _evidence_satisfied
from failsafe.schemas import Evidence, FindingStatus, Severity, StepKind, Trace

SUCCESS_PATTERNS = re.compile(
    r"(?i)\b(done|complete[d]?|success(ful(ly)?)?|accomplished|finished|updated|created|deleted|added|resolved|all set|task (is )?complete)\b"
)

FAILURE_PATTERNS = re.compile(
    r"(?i)\b(fail(ed|ure|s)?|unable to|cannot|can not|could not|couldn't|not possible|"
    r"did not|didn't|does not|doesn't|no changes|left unchanged|not apply|not applied|"
    r"not complete|incomplete|blocked|denied|refused)\b"
)


def claims_success(text: str, explicit: bool | None = None) -> bool:
    """Whether a final response claims success.

    An explicit failure report overrides lexical success words (e.g. "could not
    be completed" is a failure report, not a success claim).  An explicit
    ``claimed_success`` metadata flag overrides both.
    """
    if explicit is True:
        return True
    if explicit is False:
        return False
    if FAILURE_PATTERNS.search(text or ""):
        return False
    return bool(SUCCESS_PATTERNS.search(text or ""))


@registry.register
class UnsupportedSuccessDetector(Detector):
    id = "unsupported_success"
    category = "unsupported_success"
    description = (
        "Flags final responses that claim success without supporting evidence: "
        "required outcomes unsatisfied, or failures on tools the claim depends on."
    )
    known_limitations = [
        "Success-claim detection is lexical; unusual phrasings may be missed.",
        "Ambiguous cases are emitted as SEMANTIC_SUSPECT / UNRESOLVED and must "
        "be confirmed by a human or an optional LLM judge; they are never "
        "treated as verified failures on their own.",
    ]

    def detect(self, trace: Trace) -> list:
        ordered = trace.ordered_steps()
        response_step = None
        for s in reversed(ordered):
            if s.kind == StepKind.RESPONSE:
                response_step = s
                break
        if response_step is None:
            return []
        text = str(response_step.output or "")
        claims = claims_success(text, response_step.metadata.get("claimed_success"))
        if not claims:
            return []

        findings = []
        task = trace.task
        unmet = []
        if task and task.required_outcomes:
            for outcome in task.required_outcomes:
                ok, _ = _evidence_satisfied(trace, outcome)
                if not ok:
                    unmet.append(outcome)

        failed_tools = []
        from failsafe.detectors.repeated_actions import normalize_arguments

        # join tool_call steps to their result steps via tool_call_id
        result_by_call: dict[str, object] = {}
        for s in ordered:
            if s.tool_result is not None and s.tool_result.tool_call_id:
                result_by_call[s.tool_result.tool_call_id] = s.tool_result
        call_by_id = {s.tool_call.id: s for s in ordered if s.tool_call is not None}

        for s in ordered:
            tr = s.tool_result
            if tr is None or tr.success:
                continue
            # failures later recovered by a successful retry with identical
            # arguments are not "unrecovered" and do not warrant a review finding
            recovered = False
            orig_call = call_by_id.get(tr.tool_call_id or "")
            orig_args = orig_call.tool_call.arguments if orig_call and orig_call.tool_call else {}
            for call_id, call_step in call_by_id.items():
                if call_id == tr.tool_call_id:
                    continue
                if call_step.tool_call.name != (tr.name or ""):
                    continue
                later_result = result_by_call.get(call_id)
                if later_result is None or not later_result.success:  # type: ignore[union-attr]
                    continue
                if normalize_arguments(call_step.tool_call.arguments) == normalize_arguments(orig_args):
                    recovered = True
                    break
            if not recovered:
                failed_tools.append(s)

        if unmet:
            findings.append(
                self.make_finding(
                    trace,
                    summary=(
                        "Final response claims success but required task outcomes have no evidence."
                    ),
                    evidence=[
                        Evidence(
                            kind="success_claim_without_evidence",
                            step_ids=[response_step.id],
                            detail={"unmet_outcomes": unmet, "claimed_text": text[:300]},
                            quote=text[:200],
                        )
                    ],
                    severity=Severity.HIGH,
                    status=FindingStatus.RULE_TRIGGERED,
                    metrics={"unmet_count": len(unmet)},
                    description=self.description,
                )
            )
        elif failed_tools:
            # ambiguous: failures happened, but outcomes might still be met
            findings.append(
                self.make_finding(
                    trace,
                    summary=(
                        "Final response claims success after tool failures; "
                        "needs human or semantic review."
                    ),
                    evidence=[
                        Evidence(
                            kind="success_claim_after_errors",
                            step_ids=[response_step.id] + [s.id for s in failed_tools],
                            detail={"failed_tools": [s.tool_result.name for s in failed_tools]},
                            quote=text[:200],
                        )
                    ],
                    severity=Severity.LOW,
                    status=FindingStatus.UNRESOLVED,
                    metrics={"failed_tool_count": len(failed_tools)},
                    description=self.description,
                )
            )
        return findings
