"""Detector F: lost or repeated context.

Detects exact repeated requests programmatically (identical LLM prompts or
identical retrieval queries issued more than once) and emits UNRESOLVED
findings when a later prompt omits information supplied earlier (potential
lost context) — the latter is only flagged when structural evidence is
strong, and supports optional semantic comparison via the LLM judge.
"""

from __future__ import annotations

import json
from collections import defaultdict

from failsafe.detectors.base import Detector, registry
from failsafe.schemas import Evidence, FindingStatus, Severity, StepKind, Trace


def _prompt_text(step) -> str:
    if isinstance(step.input, dict):
        for key in ("prompt", "messages", "input", "query", "text"):
            if key in step.input:
                val = step.input[key]
                return json.dumps(val, sort_keys=True, default=str) if not isinstance(val, str) else val
        return json.dumps(step.input, sort_keys=True, default=str)
    return str(step.input or "")


@registry.register
class LostContextDetector(Detector):
    id = "lost_context"
    category = "lost_context"
    description = (
        "Detects exactly repeated prompts/queries and structurally suspicious "
        "context loss (a later LLM prompt that re-asks for information already "
        "provided). Exact repeats are rule-triggered; suspected loss is "
        "UNRESOLVED and suitable for semantic judging."
    )
    known_limitations = [
        "Exact matching only for repeated requests; paraphrased repeats need a "
        "semantic judge.",
        "Context-loss heuristics can false-positive on legitimate summarization.",
    ]

    def detect(self, trace: Trace) -> list:
        findings = []
        ordered = trace.ordered_steps()
        min_count = self.config.context_repeat_min_count

        # 1) exact repeated prompts
        groups: dict[str, list] = defaultdict(list)
        for s in ordered:
            if s.kind in (StepKind.LLM, StepKind.RETRIEVAL):
                text = _prompt_text(s)
                if text.strip():
                    groups[text].append(s)
        for text, occurrences in groups.items():
            if len(occurrences) >= min_count:
                findings.append(
                    self.make_finding(
                        trace,
                        summary=(
                            f"Identical request issued {len(occurrences)} times "
                            f"(possible lost context or loop)."
                        ),
                        evidence=[
                            Evidence(
                                kind="repeated_request",
                                step_ids=[s.id for s in occurrences],
                                detail={"count": len(occurrences)},
                                quote=text[:200],
                            )
                        ],
                        severity=Severity.MEDIUM,
                        metrics={"count": len(occurrences)},
                        description=self.description,
                    )
                )

        # 2) suspicious question re-asking (structural): '?' questions in a later
        # prompt that were answered in an earlier tool result but appear again
        # verbatim in a later LLM prompt's user portion.
        llm_steps = [s for s in ordered if s.kind == StepKind.LLM]
        for i in range(1, len(llm_steps)):
            prev, cur = llm_steps[i - 1], llm_steps[i]
            prev_text = _prompt_text(prev)
            cur_text = _prompt_text(cur)
            # If the current prompt is a strict subset asking again for an entity
            # that appeared in prev prompt and prev output, flag as unresolved.
            if not cur_text.strip() or cur_text == prev_text:
                continue
            if "could you" in cur_text.lower() or "again" in cur_text.lower():
                if any(tok in prev_text for tok in cur_text.split()[:6]):
                    findings.append(
                        self.make_finding(
                            trace,
                            summary="Later prompt re-requests information supplied earlier (possible context loss).",
                            evidence=[
                                Evidence(
                                    kind="possible_context_loss",
                                    step_ids=[prev.id, cur.id],
                                    detail={},
                                    quote=cur_text[:200],
                                )
                            ],
                            severity=Severity.LOW,
                            status=FindingStatus.UNRESOLVED,
                            description=self.description,
                        )
                    )
        return findings
