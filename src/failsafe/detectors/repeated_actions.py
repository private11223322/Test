"""Detector A: repeated actions.

Detects repeated tool calls with normalized arguments and reports exact
step IDs.  A single duplicate is not a confirmed failure: the finding is
``RULE_TRIGGERED`` until ``repeat_min_count`` identical calls are observed
(within an optional window).
"""

from __future__ import annotations

import json
from collections import defaultdict
from typing import Any

from failsafe.detectors.base import Detector, registry
from failsafe.schemas import Evidence, Severity, Trace


def normalize_arguments(arguments: dict[str, Any], *, normalize: bool = True) -> str:
    """Canonical string for tool arguments.

    When ``normalize`` is true the keys are sorted and whitespace in string
    values is collapsed, so ``{"q": "hello  world"}`` matches
    ``{"q": "hello world"}``.
    """

    def _norm(v: Any) -> Any:
        if isinstance(v, str) and normalize:
            return " ".join(v.split())
        if isinstance(v, dict):
            return {k: _norm(v[k]) for k in sorted(v)}
        if isinstance(v, list):
            return [_norm(x) for x in v]
        return v

    if not normalize:
        return json.dumps(arguments, sort_keys=True, default=str)
    return json.dumps(_norm(arguments), sort_keys=True, default=str)


@registry.register
class RepeatedActionsDetector(Detector):
    id = "repeated_actions"
    category = "repeated_actions"
    description = (
        "Detects tool calls repeated with normalized identical arguments. "
        "Reports the exact step IDs of every repetition. A duplicate alone is "
        "not a confirmed failure; the finding escalates at the configured count."
    )
    known_limitations = [
        "Legitimate intentional retries with identical arguments may be flagged; "
        "read the evidence and the task policy before treating this as a failure.",
        "Arguments are compared by normalized JSON; semantically equivalent but "
        "structurally different arguments are not grouped.",
    ]

    def detect(self, trace: Trace) -> list:
        min_count = self.config.repeat_min_count
        window = self.config.repeat_window
        groups: dict[tuple[str, str], list] = defaultdict(list)
        ordered = trace.ordered_steps()

        # Map tool_call id -> whether its result was a failure (to recognize retries)
        call_failed: dict[str, bool] = {}
        for step in ordered:
            if step.tool_result is not None and step.tool_result.tool_call_id:
                call_failed[step.tool_result.tool_call_id] = not step.tool_result.success

        for idx, step in enumerate(ordered):
            if step.tool_call is None:
                continue
            key = (
                step.tool_call.name,
                normalize_arguments(step.tool_call.arguments, normalize=self.config.repeat_normalize_args),
            )
            groups[key].append((idx, step))

        findings = []
        for (name, args_key), occurrences in groups.items():
            # apply window constraint: only consecutive-within-window groups
            indexes = [i for i, _ in occurrences]
            if window is not None:
                # keep occurrences where max(indexes)-min(indexes) <= window
                if indexes[-1] - indexes[0] > window:
                    continue
            if len(occurrences) < 2:
                continue

            # A call that directly retries an identical FAILED call is a retry
            # chain (tracked by excessive_retries), not a repeated action.
            # Exclude those from the escalation count but keep them in evidence.
            is_retry: list[bool] = []
            prev_failed = False
            for _, s in occurrences:
                if prev_failed:
                    is_retry.append(True)
                else:
                    is_retry.append(False)
                prev_failed = call_failed.get(s.tool_call.id, False) if s.tool_call else False

            non_retry_count = sum(1 for r in is_retry if not r)
            step_ids = [s.id for _, s in occurrences]
            severity = Severity.INFO if non_retry_count < min_count else Severity.MEDIUM
            if non_retry_count >= max(min_count, 5):
                severity = Severity.HIGH
            note = ""
            if any(is_retry):
                note = " (includes retry-after-failure occurrences; see excessive_retries for budgets)"
            findings.append(
                self.make_finding(
                    trace,
                    summary=(
                        f"Tool '{name}' called {len(occurrences)} times with identical "
                        f"normalized arguments ({non_retry_count} non-retry)."
                    ),
                    evidence=[
                        Evidence(
                            kind="repeated_tool_calls",
                            step_ids=step_ids,
                            detail={
                                "tool": name,
                                "arguments_normalized": args_key,
                                "count": len(occurrences),
                                "non_retry_count": non_retry_count,
                                "occurrence_indexes": indexes,
                                "is_retry": is_retry,
                                "note": note.strip() or None,
                            },
                            quote=f"{name}({args_key})",
                        )
                    ],
                    severity=severity,
                    metrics={
                        "count": len(occurrences),
                        "non_retry_count": non_retry_count,
                        "min_count": min_count,
                    },
                    description=self.description,
                )
            )
        findings.sort(key=lambda f: f.summary)
        return findings
