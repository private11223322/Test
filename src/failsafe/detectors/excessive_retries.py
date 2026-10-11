"""Detector B: excessive retries.

Detects repeated failed operations beyond a configurable retry budget.
A "retry" is a tool call with normalized-identical arguments that follows a
failed result for a previous identical call.
"""

from __future__ import annotations

from collections import defaultdict

from failsafe.detectors.base import Detector, registry
from failsafe.detectors.repeated_actions import normalize_arguments
from failsafe.schemas import Evidence, Severity, Trace


@registry.register
class ExcessiveRetriesDetector(Detector):
    id = "excessive_retries"
    category = "excessive_retries"
    description = (
        "Detects repeated failed operations exceeding the configured retry budget. "
        "Evidence lists every step ID in the retry chain."
    )
    known_limitations = [
        "Distinguishes retries only by normalized-identical arguments; "
        "a retry with changed arguments is treated as a different operation.",
        "Does not model distributed tracing across multiple runs.",
    ]

    def detect(self, trace: Trace) -> list:
        budget = self.config.retry_budget
        # chains[key] = list of (step, was_failure)
        chains: dict[tuple[str, str], list] = defaultdict(list)
        ordered = trace.ordered_steps()
        result_by_call: dict[str, bool] = {}
        for step in ordered:
            if step.tool_result is not None and step.tool_result.tool_call_id:
                result_by_call[step.tool_result.tool_call_id] = step.tool_result.success

        for step in ordered:
            if step.tool_call is None:
                continue
            key = (
                step.tool_call.name,
                normalize_arguments(step.tool_call.arguments, normalize=self.config.repeat_normalize_args),
            )
            # the call's own failure is determined by its result
            failed = result_by_call.get(step.tool_call.id, False) is False and (
                step.tool_call.id in result_by_call
            )
            chains[key].append((step, failed))

        findings = []
        for (name, args_key), chain in chains.items():
            # count trailing failure sequences: consecutive failed identical calls
            run_failed_steps = []
            max_run = 0
            max_run_steps = []
            current_run: list = []
            for step, failed in chain:
                if failed:
                    current_run.append(step)
                    if len(current_run) > max_run:
                        max_run = len(current_run)
                        max_run_steps = list(current_run)
                else:
                    current_run = []
            run_failed_steps = max_run_steps
            if max_run > budget:
                findings.append(
                    self.make_finding(
                        trace,
                        summary=(
                            f"Tool '{name}' failed {max_run} consecutive times with identical "
                            f"arguments, exceeding retry budget {budget}."
                        ),
                        evidence=[
                            Evidence(
                                kind="retry_chain",
                                step_ids=[s.id for s in run_failed_steps],
                                detail={
                                    "tool": name,
                                    "arguments_normalized": args_key,
                                    "consecutive_failures": max_run,
                                    "retry_budget": budget,
                                },
                                quote=f"{name}({args_key}) failed x{max_run}",
                            )
                        ],
                        severity=Severity.HIGH if max_run >= budget + 2 else Severity.MEDIUM,
                        metrics={"consecutive_failures": max_run, "retry_budget": budget},
                        description=self.description,
                    )
                )
        findings.sort(key=lambda f: f.summary)
        return findings
