"""Agentic components: regression-test generation and LangGraph investigation."""

from __future__ import annotations

from failsafe.agents.regression import (
    RegressionTestValidator,
    generate_from_findings,
    generate_regression_test,
    regression_reproduces_issue,
    task_from_regression_test,
    validate_test,
)

__all__ = [
    "RegressionTestValidator",
    "generate_regression_test",
    "generate_from_findings",
    "validate_test",
    "task_from_regression_test",
    "regression_reproduces_issue",
]


def __getattr__(name: str):
    # Lazy import so the deterministic core works without langgraph installed.
    if name in ("InvestigationAgent", "investigate_run", "InvestigationReport", "InvestigationState"):
        from failsafe.agents import investigation

        return getattr(investigation, name)
    if name == "InvestigationTools":
        from failsafe.agents import tools

        return tools.InvestigationTools
    raise AttributeError(name)
