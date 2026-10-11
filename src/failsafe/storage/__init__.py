"""Persistent storage for Failsafe (SQLite by default, PostgreSQL optional)."""

from __future__ import annotations

from failsafe.storage.models import (
    Base,
    EvaluationRecord,
    FaultRecord,
    FindingRecord,
    OracleRecord,
    RegressionTestRecord,
    ReportRecord,
    RunRecord,
    StepRecord,
    ToolCallRecord,
    make_engine,
    make_session_factory,
)
from failsafe.storage.repository import Repository

__all__ = [
    "Base",
    "Repository",
    "RunRecord",
    "StepRecord",
    "ToolCallRecord",
    "FaultRecord",
    "FindingRecord",
    "OracleRecord",
    "RegressionTestRecord",
    "EvaluationRecord",
    "ReportRecord",
    "make_engine",
    "make_session_factory",
]
