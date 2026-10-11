"""Core trace schemas for Failsafe.

The normalized internal trace model.  Every external trace format is
converted to these models during ingestion.  Missing information is kept as
``None`` and never invented.
"""

from __future__ import annotations

import uuid
from datetime import UTC, datetime
from enum import StrEnum
from typing import Any

from pydantic import BaseModel, ConfigDict, Field, field_validator


def _new_id(prefix: str) -> str:
    return f"{prefix}-{uuid.uuid4().hex[:12]}"


def utcnow() -> datetime:
    return datetime.now(UTC)


class StepKind(StrEnum):
    """What a step represents in an execution."""

    LLM = "llm"
    TOOL_CALL = "tool_call"
    TOOL_RESULT = "tool_result"
    RETRIEVAL = "retrieval"
    PLANNING = "planning"
    RESPONSE = "response"
    ERROR = "error"
    NOTE = "note"
    OTHER = "other"


class StepStatus(StrEnum):
    OK = "ok"
    ERROR = "error"
    TIMEOUT = "timeout"
    UNKNOWN = "unknown"


class RunStatus(StrEnum):
    SUCCESS = "success"
    FAILURE = "failure"
    ERROR = "error"
    UNKNOWN = "unknown"
    RUNNING = "running"


class Severity(StrEnum):
    INFO = "info"
    LOW = "low"
    MEDIUM = "medium"
    HIGH = "high"
    CRITICAL = "critical"


class FindingStatus(StrEnum):
    """How much a finding has been verified.

    VERIFIED        - confirmed against evidence / oracle.
    RULE_TRIGGERED  - deterministic rule fired; context may be needed.
    SEMANTIC_SUSPECT - LLM judge raised a suspicion; unverified.
    UNRESOLVED      - ambiguous; needs human review.
    DISMISSED       - reviewed and rejected.
    CONFIRMED_BY_HUMAN - a human confirmed the finding.
    REJECTED_BY_HUMAN - a human rejected the finding.
    """

    VERIFIED = "verified"
    RULE_TRIGGERED = "rule_triggered"
    SEMANTIC_SUSPECT = "semantic_suspect"
    UNRESOLVED = "unresolved"
    DISMISSED = "dismissed"
    CONFIRMED_BY_HUMAN = "confirmed_by_human"
    REJECTED_BY_HUMAN = "rejected_by_human"


class FaultKind(StrEnum):
    TIMEOUT = "timeout"
    ERROR = "error"
    EMPTY_RESPONSE = "empty_response"
    MALFORMED_RESPONSE = "malformed_response"
    STALE_RESPONSE = "stale_response"
    RATE_LIMIT = "rate_limit"
    PARTIAL_RESULT = "partial_result"
    SCHEMA_DRIFT = "schema_drift"


class StrictModel(BaseModel):
    """Base model: forbid unknown keys on ingestion of external data."""

    model_config = ConfigDict(extra="forbid", validate_assignment=True, use_enum_values=False)


class LooseModel(BaseModel):
    """Base model that tolerates extra keys (forward compatibility)."""

    model_config = ConfigDict(extra="allow", validate_assignment=True, use_enum_values=False)


class ToolCall(StrictModel):
    id: str = Field(default_factory=lambda: _new_id("tc"))
    name: str
    arguments: dict[str, Any] = Field(default_factory=dict)
    step_id: str | None = None


class ToolResult(StrictModel):
    id: str = Field(default_factory=lambda: _new_id("tr"))
    tool_call_id: str | None = None
    name: str | None = None
    success: bool = True
    output: Any = None
    error_type: str | None = None
    error_message: str | None = None
    duration_ms: float | None = None
    step_id: str | None = None
    metadata: dict[str, Any] = Field(default_factory=dict)

    @field_validator("output", mode="before")
    @classmethod
    def _keep_missing_as_none(cls, v: Any) -> Any:
        return v


class FaultInjectionRecord(StrictModel):
    id: str = Field(default_factory=lambda: _new_id("fi"))
    run_id: str | None = None
    step_id: str | None = None
    tool_call_id: str | None = None
    tool_name: str | None = None
    kind: FaultKind
    scheduled_index: int | None = None
    seed: int | None = None
    triggered: bool = True
    injected_at: datetime = Field(default_factory=utcnow)
    details: dict[str, Any] = Field(default_factory=dict)


class Step(StrictModel):
    id: str = Field(default_factory=lambda: _new_id("step"))
    kind: StepKind = StepKind.OTHER
    name: str | None = None
    status: StepStatus = StepStatus.OK
    parent_id: str | None = None
    timestamp: datetime = Field(default_factory=utcnow)
    end_timestamp: datetime | None = None
    input: Any = None
    output: Any = None
    tool_call: ToolCall | None = None
    tool_result: ToolResult | None = None
    error_type: str | None = None
    error_message: str | None = None
    metadata: dict[str, Any] = Field(default_factory=dict)

    @property
    def duration_ms(self) -> float | None:
        if self.end_timestamp is None:
            return None
        return (self.end_timestamp - self.timestamp).total_seconds() * 1000.0


class TaskSpec(StrictModel):
    """Explicit task specification used for oracle checks and premature
    termination detection.  ``required_outcomes`` are deterministic checks;
    ``description`` is free text for semantic judging."""

    id: str = Field(default_factory=lambda: _new_id("task"))
    name: str
    description: str = ""
    instruction: str = ""
    required_outcomes: list[dict[str, Any]] = Field(default_factory=list)
    forbidden_outcomes: list[dict[str, Any]] = Field(default_factory=list)
    tool_policy: dict[str, Any] = Field(default_factory=dict)
    metadata: dict[str, Any] = Field(default_factory=dict)


class RunMetadata(LooseModel):
    run_id: str = Field(default_factory=lambda: _new_id("run"))
    agent_name: str = "unknown-agent"
    agent_version: str | None = None
    task_id: str | None = None
    task_name: str | None = None
    status: RunStatus = RunStatus.UNKNOWN
    started_at: datetime = Field(default_factory=utcnow)
    ended_at: datetime | None = None
    seed: int | None = None
    environment: dict[str, Any] = Field(default_factory=dict)
    llm_usage: dict[str, Any] = Field(default_factory=dict)
    labels: dict[str, str] = Field(default_factory=dict)
    final_response: str | None = None
    error: str | None = None


class Trace(StrictModel):
    """A complete normalized execution trace."""

    id: str = Field(default_factory=lambda: _new_id("trace"))
    metadata: RunMetadata = Field(default_factory=RunMetadata)
    task: TaskSpec | None = None
    steps: list[Step] = Field(default_factory=list)
    fault_injections: list[FaultInjectionRecord] = Field(default_factory=list)
    source_format: str = "failsafe"
    schema_version: str = "1.0"

    def step_by_id(self, step_id: str) -> Step | None:
        for s in self.steps:
            if s.id == step_id:
                return s
        return None

    def steps_of_kind(self, kind: StepKind) -> list[Step]:
        return [s for s in self.steps if s.kind == kind]

    def tool_calls(self) -> list[ToolCall]:
        return [s.tool_call for s in self.steps if s.tool_call is not None]

    def tool_results(self) -> list[ToolResult]:
        return [s.tool_result for s in self.steps if s.tool_result is not None]

    def ordered_steps(self) -> list[Step]:
        return sorted(self.steps, key=lambda s: (s.timestamp, s.id))

    def children_of(self, step_id: str) -> list[Step]:
        return [s for s in self.steps if s.parent_id == step_id]


class TraceBundle(StrictModel):
    """One or more traces plus optional expected findings (for evaluation)."""

    traces: list[Trace] = Field(default_factory=list)
    expected_findings: list[dict[str, Any]] = Field(default_factory=list)
    description: str = ""


class Evidence(StrictModel):
    kind: str
    step_ids: list[str] = Field(default_factory=list)
    detail: dict[str, Any] = Field(default_factory=dict)
    quote: str | None = None


class DetectorFinding(StrictModel):
    id: str = Field(default_factory=lambda: _new_id("finding"))
    run_id: str
    detector_id: str
    category: str
    severity: Severity = Severity.MEDIUM
    status: FindingStatus = FindingStatus.RULE_TRIGGERED
    summary: str
    description: str = ""
    evidence: list[Evidence] = Field(default_factory=list)
    metrics: dict[str, Any] = Field(default_factory=dict)
    created_at: datetime = Field(default_factory=utcnow)

    @property
    def step_ids(self) -> list[str]:
        out: list[str] = []
        for e in self.evidence:
            out.extend(e.step_ids)
        # preserve order, unique
        seen: set[str] = set()
        uniq: list[str] = []
        for s in out:
            if s not in seen:
                seen.add(s)
                uniq.append(s)
        return uniq


class OracleCheck(StrictModel):
    name: str
    passed: bool
    detail: str = ""
    expected: Any = None
    actual: Any = None


class OracleOutcome(StrictModel):
    id: str = Field(default_factory=lambda: _new_id("oracle"))
    run_id: str
    task_id: str
    task_name: str | None = None
    success: bool
    checks: list[OracleCheck] = Field(default_factory=list)
    detail: str = ""
    created_at: datetime = Field(default_factory=utcnow)

    @property
    def failed_checks(self) -> list[OracleCheck]:
        return [c for c in self.checks if not c.passed]


class EvaluationRecord(StrictModel):
    id: str = Field(default_factory=lambda: _new_id("eval"))
    name: str
    kind: str
    run_id: str | None = None
    data: dict[str, Any] = Field(default_factory=dict)
    created_at: datetime = Field(default_factory=utcnow)


class RegressionTestState(StrEnum):
    PROPOSED = "proposed"
    VALIDATED = "validated"
    FAILING = "failing"
    PASSING = "passing"
    REJECTED = "rejected"


class RegressionTest(StrictModel):
    id: str = Field(default_factory=lambda: _new_id("reg"))
    state: RegressionTestState = RegressionTestState.PROPOSED
    scenario: str
    preconditions: list[str] = Field(default_factory=list)
    task: TaskSpec
    tools_required: list[str] = Field(default_factory=list)
    fault_schedule: list[dict[str, Any]] = Field(default_factory=list)
    expected_behavior: str = ""
    oracle: dict[str, Any] = Field(default_factory=dict)
    reproduction: str = ""
    original_finding_ids: list[str] = Field(default_factory=list)
    original_run_id: str | None = None
    source: str = "generated"
    created_at: datetime = Field(default_factory=utcnow)
    last_result: dict[str, Any] = Field(default_factory=dict)
