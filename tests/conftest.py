"""Shared helpers for Failsafe tests."""

from __future__ import annotations

from datetime import UTC, datetime, timedelta

import pytest

from failsafe.schemas import (
    RunMetadata,
    RunStatus,
    Step,
    StepKind,
    StepStatus,
    TaskSpec,
    ToolCall,
    ToolResult,
    Trace,
)

BASE = datetime(2026, 1, 1, 12, 0, 0, tzinfo=UTC)


class TraceBuilder:
    """Compact deterministic trace builder for tests."""

    def __init__(self, run_id: str = "run-test", task: TaskSpec | None = None):
        self.trace = Trace(metadata=RunMetadata(run_id=run_id, agent_name="test-agent"), task=task)
        self._i = 0
        self._calls: dict[str, ToolCall] = {}

    def _next(self, prefix: str) -> str:
        self._i += 1
        return f"{prefix}-{self._i:03d}"

    def tool_call(self, name: str, arguments: dict | None = None, call_id: str | None = None) -> ToolCall:
        tc = ToolCall(id=call_id or self._next("tc"), name=name, arguments=arguments or {})
        step = Step(
            id=self._next("step"), kind=StepKind.TOOL_CALL, name=name,
            timestamp=BASE + timedelta(seconds=self._i),
            tool_call=tc, input=arguments or {},
        )
        tc.step_id = step.id
        self._calls[tc.id] = tc
        self.trace.steps.append(step)
        return tc

    def tool_result(self, tc: ToolCall, success: bool = True, output=None,
                    error_message: str | None = None, name: str | None = None) -> ToolResult:
        tr = ToolResult(
            tool_call_id=tc.id, name=name or tc.name, success=success,
            output=output, error_message=error_message,
            error_type=None if success else "tool_error",
        )
        step = Step(
            id=self._next("step"), kind=StepKind.TOOL_RESULT, name=tc.name,
            timestamp=BASE + timedelta(seconds=self._i),
            status=StepStatus.OK if success else StepStatus.ERROR,
            tool_result=tr, output=output, error_message=error_message,
        )
        tr.step_id = step.id
        self.trace.steps.append(step)
        return tr

    def call(self, name: str, arguments: dict | None = None, success: bool = True, output=None,
             error_message: str | None = None) -> ToolCall:
        tc = self.tool_call(name, arguments)
        self.tool_result(tc, success=success, output=output, error_message=error_message)
        return tc

    def llm(self, prompt: str, response: str = "") -> Step:
        step = Step(
            id=self._next("step"), kind=StepKind.LLM, name="llm",
            timestamp=BASE + timedelta(seconds=self._i),
            input={"prompt": prompt}, output={"response": response},
        )
        self.trace.steps.append(step)
        return step

    def response(self, text: str, claimed_success: bool | None = None) -> Step:
        meta = {} if claimed_success is None else {"claimed_success": claimed_success}
        step = Step(
            id=self._next("step"), kind=StepKind.RESPONSE, name="final",
            timestamp=BASE + timedelta(seconds=self._i),
            output=text, metadata=meta,
        )
        self.trace.steps.append(step)
        return step

    def error(self, message: str) -> Step:
        step = Step(
            id=self._next("step"), kind=StepKind.ERROR, name="error",
            timestamp=BASE + timedelta(seconds=self._i),
            status=StepStatus.ERROR, error_type="agent_error", error_message=message,
        )
        self.trace.steps.append(step)
        return step

    def finish(self, status: RunStatus = RunStatus.SUCCESS, final_response: str | None = None) -> Trace:
        self.trace.metadata.status = status
        self.trace.metadata.ended_at = BASE + timedelta(seconds=self._i + 1)
        if final_response is not None:
            self.trace.metadata.final_response = final_response
        return self.trace


@pytest.fixture
def builder():
    return TraceBuilder
