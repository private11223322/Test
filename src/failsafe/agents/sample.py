"""Built-in sample agents for demonstrations and tests.

These agents are deterministic scripted policies (no LLMs) that run against
the synthetic oracle tasks through the tool API.  They make excellent clean
controls and known-buggy baselines:

- ``good``    completes tasks correctly and recovers from transient faults.
- ``buggy``   claims success without doing the work (silent failures).
- ``looping`` repeats identical tool calls (repeated actions).
- ``flaky``   ignores tool errors and continues as if nothing happened.

User agents plug into Failsafe by implementing the same
``run(task, tools, recorder) -> str`` interface.
"""

from __future__ import annotations

from collections.abc import Callable
from typing import Any

from failsafe.injection.faults import InjectedFault
from failsafe.recording import TraceRecorder
from failsafe.schemas import StepKind, TaskSpec


class BaseSampleAgent:
    name = "base"
    version = "0.1"

    def run(self, task: TaskSpec, tools: dict[str, Callable[..., Any]], recorder: TraceRecorder) -> str:
        raise NotImplementedError


def _call(
    recorder: TraceRecorder,
    tools: dict[str, Callable[..., Any]],
    name: str,
    arguments: dict[str, Any],
    retries: int = 0,
) -> tuple[Any, bool]:
    """Call a tool with recording and optional retries. Returns (output, ok)."""
    attempts = 0
    while True:
        attempts += 1
        tc = recorder.record_tool_call(name, arguments)
        try:
            out = tools[name](**arguments)
            recorder.record_tool_result(tc, out)
            return out, True
        except InjectedFault as fault:
            recorder.record_tool_result(tc, success=False, error_type=fault.kind.value,
                                        error_message=str(fault))
            if attempts <= retries:
                continue
            return None, False
        except Exception as exc:  # noqa: BLE001 - tool errors are data for the agent
            recorder.record_tool_result(tc, success=False, error_type=type(exc).__name__,
                                        error_message=str(exc))
            if attempts <= retries:
                continue
            return None, False


class GoodAgent(BaseSampleAgent):
    """A correct agent: looks things up, updates exactly once, retries transient
    faults once, and reports failure honestly when the task cannot be done."""

    name = "good-agent"
    version = "0.1"

    def run(self, task: TaskSpec, tools, recorder) -> str:  # type: ignore[override]
        tid = task.id

        if tid == "task-db-lookup":
            out, ok = _call(recorder, tools, "db_get", {"record_id": "u-1001"})
            if not ok:
                return "Failed to look up u-1001: tool error."
            return f"The plan for u-1001 is {out.get('plan', 'unknown')}."

        if tid == "task-multi-step":
            rec, ok = _call(recorder, tools, "db_get", {"record_id": "ord-55"})
            if not ok:
                return "Failed to read order ord-55."
            if rec.get("status") == "pending":
                _, ok2 = _call(recorder, tools, "db_update",
                               {"record_id": "ord-55", "fields": {"status": "shipped"}}, retries=1)
                if not ok2:
                    return "Failed to update order ord-55."
                return "Order ord-55 has been shipped."
            return f"Order ord-55 status is {rec.get('status')}."

        if tid == "task-search-corpus":
            out, ok = _call(recorder, tools, "search", {"query": "refund"})
            if not ok or not out:
                return "Failed to search the knowledge base."
            body = out[0].get("body", "")
            return f"Refunds take 14 days ({body})"

        if tid == "task-update-record":
            _, ok = _call(recorder, tools, "db_update",
                          {"record_id": "u-1002", "fields": {"plan": "enterprise"}}, retries=1)
            if not ok:
                return "Failed to update u-1002."
            return "Updated u-1002 plan to enterprise."

        if tid == "task-insert-record":
            _, ok = _call(recorder, tools, "db_insert", {
                "record_id": "u-2001",
                "record": {"id": "u-2001", "name": "Katherine Johnson", "plan": "trial", "active": True},
            })
            if not ok:
                return "Failed to create u-2001."
            return "Created user u-2001."

        if tid == "task-recoverable-fault":
            _, ok = _call(recorder, tools, "db_update",
                          {"record_id": "u-1001", "fields": {"plan": "pro"}}, retries=2)
            if not ok:
                return "Failed to upgrade u-1001 after retries."
            return "Successfully updated u-1001 plan to pro."

        if tid == "task-unrecoverable-fault":
            _, ok = _call(recorder, tools, "db_update",
                          {"record_id": "u-9999", "fields": {"plan": "pro"}}, retries=1)
            if not ok:
                return ("Failed to update u-9999: the record does not exist, so the update "
                        "could not be completed. No changes were made.")
            return "Updated u-9999."

        if tid == "task-stale-result":
            out, ok = _call(recorder, tools, "fetch_url", {"url": "https://api.example.com/config"})
            if not ok:
                return "Failed to fetch config."
            stale = isinstance(out, dict) and (out.get("body", {}) or {}).get("stale")
            if stale:
                return ("The fetched config is marked stale, so I did not apply it. "
                        "The record was left unchanged; a refresh is required.")
            value = (out.get("body") or {}).get("value")
            _, ok2 = _call(recorder, tools, "db_update", {"record_id": "cfg-1", "fields": {"value": value}})
            return "Applied fresh config value." if ok2 else "Failed to apply config."

        if tid == "task-duplicate-mutation":
            # read state first; then one update; on timeout, verify state before retrying
            _, ok = _call(recorder, tools, "db_update",
                          {"record_id": "u-1003", "fields": {"active": True}}, retries=0)
            if not ok:
                rec, ok2 = _call(recorder, tools, "db_get", {"record_id": "u-1003"})
                if ok2 and rec.get("active") is True:
                    return "u-1003 is already active."
                _, ok3 = _call(recorder, tools, "db_update",
                               {"record_id": "u-1003", "fields": {"active": True}})
                if not ok3:
                    return "Failed to activate u-1003."
            return "Successfully updated u-1003 active=true."

        if tid == "task-email-notification":
            _, ok = _call(recorder, tools, "send_email",
                          {"to": "ada@example.com", "body": "Your plan was upgraded."})
            return "Email sent." if ok else "Failed to send email."

        if tid == "task-delete-record":
            _, ok = _call(recorder, tools, "db_delete", {"record_id": "inv-9"})
            return "Deleted inv-9." if ok else "Failed to delete inv-9."

        if tid == "task-multi-record-transfer":
            r9, ok1 = _call(recorder, tools, "db_get", {"record_id": "inv-9"})
            r10, ok2 = _call(recorder, tools, "db_get", {"record_id": "inv-10"})
            if not (ok1 and ok2):
                return "Failed to read inventory records."
            _, ok3 = _call(recorder, tools, "db_update",
                           {"record_id": "inv-9", "fields": {"qty": r9["qty"] - 1}})
            _, ok4 = _call(recorder, tools, "db_update",
                           {"record_id": "inv-10", "fields": {"qty": r10["qty"] + 1}})
            if not (ok3 and ok4):
                return "Failed to transfer stock."
            return f"Transferred 1 unit: inv-9 qty={r9['qty'] - 1}, inv-10 qty={r10['qty'] + 1}."

        if tid == "task-rate-limited-retry":
            _, ok = _call(recorder, tools, "db_update",
                          {"record_id": "u-1002", "fields": {"active": False}}, retries=2)
            return "Successfully updated u-1002 active=false." if ok else "Failed to update u-1002."

        return f"I don't know how to handle task '{tid}'."


class BuggyAgent(BaseSampleAgent):
    """Claims success without doing the work — produces verified silent failures."""

    name = "buggy-agent"
    version = "0.1"

    def run(self, task: TaskSpec, tools, recorder) -> str:  # type: ignore[override]
        # does nothing, claims success
        recorder.add_step(
            StepKind.NOTE,
            "skip_work",
            output="Skipping actual tool calls; claiming success.",
        )
        return "Task completed successfully! All updates applied successfully."


class LoopingAgent(BaseSampleAgent):
    """Repeats identical tool calls — triggers the repeated_actions detector."""

    name = "looping-agent"
    version = "0.1"

    def run(self, task: TaskSpec, tools, recorder) -> str:  # type: ignore[override]
        for _ in range(5):
            _call(recorder, tools, "db_get", {"record_id": "u-1001"})
        return "I checked the record many times. The plan is free."


class FlakyAgent(BaseSampleAgent):
    """Ignores tool errors and continues as if nothing happened."""

    name = "flaky-agent"
    version = "0.1"

    def run(self, task: TaskSpec, tools, recorder) -> str:  # type: ignore[override]
        tid = task.id
        if tid == "task-recoverable-fault":
            _call(recorder, tools, "db_update", {"record_id": "u-1001", "fields": {"plan": "pro"}})
            # ignores whether it failed; claims success regardless
            return "Successfully updated u-1001 plan to pro."
        _call(recorder, tools, "db_get", {"record_id": "u-1001"})
        return "Done, everything looks successful."


AGENTS: dict[str, type[BaseSampleAgent]] = {
    "good": GoodAgent,
    "buggy": BuggyAgent,
    "looping": LoopingAgent,
    "flaky": FlakyAgent,
}


def get_agent(name: str) -> BaseSampleAgent:
    if name not in AGENTS:
        raise KeyError(f"Unknown sample agent '{name}'. Known: {sorted(AGENTS)}")
    return AGENTS[name]()
