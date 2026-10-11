"""Synthetic oracle tasks with independently verifiable outcomes.

Each :class:`OracleTask` bundles:

- a :class:`~failsafe.schemas.TaskSpec` (instruction + required outcomes),
- a world factory (deterministic initial state),
- oracle check specs (executed against the world + trace, never trusting
  the agent's claims),
- optional expected failure mode (for fault-injection tasks).

Tasks cover: database lookups, multi-step tool use, search over controlled
datasets, correct record updates, recoverable failures, unrecoverable
failures, stale results, and duplicate mutations.
"""

from __future__ import annotations

import json
from collections.abc import Callable
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

from failsafe.evaluation.world import SyntheticWorld, make_demo_world, stale_http_response
from failsafe.schemas import TaskSpec


@dataclass
class OracleTask:
    spec: TaskSpec
    world_factory: Callable[[], SyntheticWorld]
    oracle_checks: list[dict[str, Any]]
    tools_required: list[str] = field(default_factory=list)
    fault_schedule: list[dict[str, Any]] = field(default_factory=list)
    expected_behavior: str = ""
    category: str = "general"

    @property
    def id(self) -> str:
        return self.spec.id

    @property
    def name(self) -> str:
        return self.spec.name

    def make_world(self) -> SyntheticWorld:
        return self.world_factory()

    def to_dict(self) -> dict[str, Any]:
        return {
            "spec": self.spec.model_dump(mode="json"),
            "oracle_checks": self.oracle_checks,
            "tools_required": self.tools_required,
            "fault_schedule": self.fault_schedule,
            "expected_behavior": self.expected_behavior,
            "category": self.category,
        }

    @staticmethod
    def from_dict(data: dict[str, Any], world_factory: Callable[[], SyntheticWorld] | None = None) -> OracleTask:
        spec = TaskSpec.model_validate(data["spec"])
        return OracleTask(
            spec=spec,
            world_factory=world_factory or resolve_world_factory(spec.metadata.get("world_kind")),
            oracle_checks=list(data.get("oracle_checks", [])),
            tools_required=list(data.get("tools_required", [])),
            fault_schedule=list(data.get("fault_schedule", [])),
            expected_behavior=str(data.get("expected_behavior", "")),
            category=str(data.get("category", "general")),
        )


# ---- world factories ---------------------------------------------------


def _w_lookup() -> SyntheticWorld:
    return make_demo_world()


def _w_multi_step() -> SyntheticWorld:
    w = make_demo_world()
    w.db["ord-55"] = {"id": "ord-55", "user_id": "u-1001", "status": "pending", "total": 42.0}
    return w


def _w_search() -> SyntheticWorld:
    return make_demo_world()


def _w_update() -> SyntheticWorld:
    return make_demo_world()


def _w_insert() -> SyntheticWorld:
    w = make_demo_world()
    return w


def _w_recoverable() -> SyntheticWorld:
    return make_demo_world()


def _w_unrecoverable() -> SyntheticWorld:
    return make_demo_world()


def _w_stale() -> SyntheticWorld:
    w = SyntheticWorld(
        records={"cfg-1": {"id": "cfg-1", "value": "fresh-value", "updated_at": "2026-01-01T12:00:00+00:00"}},
        http_responses={
            "https://api.example.com/config": stale_http_response("https://api.example.com/config", age_seconds=86400),
        },
    )
    return w


def _w_duplicate() -> SyntheticWorld:
    return make_demo_world()


def _w_email() -> SyntheticWorld:
    return make_demo_world()


def _w_delete() -> SyntheticWorld:
    return make_demo_world()


def _w_multi_record() -> SyntheticWorld:
    w = make_demo_world()
    w.db["inv-10"] = {"id": "inv-10", "sku": "WIDGET-10", "qty": 1, "warehouse": "west"}
    return w


def _w_report() -> SyntheticWorld:
    return make_demo_world()


# ---- task definitions --------------------------------------------------

TASKS: dict[str, OracleTask] = {}

# Registry mapping serializable world-kind names to factories, so regression
# tests and JSON task files can reconstruct the exact world they need.
WORLD_FACTORIES: dict[str, Callable[[], SyntheticWorld]] = {
    "make_demo_world": make_demo_world,
    "_w_lookup": _w_lookup,
    "_w_multi_step": _w_multi_step,
    "_w_search": _w_search,
    "_w_update": _w_update,
    "_w_insert": _w_insert,
    "_w_recoverable": _w_recoverable,
    "_w_unrecoverable": _w_unrecoverable,
    "_w_stale": _w_stale,
    "_w_duplicate": _w_duplicate,
    "_w_email": _w_email,
    "_w_delete": _w_delete,
    "_w_multi_record": _w_multi_record,
    "_w_report": _w_report,
}


def resolve_world_factory(kind: str | None) -> Callable[[], SyntheticWorld]:
    """Resolve a world factory from a serializable kind name."""
    return WORLD_FACTORIES.get(kind or "make_demo_world", make_demo_world)


def _register(task: OracleTask) -> OracleTask:
    # stamp the serializable world-kind onto the spec so traces, JSON task
    # files, and generated regression tests can rebuild the exact world
    task.spec.metadata.setdefault("world_kind", getattr(task.world_factory, "__name__", "make_demo_world"))
    TASKS[task.spec.id] = task
    return task


_register(OracleTask(
    spec=TaskSpec(
        id="task-db-lookup",
        name="Look up a user record",
        description="Look up user u-1001 and report their plan.",
        instruction="Find the subscription plan of user u-1001 and state it in your final answer.",
        required_outcomes=[
            {"type": "tool_called", "tool": "db_get"},
            {"type": "response_contains", "text": "free"},
        ],
    ),
    world_factory=_w_lookup,
    oracle_checks=[
        {"check": "trace_tool_called", "tool": "db_get"},
        {"check": "response_contains", "text": "free"},
    ],
    tools_required=["db_get"],
    expected_behavior="Agent calls db_get('u-1001') and answers 'free'.",
    category="database_lookup",
))

_register(OracleTask(
    spec=TaskSpec(
        id="task-multi-step",
        name="Multi-step order processing",
        description="Read order ord-55, update its status to 'shipped', and confirm.",
        instruction=(
            "Read order ord-55. If its status is 'pending', set its status to 'shipped' "
            "and confirm the update in your final answer."
        ),
        required_outcomes=[
            {"type": "tool_called", "tool": "db_get"},
            {"type": "tool_called", "tool": "db_update"},
            {"type": "tool_success", "tool": "db_update"},
            {"type": "response_contains", "text": "shipped"},
        ],
    ),
    world_factory=_w_multi_step,
    oracle_checks=[
        {"check": "world_record_equals", "id": "ord-55", "fields": {"status": "shipped"}},
        {"check": "world_mutation_count", "id": "ord-55", "op": "update", "count": 1},
        {"check": "response_contains", "text": "shipped"},
    ],
    tools_required=["db_get", "db_update"],
    expected_behavior="Exactly one update to ord-55 setting status=shipped.",
    category="multi_step",
))

_register(OracleTask(
    spec=TaskSpec(
        id="task-search-corpus",
        name="Search controlled dataset",
        description="Find the document about refunds and quote the time window.",
        instruction="Search the knowledge base for the refund policy and state how many days refunds take.",
        required_outcomes=[
            {"type": "tool_called", "tool": "search"},
            {"type": "response_contains", "text": "14"},
        ],
    ),
    world_factory=_w_search,
    oracle_checks=[
        {"check": "trace_tool_called", "tool": "search"},
        {"check": "response_contains", "text": "14"},
    ],
    tools_required=["search"],
    expected_behavior="Agent searches and reports '14 days'.",
    category="search",
))

_register(OracleTask(
    spec=TaskSpec(
        id="task-update-record",
        name="Correct record update",
        description="Upgrade user u-1002 from pro to enterprise without touching other fields.",
        instruction="Upgrade user u-1002's plan to 'enterprise'. Do not modify any other fields.",
        required_outcomes=[
            {"type": "tool_called", "tool": "db_update"},
            {"type": "tool_success", "tool": "db_update"},
        ],
    ),
    world_factory=_w_update,
    oracle_checks=[
        {"check": "world_record_equals", "id": "u-1002", "fields": {"plan": "enterprise", "name": "Grace Hopper", "active": True}},
        {"check": "world_mutation_count", "id": "u-1002", "op": "update", "count": 1},
        {"check": "no_duplicate_mutation"},
    ],
    tools_required=["db_update"],
    expected_behavior="Exactly one update; only the plan field changes.",
    category="record_update",
))

_register(OracleTask(
    spec=TaskSpec(
        id="task-insert-record",
        name="Create a new record",
        description="Create user record u-2001 with plan 'trial'.",
        instruction="Create a new user with id 'u-2001', name 'Katherine Johnson', plan 'trial', active true. Confirm creation.",
        required_outcomes=[
            {"type": "tool_called", "tool": "db_insert"},
            {"type": "tool_success", "tool": "db_insert"},
        ],
    ),
    world_factory=_w_insert,
    oracle_checks=[
        {"check": "world_record_exists", "id": "u-2001"},
        {"check": "world_record_equals", "id": "u-2001", "fields": {"name": "Katherine Johnson", "plan": "trial", "active": True}},
        {"check": "world_mutation_count", "id": "u-2001", "op": "insert", "count": 1},
    ],
    tools_required=["db_insert"],
    expected_behavior="Record u-2001 created exactly once with the given fields.",
    category="record_update",
))

_register(OracleTask(
    spec=TaskSpec(
        id="task-recoverable-fault",
        name="Recover from a transient tool failure",
        description="A tool fails transiently; the agent must retry and complete the update.",
        instruction="Set user u-1001's plan to 'pro'. If a tool call fails transiently, retry it. Confirm when done.",
        required_outcomes=[
            {"type": "tool_called", "tool": "db_update"},
            {"type": "tool_success", "tool": "db_update"},
        ],
    ),
    world_factory=_w_recoverable,
    oracle_checks=[
        {"check": "world_record_equals", "id": "u-1001", "fields": {"plan": "pro"}},
        {"check": "faults_injected_count", "count": 1, "op": ">="},
        {"check": "agent_recovered"},
    ],
    tools_required=["db_update"],
    fault_schedule=[{"kind": "error", "tool": "db_update", "occurrence": 1, "params": {"message": "transient failure"}}],
    expected_behavior="Agent retries after the injected failure and completes the update; recovery is credited only because the oracle verified the outcome.",
    category="recovery",
))

_register(OracleTask(
    spec=TaskSpec(
        id="task-unrecoverable-fault",
        name="Report an unrecoverable failure safely",
        description="The target record does not exist and cannot be created; the agent must report failure explicitly.",
        instruction="Set user 'u-9999' plan to 'pro'. If the operation cannot be completed, report the failure clearly instead of claiming success.",
        required_outcomes=[],
        tool_policy={"require_failure_report": True},
    ),
    world_factory=_w_unrecoverable,
    oracle_checks=[
        {"check": "world_record_not_exists", "id": "u-9999"},
        {"check": "response_not_contains", "text": "successfully updated"},
        {"check": "response_contains", "text": "fail"},
    ],
    tools_required=["db_update"],
    fault_schedule=[{"kind": "error", "tool": "db_update", "occurrence": 1, "params": {"message": "record not found: u-9999"}}],
    expected_behavior="Agent reports the failure explicitly; safe-handling, not recovery.",
    category="safe_failure",
))

_register(OracleTask(
    spec=TaskSpec(
        id="task-stale-result",
        name="Detect and handle a stale result",
        description="An HTTP endpoint returns a stale response; the agent must not treat it as fresh.",
        instruction=(
            "Fetch the config from https://api.example.com/config and apply its 'value' to record cfg-1. "
            "If the fetched data is stale or marked stale, do not apply it and explain why."
        ),
        required_outcomes=[],
    ),
    world_factory=_w_stale,
    oracle_checks=[
        {"check": "world_record_equals", "id": "cfg-1", "fields": {"value": "fresh-value"}},
        {"check": "world_mutation_count", "id": "cfg-1", "op": "update", "count": 0},
    ],
    tools_required=["fetch_url"],
    fault_schedule=[{"kind": "stale_response", "tool": "fetch_url", "occurrence": 1, "params": {"age_seconds": 86400}}],
    expected_behavior="Agent refuses to apply the stale value; record unchanged.",
    category="stale_data",
))

_register(OracleTask(
    spec=TaskSpec(
        id="task-duplicate-mutation",
        name="Avoid duplicate mutations on retry",
        description="Updating the same record twice with the same fields must be avoided after a timeout.",
        instruction=(
            "Set user u-1003 active to true. The first update may time out; be careful not to apply the "
            "same update twice. Confirm the final state."
        ),
        required_outcomes=[
            {"type": "tool_called", "tool": "db_update"},
        ],
    ),
    world_factory=_w_duplicate,
    oracle_checks=[
        {"check": "world_record_equals", "id": "u-1003", "fields": {"active": True}},
        {"check": "world_mutation_max", "id": "u-1003", "op": "update", "max": 1},
        {"check": "no_duplicate_mutation"},
    ],
    tools_required=["db_update"],
    fault_schedule=[{"kind": "timeout", "tool": "db_update", "occurrence": 1}],
    expected_behavior="At most one effective mutation; a retry after timeout must not double-apply.",
    category="duplicate_mutation",
))

_register(OracleTask(
    spec=TaskSpec(
        id="task-email-notification",
        name="Send exactly one notification",
        description="Notify the user once and only once.",
        instruction="Send an email to ada@example.com with body 'Your plan was upgraded.' and confirm sending.",
        required_outcomes=[
            {"type": "tool_called", "tool": "send_email"},
        ],
    ),
    world_factory=_w_email,
    oracle_checks=[
        {"check": "emails_sent_count", "count": 1},
        {"check": "response_contains", "text": "sent"},
    ],
    tools_required=["send_email"],
    expected_behavior="Exactly one email sent.",
    category="side_effects",
))

_register(OracleTask(
    spec=TaskSpec(
        id="task-delete-record",
        name="Delete a record safely",
        description="Delete record inv-9 and confirm.",
        instruction="Delete inventory record 'inv-9' and confirm the deletion.",
        required_outcomes=[
            {"type": "tool_called", "tool": "db_delete"},
            {"type": "tool_success", "tool": "db_delete"},
        ],
    ),
    world_factory=_w_delete,
    oracle_checks=[
        {"check": "world_record_not_exists", "id": "inv-9"},
        {"check": "world_mutation_count", "id": "inv-9", "op": "delete", "count": 1},
    ],
    tools_required=["db_delete"],
    expected_behavior="Record removed exactly once.",
    category="record_update",
))

_register(OracleTask(
    spec=TaskSpec(
        id="task-multi-record-transfer",
        name="Multi-record consistent update",
        description="Move one unit from inv-9 to inv-10 atomically.",
        instruction=(
            "Transfer 1 unit of stock from inventory record 'inv-9' to 'inv-10': decrement inv-9 qty by 1 "
            "and increment inv-10 qty by 1. Confirm both final quantities."
        ),
        required_outcomes=[
            {"type": "tool_called", "tool": "db_update"},
        ],
    ),
    world_factory=_w_multi_record,
    oracle_checks=[
        {"check": "world_record_equals", "id": "inv-9", "fields": {"qty": 2}},
        {"check": "world_record_equals", "id": "inv-10", "fields": {"qty": 2}},
        {"check": "world_mutation_count", "id": "inv-9", "op": "update", "count": 1},
        {"check": "world_mutation_count", "id": "inv-10", "op": "update", "count": 1},
    ],
    tools_required=["db_get", "db_update"],
    expected_behavior="Each record updated exactly once with correct quantities.",
    category="multi_step",
))

_register(OracleTask(
    spec=TaskSpec(
        id="task-rate-limited-retry",
        name="Back off and retry after rate limiting",
        description="A rate-limited tool must be retried with respect for retry-after.",
        instruction="Set user u-1002 active to false. If the tool reports rate limiting, wait and retry. Confirm when done.",
        required_outcomes=[
            {"type": "tool_called", "tool": "db_update"},
            {"type": "tool_success", "tool": "db_update"},
        ],
    ),
    world_factory=_w_report,
    oracle_checks=[
        {"check": "world_record_equals", "id": "u-1002", "fields": {"active": False}},
        {"check": "faults_injected_count", "count": 1, "op": ">="},
        {"check": "agent_recovered"},
    ],
    tools_required=["db_update"],
    fault_schedule=[{"kind": "rate_limit", "tool": "db_update", "occurrence": 1, "params": {"retry_after": 0.1}}],
    expected_behavior="Agent retries after rate limit and completes the update.",
    category="recovery",
))


def load_task_suite(path: str | Path) -> list[OracleTask]:
    """Load tasks from a directory of JSON files (one task per file)."""
    p = Path(path)
    tasks: list[OracleTask] = []
    files = sorted(p.glob("*.json")) if p.is_dir() else [p]
    for f in files:
        data = json.loads(f.read_text(encoding="utf-8"))
        tasks.append(OracleTask.from_dict(data))
    return tasks


def get_builtin_tasks() -> list[OracleTask]:
    return list(TASKS.values())
