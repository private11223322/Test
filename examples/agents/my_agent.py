"""Example: plugging your own agent into Failsafe.

An agent is any object with ``name`` and a ``run(task, tools, recorder)``
method.  ``tools`` is a dict of callables already wrapped with fault
injection; ``recorder`` captures the execution as a Failsafe trace.

This file contains a documented sample agent (``LookupAgent``) plus a demo
that runs it against the built-in oracle tasks.  Run it with:

    python examples/agents/my_agent.py
"""

from __future__ import annotations

import sys
from pathlib import Path
from typing import Any, Callable

sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "src"))

from failsafe.agents.sample import _call  # small recorded-call helper
from failsafe.evaluation import (
    RunnerConfig,
    compute_run_metrics,
    execute_suite,
    get_builtin_tasks,
)
from failsafe.recording import TraceRecorder
from failsafe.schemas import TaskSpec


class LookupAgent:
    """A minimal honest agent: does its best with the tools it is given.

    Behavior contract:
    - calls each tool at most once per attempt (no loops),
    - retries a transient failure at most once,
    - never claims success it cannot demonstrate,
    - reports failure explicitly when the task cannot be completed.
    """

    name = "lookup-agent"
    version = "0.1"

    def run(self, task: TaskSpec, tools: dict[str, Callable[..., Any]], recorder: TraceRecorder) -> str:
        if "db_get" in tools and task.id == "task-db-lookup":
            out, ok = _call(recorder, tools, "db_get", {"record_id": "u-1001"})
            if not ok:
                return "Failed to look up u-1001: the lookup tool returned an error."
            return f"The plan for u-1001 is {out.get('plan', 'unknown')}."

        if "db_update" in tools and "upgrade" in task.instruction.lower():
            _, ok = _call(recorder, tools, "db_update", {"record_id": "u-1002", "fields": {"plan": "enterprise"}}, retries=1)
            if not ok:
                return "Failed to update u-1002: the update tool returned an error."
            return "Updated u-1002 plan to enterprise."

        return f"I do not have a policy for task '{task.id}' and will not claim completion."


def main() -> None:
    agent = LookupAgent()
    tasks = [t for t in get_builtin_tasks() if t.id in ("task-db-lookup", "task-update-record")]
    results = execute_suite(agent, tasks, RunnerConfig(seed=1337))
    for r in results:
        print(f"{r.task_id:24s} oracle={'PASS' if r.oracle.success else 'FAIL'} "
              f"findings={len(r.findings)} response={r.trace.metadata.final_response!r}")
    metrics = compute_run_metrics(results)
    for v in metrics.values:
        print(f"  {v.name:32s} = {v.value}")
    print("\nNow analyze an example trace from the CLI:")
    print("  failsafe analyze examples/traces/sample_clean.json")
    print("  failsafe analyze examples/traces/sample_repeated_actions.json")


if __name__ == "__main__":
    main()
