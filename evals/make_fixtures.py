"""Regenerate labelled detector-eval fixtures (deterministic).

Each fixture contains a trace produced by a scripted sample agent plus
hand-reviewed expected findings (category + evidence step IDs).  The
expected labels encode what a human reviewer would confirm given the case
definition — see tests/unit/test_detectors.py for the nuanced per-detector
contract (INFO retries, UNRESOLVED ambiguities, etc.).

Run:  python evals/make_fixtures.py
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from failsafe.agents.sample import get_agent  # noqa: E402
from failsafe.evaluation import execute_task, get_builtin_tasks  # noqa: E402
from failsafe.recording import TraceRecorder  # noqa: E402
from failsafe.schemas import RunStatus  # noqa: E402

OUT = ROOT / "evals" / "fixtures"
OUT.mkdir(parents=True, exist_ok=True)

tasks = {t.spec.id: t for t in get_builtin_tasks()}


def save(name: str, trace, expected: list[dict]) -> None:
    run_id = trace.metadata.run_id
    data = {
        "description": name,
        "traces": [trace.model_dump(mode="json")],
        "expected_findings": [{**e, "run_id": run_id} for e in expected],
    }
    path = OUT / f"{name}.json"
    path.write_text(json.dumps(data, indent=2, default=str), encoding="utf-8")
    print(f"{path.relative_to(ROOT)}  expected={len(expected)}")


def response_steps(trace) -> list[str]:
    return [s.id for s in trace.ordered_steps() if s.kind.value == "response"]


# 1. Clean control: correct agent, no findings expected (true negatives).
res = execute_task(get_agent("good"), tasks["task-db-lookup"])
save("detector_cases_clean", res.trace, [])

# 2. Repeated actions: looping agent repeats one identical lookup 5x.
res = execute_task(get_agent("looping"), tasks["task-db-lookup"])
call_steps = [s.id for s in res.trace.ordered_steps() if s.tool_call]
save("detector_cases_repeated", res.trace, [
    {"category": "repeated_actions", "step_ids": call_steps, "summary_contains": "identical"},
])

# 3. Ignored tool error + consequences: flaky agent swallows a transient fault
#    and claims success.  Human label: all three findings are genuine.
res = execute_task(get_agent("flaky"), tasks["task-recoverable-fault"])
err_steps = [s.id for s in res.trace.ordered_steps() if s.tool_result and not s.tool_result.success]
resp = response_steps(res.trace)
save("detector_cases_errors", res.trace, [
    {"category": "ignored_tool_errors", "step_ids": err_steps},
    {"category": "premature_termination", "step_ids": resp[-1:]},
    {"category": "unsupported_success", "step_ids": resp[-1:], "summary_contains": "claims success"},
])

# 4. Premature termination: buggy agent claims success without acting on a
#    4-outcome task.  Human label: 4 unmet-outcome findings + the success claim.
res = execute_task(get_agent("buggy"), tasks["task-multi-step"])
resp = response_steps(res.trace)
save("detector_cases_premature", res.trace, [
    {"category": "premature_termination", "step_ids": resp[-1:], "summary_contains": "tool_called"},
    {"category": "premature_termination", "step_ids": resp[-1:], "summary_contains": "tool_called"},
    {"category": "premature_termination", "step_ids": resp[-1:], "summary_contains": "tool_success"},
    {"category": "premature_termination", "step_ids": resp[-1:], "summary_contains": "response_contains"},
    {"category": "unsupported_success", "step_ids": resp[-1:], "summary_contains": "claims success"},
])

# 5. Unsupported success claim: buggy agent on a record-update task
#    (2 unmet outcomes + the false success claim).
res = execute_task(get_agent("buggy"), tasks["task-update-record"])
resp = response_steps(res.trace)
save("detector_cases_success_claim", res.trace, [
    {"category": "premature_termination", "step_ids": resp[-1:], "summary_contains": "tool_called"},
    {"category": "premature_termination", "step_ids": resp[-1:], "summary_contains": "tool_success"},
    {"category": "unsupported_success", "step_ids": resp[-1:], "summary_contains": "claims success"},
])

# 6. Lost / repeated context: the exact same LLM request is issued twice.
rec = TraceRecorder(agent_name="ctx-agent")
s1 = rec.record_llm("llm", "What is the plan for u-1001?", "free")
s2 = rec.record_llm("llm", "What is the plan for u-1001?", "free")
rec.record_response("The plan is free.")
trace = rec.finish(status=RunStatus.SUCCESS, final_response="The plan is free.")
save("detector_cases_context", trace, [
    {"category": "lost_context", "step_ids": [s1.id, s2.id], "summary_contains": "Identical request"},
])
