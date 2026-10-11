"""Experiment 3: fault-injection recovery evaluation.

For every fault kind, inject the fault into a recoverable task and measure:
- injection success rate (faults actually reaching the agent),
- fault recovery rate (oracle-verified completion after the fault),
- safe-handling rate (explicit correct failure reports),
- verified silent failures.

Run:  python evals/fault_injection_eval.py
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from failsafe.agents.sample import get_agent  # noqa: E402
from failsafe.evaluation import RunnerConfig, execute_task, get_builtin_tasks  # noqa: E402
from failsafe.schemas import FaultKind  # noqa: E402

OUT = ROOT / "evals" / "results" / "fault_injection_eval.json"


def main() -> int:
    tasks = {t.spec.id: t for t in get_builtin_tasks()}
    base = tasks["task-recoverable-fault"]
    rows = []
    for kind in FaultKind:
        for agent_name in ("good", "flaky"):
            task = type(base)(
                spec=base.spec,
                world_factory=base.world_factory,
                oracle_checks=base.oracle_checks,
                tools_required=base.tools_required,
                fault_schedule=[{"kind": kind.value, "tool": "db_update", "occurrence": 1,
                                 "params": {"retry_after": 0.1, "age_seconds": 3600}}],
                expected_behavior=base.expected_behavior,
                category=base.category,
            )
            res = execute_task(get_agent(agent_name), task, RunnerConfig(seed=1337))
            rows.append({
                "fault_kind": kind.value,
                "agent": agent_name,
                "injected": res.faults_injected > 0,
                "oracle_success": res.oracle.success,
                "recovered": res.recovered,
                "safe_handled": res.safe_handled,
                "silent_failure": res.silent_failure,
                "findings": [f.detector_id for f in res.findings],
            })

    total = len(rows)
    injected = sum(1 for r in rows if r["injected"])
    print(f"Runs: {total}, injections triggered: {injected}")
    print(f"{'fault_kind':22s} {'agent':7s} {'injected':8s} {'oracle':7s} {'recovered':10s} {'safe':5s} {'silent':7s}")
    for r in rows:
        print(f"{r['fault_kind']:22s} {r['agent']:7s} {str(r['injected']):8s} "
              f"{str(r['oracle_success']):7s} {str(r['recovered']):10s} "
              f"{str(r['safe_handled']):5s} {str(r['silent_failure']):7s}")

    by_agent = {}
    for agent in ("good", "flaky"):
        sub = [r for r in rows if r["agent"] == agent]
        rec = sum(1 for r in sub if r["recovered"])
        inj = sum(1 for r in sub if r["injected"])
        by_agent[agent] = {
            "recovery_rate": rec / inj if inj else None,
            "safe_handling_rate": sum(1 for r in sub if r["safe_handled"]) / len(sub),
            "silent_failure_rate": sum(1 for r in sub if r["silent_failure"]) / len(sub),
            "runs": len(sub),
        }
        print(f"\nAgent '{agent}': recovery_rate={by_agent[agent]['recovery_rate']} "
              f"safe_handling_rate={by_agent[agent]['safe_handling_rate']} "
              f"silent_failure_rate={by_agent[agent]['silent_failure_rate']}")

    OUT.parent.mkdir(parents=True, exist_ok=True)
    OUT.write_text(json.dumps({"rows": rows, "by_agent": by_agent}, indent=2, default=str), encoding="utf-8")
    print(f"\nWrote {OUT}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
