"""Experiment 2: oracle-task reliability evaluation.

Runs every sample agent against the built-in oracle task suite and reports
task success rate, safe-handling rate, verified silent-failure rate, and
per-task outcomes.  All outcomes are oracle-verified, never self-reported.

Run:  python evals/oracle_reliability_eval.py
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from failsafe.agents.sample import AGENTS, get_agent  # noqa: E402
from failsafe.evaluation import compute_run_metrics, execute_suite, get_builtin_tasks  # noqa: E402

OUT = ROOT / "evals" / "results" / "oracle_reliability_eval.json"


def main() -> int:
    tasks = get_builtin_tasks()
    all_rows = {}
    for name in sorted(AGENTS):
        results = execute_suite(get_agent(name), tasks, seed=1337)
        metrics = compute_run_metrics(results)
        rows = [
            {
                "task_id": r.task_id,
                "oracle_success": r.task_success,
                "recovered": r.recovered,
                "safe_handled": r.safe_handled,
                "silent_failure": r.silent_failure,
                "findings": len(r.findings),
                "faults_injected": r.faults_injected,
                "runtime_ms": round(r.runtime_ms, 3),
            }
            for r in results
        ]
        all_rows[name] = {"tasks": rows, "metrics": metrics.to_dict()}
        succ = sum(1 for r in rows if r["oracle_success"])
        print(f"\nAgent '{name}': {succ}/{len(rows)} tasks verified | "
              f"silent failures {sum(1 for r in rows if r['silent_failure'])} | "
              f"recovered {sum(1 for r in rows if r['recovered'])} | "
              f"safe {sum(1 for r in rows if r['safe_handled'])}")
        for v in metrics.values:
            if v.value is not None:
                frac = f" ({v.numerator}/{v.denominator})" if v.numerator is not None else ""
                print(f"    {v.name:32s} = {v.value}{frac}")

    OUT.parent.mkdir(parents=True, exist_ok=True)
    OUT.write_text(json.dumps(all_rows, indent=2, default=str), encoding="utf-8")
    print(f"\nWrote {OUT}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
