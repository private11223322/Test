"""Experiment 6: adaptive versus fixed fault-injection testing.

Runs an adaptive campaign and a fixed-schedule baseline under EQUAL budgets
(same seed, same max scenarios, same max faults) and compares how many
failure-producing scenarios each policy discovers.

Run:  python evals/adaptive_vs_fixed.py
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from failsafe.agents.sample import get_agent  # noqa: E402
from failsafe.evaluation import (  # noqa: E402
    AdaptiveCampaign,
    CampaignBudget,
    FixedCampaign,
    get_builtin_tasks,
)

OUT = ROOT / "evals" / "results" / "adaptive_vs_fixed.json"


def summarize(report) -> dict:
    results = report.results
    return {
        "policy": report.policy,
        "scenarios_used": report.budgets.get("scenarios_used"),
        "faults_injected_total": report.budgets.get("faults_injected_total"),
        "failure_producing_scenarios": sum(1 for r in results if r.get("produced_failure")),
        "oracle_failures": sum(1 for r in results if not r.get("oracle_success", True)),
        "elapsed_seconds": report.budgets.get("elapsed_seconds"),
    }


def main() -> int:
    tasks = get_builtin_tasks()
    budget = CampaignBudget(max_scenarios=8, max_seconds=30.0, max_faults_total=8)
    seed = 1337

    adaptive = AdaptiveCampaign(get_agent("flaky"), tasks, seed=seed, budget=budget).run()
    fixed = FixedCampaign(get_agent("flaky"), tasks, seed=seed, budget=budget).run()

    a, f = summarize(adaptive), summarize(fixed)
    print("Equal budgets: max_scenarios=8, max_faults_total=8, seed=1337")
    print(f"\nAdaptive ({adaptive.policy}):")
    for k, v in a.items():
        if k != "policy":
            print(f"  {k:28s} = {v}")
    print(f"\nFixed ({fixed.policy}):")
    for k, v in f.items():
        if k != "policy":
            print(f"  {k:28s} = {v}")
    print(f"\nAdaptive advantage (failure-producing scenarios): "
          f"{a['failure_producing_scenarios'] - f['failure_producing_scenarios']:+d}")
    print("Note: this is a deterministic synthetic comparison; live LLM runs add variability.")
    print("Adaptive testing is an evaluated feature here; learned scenario generation is a stretch goal.")

    OUT.parent.mkdir(parents=True, exist_ok=True)
    OUT.write_text(json.dumps(
        {"adaptive": a, "fixed": f,
         "adaptive_full": adaptive.to_dict(), "fixed_full": fixed.to_dict()},
        indent=2, default=str), encoding="utf-8")
    print(f"\nWrote {OUT}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
