"""Experiment 5: investigation-agent regression-test evaluation.

Measures how often regression tests generated from observed failures
reproduce the original issue (i.e. fail while the defect is present and pass
once corrected).  Uses the buggy agent as the defect carrier and the good
agent as the corrected agent — the classic failing-test-becomes-passing
demonstration.

Run:  python evals/regression_eval.py
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from failsafe.agents.investigation import LANGGRAPH_AVAILABLE, investigate_run  # noqa: E402
from failsafe.agents.regression import (  # noqa: E402
    generate_from_findings,
    task_from_regression_test,
)
from failsafe.agents.sample import get_agent  # noqa: E402
from failsafe.evaluation import RunnerConfig, execute_suite, execute_task, get_builtin_tasks  # noqa: E402

OUT = ROOT / "evals" / "results" / "regression_eval.json"


def main() -> int:
    tasks = get_builtin_tasks()
    buggy_results = execute_suite(get_agent("buggy"), tasks, seed=1337)
    good_results = execute_suite(get_agent("good"), tasks, seed=1337)

    generated = validated = reproduced_by_defect = passed_after_fix = 0
    rows = []
    for buggy, good in zip(buggy_results, good_results):
        tests = generate_from_findings(buggy.trace, buggy.findings)
        for test in tests:
            generated += 1
            if test.state.value == "validated":
                validated += 1
            exec_task = task_from_regression_test(test)
            # defect present (buggy agent): a reproducing test FAILS the oracle
            # or the original detector category fires again on the new run.
            defect_run = execute_task(get_agent("buggy"), exec_task, RunnerConfig(seed=1337))
            from failsafe.agents.regression import regression_reproduces_issue

            refired_categories = {f.category for f in defect_run.findings}
            original_categories = {
                next((x.category for x in buggy.findings if x.id == fid), None)
                for fid in test.original_finding_ids
            }
            detector_refired = any(c in refired_categories for c in original_categories if c)
            reproduces = regression_reproduces_issue(
                test, result_oracle_success=defect_run.oracle.success, detector_refired=detector_refired
            )
            # defect corrected (good agent): the test should PASS
            fixed_run = execute_task(get_agent("good"), exec_task, RunnerConfig(seed=1337))
            passes_after_fix = fixed_run.oracle.success
            reproduced_by_defect += int(reproduces)
            passed_after_fix += int(passes_after_fix)
            rows.append({
                "test_id": test.id,
                "source_task": buggy.task_id,
                "category": test.task.name,
                "state_at_generation": test.state.value,
                "reproduces_defect": reproduces,
                "detector_refired": detector_refired,
                "passes_after_fix": passes_after_fix,
            })

    print(f"Generated tests:            {generated}")
    print(f"Validated at generation:    {validated}")
    if generated:
        print(f"Reproduce original defect:  {reproduced_by_defect}/{generated} "
              f"({reproduced_by_defect / generated:.2f})")
        print(f"Pass after defect is fixed: {passed_after_fix}/{generated} "
              f"({passed_after_fix / generated:.2f})")
    print("\nEach generated test: scenario, preconditions, task spec, tools, fault schedule,")
    print("expected behavior, oracle checker, reproduction steps, and original failure references.")

    if LANGGRAPH_AVAILABLE:
        suspicious = next((r for r in buggy_results if r.findings), None)
        if suspicious is not None:
            report = investigate_run(suspicious.trace, suspicious.findings, authorized_replay=True)
            print(f"\nInvestigation agent (LangGraph): status={report.status}")
            print(f"  hypothesis: {report.hypothesis[:120]}")
            print(f"  proposed test validated: {bool(report.proposed_test)}")
            print(f"  replay: {report.replay_result}")
            rows.append({"investigation": report.to_dict()})

    OUT.parent.mkdir(parents=True, exist_ok=True)
    OUT.write_text(json.dumps(
        {"generated": generated, "validated": validated,
         "reproduced_by_defect": reproduced_by_defect, "passed_after_fix": passed_after_fix,
         "rows": rows}, indent=2, default=str), encoding="utf-8")
    print(f"\nWrote {OUT}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
