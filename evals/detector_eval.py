"""Experiment 1: deterministic detector evaluation.

Measures detector precision / recall / F1 against expected findings on
documented fixtures (labelled traces).  Prints sample counts, class
imbalance, per-category confusion matrices, and the raw numbers.

Run:  python evals/detector_eval.py
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from failsafe.detectors import run_detectors  # noqa: E402
from failsafe.evaluation.metrics import detector_metrics  # noqa: E402
from failsafe.ingestion.loader import load_trace_bundle  # noqa: E402

FIXTURES = [
    ROOT / "evals" / "fixtures" / "detector_cases_clean.json",
    ROOT / "evals" / "fixtures" / "detector_cases_repeated.json",
    ROOT / "evals" / "fixtures" / "detector_cases_errors.json",
    ROOT / "evals" / "fixtures" / "detector_cases_premature.json",
    ROOT / "evals" / "fixtures" / "detector_cases_success_claim.json",
    ROOT / "evals" / "fixtures" / "detector_cases_context.json",
]


def main() -> int:
    findings_by_run: dict[str, list] = {}
    expected_by_run: dict[str, list[dict]] = {}
    total_traces = 0
    total_expected = 0

    for path in FIXTURES:
        if not path.is_file():
            print(f"[skip] fixture missing: {path}")
            continue
        bundle = load_trace_bundle(path)
        for trace in bundle.traces:
            total_traces += 1
            findings_by_run[trace.metadata.run_id] = run_detectors(trace)
        for exp in bundle.expected_findings:
            run_id = exp.get("run_id")
            if run_id:
                expected_by_run.setdefault(run_id, []).append(exp)
                total_expected += 1

    print(f"Traces evaluated:      {total_traces}")
    print(f"Expected findings:     {total_expected}")
    predicted = sum(len(v) for v in findings_by_run.values())
    print(f"Predicted findings:    {predicted}")
    imbalance = total_expected / max(total_traces, 1)
    print(f"Findings per trace:    {imbalance:.2f} (class imbalance indicator)")

    report = detector_metrics(findings_by_run, expected_by_run)
    print("\nAggregate detector metrics:")
    for v in report.values:
        frac = f" ({v.numerator}/{v.denominator})" if v.numerator is not None else ""
        print(f"  {v.name:24s} = {v.value}{frac}")
    print("\nPer-category confusion matrices (tp/fp/fn):")
    for cat, cm in report.confusion_matrices.items():
        print(f"  {cat:24s} tp={cm.get('tp', 0)} fp={cm.get('fp', 0)} fn={cm.get('fn', 0)}")
    print(f"\nSample counts: {report.sample_counts}")
    out = ROOT / "evals" / "results" / "detector_eval.json"
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(report.to_dict(), indent=2, default=str), encoding="utf-8")
    print(f"Wrote {out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
