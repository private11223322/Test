"""MAST evaluation harness.

Evaluates Failsafe detectors against MAST labels where the correspondence is
genuine (see docs/mast_mapping.md).  The external dataset is NOT available in
this repository; the harness runs on documented fixtures and always reports:

    EXTERNAL_BENCHMARK_STATUS: BLOCKED

until a real dataset is supplied with --dataset.

Run:  python evals/mast/harness.py [--dataset PATH]
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "src"))

from failsafe.detectors import run_detectors  # noqa: E402
from failsafe.evaluation.metrics import detector_metrics  # noqa: E402
from failsafe.ingestion.loader import load_trace_bundle  # noqa: E402

FIXTURES = Path(__file__).resolve().parent / "fixtures"
OUT = ROOT / "evals" / "results" / "mast_eval.json"

# Only labels with genuine correspondence are scored (docs/mast_mapping.md).
SCORABLE = {"repeated_actions", "premature_termination", "unsupported_success", "lost_context"}
# Detector categories excluded from MAST scoring (no genuine MAST label).
EXCLUDED = {"ignored_tool_errors", "excessive_retries"}


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset", default=None,
                        help="Path to converted MAST fixture bundles (optional).")
    args = parser.parse_args()

    external = args.dataset is not None
    if not external:
        print("EXTERNAL_BENCHMARK_STATUS: BLOCKED")
        print("Reason: the MAST dataset is not bundled and was not accessible in the")
        print("build environment. Only documented fixtures are evaluated.\n")

    paths = sorted(FIXTURES.glob("*.json"))
    if external:
        extra = sorted(Path(args.dataset).glob("*.json"))
        if not extra:
            print(f"No fixture bundles found under {args.dataset}", file=sys.stderr)
            return 2
        paths = extra

    findings_by_run = {}
    expected_by_run = {}
    excluded_predictions = 0
    uncertain = 0

    for path in paths:
        bundle = load_trace_bundle(path)
        for trace in bundle.traces:
            all_findings = run_detectors(trace)
            # exclude detectors with no genuine MAST label from scoring
            findings_by_run[trace.metadata.run_id] = [
                f for f in all_findings if f.category in SCORABLE
            ]
            excluded_predictions += sum(1 for f in all_findings if f.category in EXCLUDED)
        for exp in bundle.expected_findings:
            if exp.get("uncertain"):
                uncertain += 1
                continue
            run_id = exp.get("run_id")
            if run_id and exp.get("category") in SCORABLE:
                expected_by_run.setdefault(run_id, []).append(exp)

    report = detector_metrics(findings_by_run, expected_by_run)
    print(f"Fixtures evaluated:            {len(paths)}")
    print(f"Expected (scorable) labels:    {report.sample_counts.get('expected', 0)}")
    print(f"Predicted (scorable) findings: {report.sample_counts.get('predicted', 0)}")
    print(f"Predictions excluded (no genuine MAST label): {excluded_predictions}")
    print(f"Uncertain annotations reported separately:    {uncertain}")
    print("\nMetrics on scorable labels only:")
    for v in report.values:
        frac = f" ({v.numerator}/{v.denominator})" if v.numerator is not None else ""
        print(f"  {v.name:24s} = {v.value}{frac}")
    print("\nPer-label confusion matrices:")
    for cat, cm in report.confusion_matrices.items():
        print(f"  {cat:24s} tp={cm.get('tp', 0)} fp={cm.get('fp', 0)} fn={cm.get('fn', 0)}")

    OUT.parent.mkdir(parents=True, exist_ok=True)
    OUT.write_text(json.dumps({
        "external_benchmark_status": "BLOCKED" if not external else "USER_SUPPLIED",
        "scorable_labels": sorted(SCORABLE),
        "excluded_categories": sorted(EXCLUDED),
        "excluded_predictions": excluded_predictions,
        "uncertain_annotations": uncertain,
        "metrics": report.to_dict(),
    }, indent=2, default=str), encoding="utf-8")
    print(f"\nWrote {OUT}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
