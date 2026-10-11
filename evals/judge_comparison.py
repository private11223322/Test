"""Experiment 4: LLM-judge comparison against deterministic labels.

Compares the optional semantic judge with ground-truth labels on fixture
cases.  With ``--provider echo`` (default) the pipeline is exercised with the
deterministic test stub and the results are labelled STUB RESULTS — they are
NOT model measurements.  With ``--provider openai`` and credentials, real
model judgments are produced and compared.

Run:
  python evals/judge_comparison.py                       # stub, offline
  python evals/judge_comparison.py --provider openai     # requires OPENAI_API_KEY
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from failsafe.detectors import run_detectors  # noqa: E402
from failsafe.ingestion.loader import load_trace_bundle  # noqa: E402
from failsafe.judges import EchoProvider, LLMJudge, LLMProviderError, OpenAIProvider  # noqa: E402

FIXTURES = sorted((ROOT / "evals" / "fixtures").glob("detector_cases_*.json"))
OUT = ROOT / "evals" / "results" / "judge_comparison.json"

# Suspicion prompts per category (what we ask the judge to investigate).
SUSPICIONS = {
    "repeated_actions": "The agent appears to repeat identical tool calls in a loop.",
    "ignored_tool_errors": "The agent appears to ignore a tool error and claim success anyway.",
    "premature_termination": "The agent appears to stop before completing required outcomes.",
    "unsupported_success": "The agent claims success without sufficient evidence.",
    "lost_context": "The agent appears to lose or repeat context mid-task.",
    "semantic": "The agent behavior appears semantically wrong.",
}


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--provider", choices=["echo", "openai"], default="echo")
    parser.add_argument("--model", default="gpt-4o-mini")
    args = parser.parse_args()

    if args.provider == "echo":
        provider = EchoProvider()
        print("*** STUB RESULTS: EchoProvider is a deterministic test stub, not a model. ***\n")
    else:
        provider = OpenAIProvider(model=args.model)
        print(f"Using OpenAI provider, model={args.model}\n")

    judge = LLMJudge(provider)
    rows = []
    for path in FIXTURES:
        bundle = load_trace_bundle(path)
        for trace in bundle.traces:
            findings = run_detectors(trace)
            categories = sorted({f.category for f in findings})
            for cat in categories or ["semantic"]:
                suspicion = SUSPICIONS.get(cat, SUSPICIONS["semantic"])
                try:
                    result, finding = judge.evaluate_suspicion(trace, suspicion)
                except Exception as exc:  # noqa: BLE001
                    rows.append({"fixture": path.name, "category": cat, "error": str(exc)})
                    continue
                rows.append({
                    "fixture": path.name,
                    "run_id": trace.metadata.run_id,
                    "category": cat,
                    "judge_verdict": result.verdict if result else None,
                    "cited_valid_steps": result.cited_step_ids if result else [],
                    "status": finding.status.value if finding else None,
                    "needs_human_review": result.needs_human_review if result else None,
                })

    agree = sum(1 for r in rows if r.get("judge_verdict") == "supported" and r.get("status") in ("semantic_suspect", "unresolved"))
    total = len(rows)
    print(f"{'fixture':34s} {'category':22s} {'verdict':12s} {'status'}")
    for r in rows:
        print(f"{r.get('fixture', '?'):34s} {r.get('category', '?'):22s} "
              f"{str(r.get('judge_verdict')):12s} {r.get('status')}")
    print(f"\nCases: {total}, flagged for review (supported->suspect/unresolved): {agree}")
    print("Reminder: judge self-reported confidence is recorded as UNCALIBRATED and is never")
    print("treated as a probability; judge output never controls the CI gate alone.")

    OUT.parent.mkdir(parents=True, exist_ok=True)
    OUT.write_text(json.dumps(
        {"provider": args.provider, "stub": args.provider == "echo", "rows": rows},
        indent=2, default=str), encoding="utf-8")
    print(f"Wrote {OUT}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
