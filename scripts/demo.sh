#!/usr/bin/env bash
# Failsafe end-to-end demonstration.
#
# Runs the complete documented workflow with NO API credentials:
#   1. init project files
#   2. validate example traces (native + OpenInference)
#   3. analyze clean and failing traces (deterministic detectors)
#   4. replay sample agents (record + oracle + gate)
#   5. evaluate the oracle task suite with fault injection
#   6. run the investigation agent + regression generation (failing -> passing)
#   7. generate HTML + Markdown reports
#   8. run the automated test suite
#
# Usage:  bash scripts/demo.sh
# Exit code 0 means every stage behaved as documented.
set -euo pipefail

cd "$(dirname "$0")/.."
ROOT="$(pwd)"
export FAILSAFE_LLM_ENABLED=false

PY="${PYTHON:-python}"
command -v "$PY" >/dev/null || PY=python3
if [ -x "$ROOT/.venv/bin/python" ]; then PY="$ROOT/.venv/bin/python"; fi
if command -v "$ROOT/.venv/bin/failsafe" >/dev/null 2>&1; then
  FS="$ROOT/.venv/bin/failsafe"
elif command -v failsafe >/dev/null 2>&1; then
  FS="failsafe"
else
  FS="$PY -m failsafe.cli"
fi

hr() { printf '\n== %s ==\n' "$1"; }

hr "1. Init"
rm -rf demo-workspace && mkdir demo-workspace && cd demo-workspace
$FS init

hr "2. Validate example traces"
$FS validate-trace ../examples/traces/sample_clean.json
$FS validate-trace ../examples/traces/sample_openinference.json

hr "3. Analyze traces (clean control must PASS, known regression must FAIL)"
$FS analyze ../examples/traces/sample_clean.json
set +e
$FS analyze ../examples/traces/sample_repeated_actions.json
ANALYZE_FAIL=$?
set -e
[ "$ANALYZE_FAIL" -eq 1 ] || { echo "expected the failing trace to fail the gate"; exit 1; }
echo "known regression was caught (exit code 1) — correct."

hr "4. Replay sample agents (recorded runs + oracle verification)"
$FS replay good task-db-lookup
set +e
$FS replay buggy task-db-lookup
BUGGY_EXIT=$?
set -e
[ "$BUGGY_EXIT" -eq 1 ] || { echo "expected the buggy agent to fail the oracle"; exit 1; }
echo "buggy agent produced a verified silent failure — correct."

hr "5. Evaluate the oracle suite with fault injection"
$FS evaluate --agent good --output metrics_good.json
set +e
$FS evaluate --agent flaky --output metrics_flaky.json --save
FLAKY_EXIT=$?
set -e
[ "$FLAKY_EXIT" -eq 1 ] || { echo "expected the flaky agent to fail some oracles"; exit 1; }
echo "flaky agent produced verified silent failures — correct."

hr "6. Investigation + regression tests (failing -> passing)"
$FS replay looping task-db-lookup || true
$FS test
echo "Regression tests generated from observed failures and executed."

hr "7. Reports (HTML + Markdown with evidence step IDs)"
$FS report --all
ls reports/

hr "8. Automated test suite"
cd ..
$PY -m pytest tests/ -q

hr "Demo complete"
echo "Artifacts: demo-workspace/reports/ , demo-workspace/*.db , metrics_*.json"
echo "View traces interactively:  streamlit run ui/app.py"
