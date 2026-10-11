# Failsafe implementation status

**Last verified:** 2026-10-11 · **Test suite:** 201 passed (pytest) ·
**Lint:** ruff clean · **E2E demo:** `bash scripts/demo.sh` exits 0

Status vocabulary: *Implemented* (code exists) · *Tested* (automated tests
pass) · *Verified end to end* (exercised through the CLI/demo with observed
output) · *Blocked* (environmental limitation, stated precisely).

---

## Component checklist

| # | Component | Status | Evidence |
|---|---|---|---|
| 1 | Installable package (`pyproject.toml`, `failsafe` console script) | **Verified end to end** | `pip install -e .`; CLI used throughout `scripts/demo.sh` |
| 2 | Config system (file/env/overrides, no secrets) | **Tested** | `tests/unit/test_recording_config.py` (7 tests) |
| 3 | Trace schemas (trace/steps/tool calls/results/faults/findings/oracles/evals/regression) | **Tested** | `tests/unit/test_schemas.py` (12 tests) |
| 4 | Trace ingestion: native format, validation errors | **Tested** | `tests/unit/test_ingestion.py`; `failsafe validate-trace` verified in demo |
| 5 | OpenInference / OTLP span normalization | **Tested** | 3 ingestion tests + `examples/traces/sample_openinference.json` validated in demo |
| 6 | Secret redaction + size/step limits | **Tested** | 5 tests incl. "secrets never enter stored traces" |
| 7 | TraceRecorder for working agents | **Tested** | `tests/unit/test_recording_config.py` + regression tests for 2 recorder defects |
| 8 | Detector interface + registry | **Tested** | interface contract tests; duplicate/unknown registry tests |
| 9 | Detector A `repeated_actions` | **Tested** | positive/negative/boundary/retry-exclusion/threshold tests |
| 10 | Detector B `excessive_retries` | **Tested** | positive/negative/boundary/reset/configurable tests |
| 11 | Detector C `ignored_tool_errors` | **Tested** | positive/negative (retry, fallback, report, policy)/ambiguous tests |
| 12 | Detector D `premature_termination` | **Tested** | positive/negative/no-spec/multi-outcome tests |
| 13 | Detector E `unsupported_success` | **Tested** | positive/negative/negation-context/ambiguous/recovered-suppression tests |
| 14 | Detector F `lost_context` | **Tested** | positive/negative/boundary/ambiguous tests |
| 15 | Determinism of detector verdicts | **Tested** | `test_run_detectors_deterministic` |
| 16 | Fault injection: 8 fault kinds, seeded schedules | **Tested** | `tests/unit/test_injection.py` (10 tests) |
| 17 | Injection logging + max-injection caps | **Tested** | injection tests |
| 18 | Record-and-replay cassettes | **Tested** | cassette round-trip, error replay, determinism tests |
| 19 | Synthetic world + tools (db, search, http, email) | **Tested** | `tests/unit/test_evaluation.py` world tests |
| 20 | Oracle checkers (17 named, incl. `custom` operator-constrained) | **Tested** | checker tests incl. unknown-checker and crashing-checker behavior |
| 21 | Oracle task suite (13 tasks, 8 required categories) | **Tested + Verified end to end** | category coverage test; `failsafe evaluate` in demo |
| 22 | Recovery metrics with explicit denominators | **Tested** | metric tests incl. injection-failure-never-counts, recovery-vs-safe-handling |
| 23 | Detector precision/recall/F1 + confusion matrices | **Tested + measured** | metric tests; `evals/detector_eval.py` measured below |
| 24 | SQLite persistence (runs/steps/tool calls/faults/findings/oracles/regression/evals/reports) | **Tested** | `tests/unit/test_storage.py` (8 tests) |
| 25 | PostgreSQL support (via SQLAlchemy URL) | **Implemented** | not exercised here (no PG server in sandbox) |
| 26 | Repository layer separated from detector logic | **Implemented** | detectors have zero storage imports |
| 27 | HTML reports with evidence step IDs + status labels | **Tested** | report tests assert step IDs present, suspicion ≠ confirmed |
| 28 | Markdown reports (no HTML escaping) | **Tested** | report tests + regression test for the escaping defect |
| 29 | CI gate (strict/report-only, deterministic) | **Tested** | 10 gate tests: category/severity filters, oracle/silent failures, human authority, judge non-blocking |
| 30 | CLI: init/validate-trace/analyze/replay/evaluate/report/test/inspect | **Verified end to end** | `tests/integration/test_cli.py` (12 tests) + full demo run |
| 31 | Regression-test generation (validated states, world preservation) | **Tested + measured** | unit tests + `evals/regression_eval.py` measured below |
| 32 | LangGraph investigation agent (explicit state, bounded, 11 tools) | **Tested** | `tests/integration/test_investigation.py` (10 tests): full workflow, budgets, authorization, advisory-only |
| 33 | LLM judge (Pydantic output, citation verification, human review) | **Tested** (stub provider) | `tests/unit/test_judges.py` (11 tests) |
| 34 | LLM judge live OpenAI provider | **Implemented, not verified live** | no API key in this environment; `llm-judge.yml` runs it when configured |
| 35 | Streamlit trace viewer | **Implemented, not interactively verified** | `ui/app.py` import/syntax checked; no browser in sandbox |
| 36 | Adaptive campaigns (DISCOVER→…→REPORT) | **Tested + measured** | campaign code + `evals/adaptive_vs_fixed.py` measured below |
| 37 | Sample agents (good/buggy/looping/flaky) + example agent | **Verified end to end** | used by demo, tests, and evals |
| 38 | Example traces (clean/failing/silent/ignored/OpenInference) | **Verified end to end** | generated from real runs; validated in demo |
| 39 | Evaluation scripts 1–6 | **Implemented + run** | measured results below |
| 40 | MAST label mapping doc + harness | **Implemented + run (fixtures)** | `docs/mast_mapping.md`; harness run below |
| 41 | External MAST benchmark | **Blocked** | dataset not accessible (network restricted to GitHub/npm/PyPI); no numbers invented |
| 42 | GitHub Actions workflows (unit/integration/static/regression/failed→passed demo/report-only/llm-judge) | **Implemented, not executed** | no Actions runner in sandbox; workflows are declarative config |
| 43 | Dockerfile + docker-compose | **Implemented, image build not verified** | Docker daemon not available in sandbox |
| 44 | Documentation (README, architecture, detectors, baselines, MAST, env example) | **Implemented** | this file + `docs/` + README |
| 45 | End-to-end demonstration | **Verified end to end** | `bash scripts/demo.sh` → exit 0 |

---

## Test suite summary (actual)

```
201 passed, 1 warning in ~2.2s     (pytest tests/ -q)
ruff check src tests               → All checks passed!
```

- `tests/unit/` — schemas, ingestion, detectors (positive/negative/boundary/
  ambiguous per detector), injection, evaluation/oracles/metrics, storage,
  reporting, ci gate + regression generation, recording/config, judges.
- `tests/integration/` — CLI workflows, LangGraph investigation.
- `tests/regression/` — 6 tests encoding defects actually found during
  development (recorder kwargs, metadata detach, negation parsing,
  recovery/safe-handling conflation, Markdown escaping, retry escalation).
- The deterministic core is exercised with `FAILSAFE_LLM_ENABLED=false`
  everywhere; no test requires credentials, network, PostgreSQL, or Docker.

## Measured evaluation results (actually run in this environment)

### Detector evaluation (`evals/detector_eval.py`, 6 labelled fixtures)

| Metric | Value | Fraction |
|---|---|---|
| detector_precision | 1.00 | 13/13 |
| detector_recall | 1.00 | 13/13 |
| detector_f1 | 1.00 | — |
| false positives | 0 | 0/13 |

Per-category tp: repeated_actions 1, ignored_tool_errors 1,
premature_termination 7, unsupported_success 3, lost_context 1.
*Interpretation: these fixtures are in-distribution (built from the detector
definitions); they validate the measurement pipeline and the detectors'
contract, not generalization. `tests/unit/test_detectors.py` carries the
nuanced behavior checks.*

### Oracle reliability (`evals/oracle_reliability_eval.py`, 13 tasks × 4 agents)

| Agent | Verified success | Silent failures | Recovered | Safe-handled |
|---|---|---|---|---|
| good | 13/13 | 0 | 3 | 2 |
| buggy | 1/13 | 12 | 0 | 0 |
| looping | 2/13 | 0 (wrong/incomplete instead) | 0 | 0 |
| flaky | 1/13 | 12 | 0 | 0 |

good-agent rates: task_success 1.0, fault_recovery 0.6 (3/5 injected runs;
2/5 are safe-handling tasks by design), silent-failure 0.0.

### Fault-injection recovery (`evals/fault_injection_eval.py`, 8 fault kinds)

good agent: recovery_rate 1.0, silent 0.0 across all 8 kinds.
flaky agent: recovery_rate 0.0, silent_failure_rate 1.0. Injections triggered
in 16/16 scheduled runs.

### Regression-test evaluation (`evals/regression_eval.py`)

Generated: **32** · Validated at generation: **32** ·
Reproduce the original defect (fail on buggy agent / detector re-fires):
**32/32 (1.00)** · Pass after the defect is corrected (good agent):
**32/32 (1.00)** — the failing→passing demonstration required by the CI gate.

### Adaptive vs fixed (`evals/adaptive_vs_fixed.py`, equal budgets: 8 scenarios, 8 faults, seed 1337)

Adaptive: 8 failure-producing scenarios of 8 (adaptive policy re-uses
failure-producing fault kinds and explores untried tools). Fixed baseline: 7/8.
Adaptive advantage: +1 scenario. Deterministic synthetic comparison; live LLM
variability out of scope.

### LLM-judge comparison (`evals/judge_comparison.py`)

Stub provider run (offline): 10 cases, all `uncertain` → `unresolved`, zero
gate influence — the safety rails work. **Live model comparison: not measured**
(no credentials in this environment); run `evals/judge_comparison.py
--provider openai` or the `llm-judge.yml` workflow when a key is configured.

### MAST harness (`evals/mast/harness.py`)

`EXTERNAL_BENCHMARK_STATUS: BLOCKED` (dataset inaccessible). Fixture subset
(mappable labels only): 12 expected / 12 predicted, P=R=F1=1.0; 1 prediction
excluded (ignored_tool_errors has no genuine MAST label).

---

## Known limitations (honest list)

1. **Live LLM judging unverified here.** The OpenAI provider and LLM planner
   code paths are implemented and stub-tested; no real model was called
   (no credentials). Judge confidence is never calibrated and is labelled as
   such wherever recorded.
2. **Streamlit viewer not click-tested.** The app imports cleanly and uses the
   backend services; interactive behavior was not verified (no browser).
3. **Docker image build not verified.** Dockerfile and compose are written
   against the documented Python image; no daemon is available in the sandbox.
4. **GitHub Actions not executed here.** Workflow YAML is implemented; it will
   run on push to GitHub.
5. **PostgreSQL untested.** SQLAlchemy models are engine-agnostic; only SQLite
   was exercised.
6. **MAST external benchmark blocked.** Mapping + harness are ready; no
   dataset numbers are reported.
7. **Detector generalization.** Fixture evaluation is in-distribution; the
   detectors are structural and their limitations are documented per detector
   (`docs/detectors.md`, `failsafe inspect --limits`).
8. **Adaptive campaign policy is simple (documented heuristic).** Learned
   scenario generation remains a stretch goal; the state machine, budgets,
   and equal-budget comparison are implemented and measured.
9. **Multi-agent MAST modes (role drift, clarification, information
   withholding) are out of scope** for the single-agent deterministic core;
   `docs/mast_mapping.md` marks each one explicitly.

## Defects found and fixed during development (regression-tested)

1. `TraceRecorder.add_step` rejected `tool_call`/`tool_result` kwargs — every
   tool step corrupted (TypeError swallowed by the runner).
2. Pydantic copied `RunMetadata` on `Trace` construction — `finish()` updated a
   detached object; traces stored `status=unknown`.
3. Success-claim lexicon matched negated phrases ("could not be **completed**").
4. Recovery metric conflated safe-handling (explicit failure reports) with
   recovery.
5. Markdown renderer HTML-escaped evidence quotes (`&#34;`).
6. Retries after injected faults were escalated like genuine action loops.
7. LangGraph dropped the untyped `done` state key — the planner looped until
   budgets exhausted.
8. Generated regression tests lost the original task's world factory and task
   id — they could not reproduce scenarios requiring custom worlds.
9. `failsafe init` wrote a config key `DetectorConfig` rejected.

Each is encoded in `tests/regression/test_known_failures.py` or covered by
the affected unit tests.
