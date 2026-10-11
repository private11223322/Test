# Failsafe

**Agentic AI reliability testing framework.** Failsafe answers: *how reliable
is this AI agent, how does it fail, can it recover from tool faults, and can
we automatically test against those failures before releasing it?*

Failsafe detects AI agent failures with deterministic, evidence-backed
detectors; injects controlled tool faults; verifies task outcomes with
oracles; measures recovery behavior; investigates suspicious executions with
a bounded LangGraph agent; generates regression tests from observed failures;
and keeps known reliability failures out of your releases via a deterministic
CI gate.

**The deterministic core runs fully offline — no API keys, no GPU, no Docker.**

---

## Why Failsafe

| Principle | What it means here |
|---|---|
| Evidence before conclusions | Every finding cites exact trace step IDs |
| Deterministic checks first | Six structural detectors; same trace + config → same verdict |
| LLMs are optional and untrusted | Semantic judge findings are *suspicions*, never gate decisions |
| Explicit uncertainty | Findings are labelled: verified / rule-triggered / semantic suspicion / unresolved |
| Oracle-based verification | Task success is verified against world state, never the agent's claim |
| Strict budgets | Iteration, tool-call, time, and cost limits on all agentic components |
| No source modification | Failsafe never edits the target agent's code |
| No fake results | Blocked benchmarks are reported as blocked |

## Install

```bash
# from the repository root
python -m venv .venv && source .venv/bin/activate
pip install -e ".[dev]"                 # deterministic core + tests
pip install langgraph langchain-core    # optional: investigation agent
pip install streamlit                   # optional: trace viewer
pip install openai                      # optional: LLM judge
```

Python ≥ 3.11. SQLite is built in; PostgreSQL is optional
(`pip install -e ".[postgres]"`).

## Quickstart (no credentials required)

```bash
failsafe init                                   # config + .env.example + dirs
failsafe validate-trace examples/traces/sample_clean.json
failsafe analyze examples/traces/sample_clean.json            # gate: PASS
failsafe analyze examples/traces/sample_repeated_actions.json # gate: FAIL (caught)
failsafe replay good task-db-lookup                           # record + verify
failsafe replay buggy task-db-lookup                          # verified silent failure
failsafe evaluate --agent good --output metrics.json          # 13 oracle tasks
failsafe report --all                                         # HTML + Markdown
failsafe test                                                 # regression suite
failsafe inspect --tasks                                      # list oracle tasks
```

**Complete end-to-end demonstration** (init → validate → analyze → replay →
evaluate → investigate → regression failing→passing → reports → tests):

```bash
bash scripts/demo.sh
```

## CLI

| Command | Purpose | Exit codes |
|---|---|---|
| `failsafe init` | Create `failsafe.json`, `.env.example`, data dirs | 0 |
| `failsafe validate-trace PATH` | Schema + semantic validation of a trace | 0 valid, 2 invalid |
| `failsafe analyze PATH` | Deterministic detectors + gate + persistence | 0 gate pass, 1 gate fail, 2 bad input |
| `failsafe replay AGENT TASK` | Execute a sample agent on an oracle task | 0 oracle pass, 1 oracle fail |
| `failsafe evaluate [--suite DIR]` | Run the oracle suite + reliability metrics | 0 all pass, 1 any fail |
| `failsafe report --run-id ID \| --all` | HTML + Markdown reports with evidence | 0 |
| `failsafe test` | Generate + run regression tests | 0 |
| `failsafe inspect [--run-id ID] [--tasks] [--findings] [--limits]` | Inspect runs, tasks, detector limitations | 0 |

Run `failsafe COMMAND --help` for options. Exit code 3 = internal error.

## Sample agents

Four scripted, deterministic agents ship for demos and tests
(`failsafe replay good|buggy|looping|flaky`):

| Agent | Behavior | Used to demonstrate |
|---|---|---|
| `good` | Correct, retries transient faults once, reports failure honestly | Clean control, recovery, safe handling |
| `buggy` | Claims success without doing the work | Verified silent failures |
| `looping` | Repeats identical tool calls | Repeated-action detection |
| `flaky` | Ignores tool errors, claims success anyway | Ignored-tool-error detection |

Plug in your own agent by implementing `run(task, tools, recorder) -> str` —
see `examples/agents/my_agent.py`.

## Architecture (three layers)

```
Layer C  Agentic investigation (optional, LangGraph)
         bounded tool-using workflow: inspect → hypothesize → propose test →
         validate → authorized replay → evidence-backed report
                    │ advisory only — never controls the gate
Layer B  Semantic evaluation (optional, LLM judge)
         structured Pydantic output, verified citations, human-review outcome
                    │ never overrides verified findings
Layer A  Deterministic core (authoritative, offline)
         trace ingestion → detectors + evidence → fault injection →
         oracles → metrics → reports → CI gate
```

See `docs/architecture.md` for details.

## Deterministic detectors

| Detector | Catches | Evidence |
|---|---|---|
| `repeated_actions` | Identical tool calls (normalized args), loop escalation | Exact call step IDs |
| `excessive_retries` | Retry chains beyond the configured budget | Full retry-chain step IDs |
| `ignored_tool_errors` | Failed tools without retry/fallback/escalation/report | The failed result's step ID |
| `premature_termination` | Ending before required outcomes have evidence | Terminal step + unmet outcome |
| `unsupported_success` | Success claims without evidence / after unrecovered errors | Response step + contradicted steps |
| `lost_context` | Exact repeated requests; suspicious re-asking | The repeated request step IDs |

Every detector has a documented interface, configurable thresholds,
positive/negative/boundary/ambiguous tests, and known limitations
(`failsafe inspect --limits`, `docs/detectors.md`). New detectors register
via a registry without touching unrelated code.

## Fault injection

Tool-wrapper injector with seeded schedules: **timeout, explicit error, empty
response, malformed response, stale response, rate limit (with retry-after),
partial result, schema drift**. Every injection is logged. A fixed seed
reproduces the *schedule* — it never pretends to pin live LLM behavior.
Record-and-replay cassettes give deterministic CI replays. Synthetic tools
only by default; real-system injection requires explicit configuration.

## Oracle tasks

13 built-in tasks with executable success checkers: database lookups,
multi-step tool use, controlled-dataset search, correct record updates,
recoverable faults, unrecoverable faults, stale results, duplicate mutations,
rate limiting, notifications, deletes, and multi-record transfers. Task
outcome = oracle verdict over world state + trace structure — never the
agent's final answer. `failsafe inspect --tasks` lists them.

## Reliability metrics

Task success rate · fault recovery rate (verified outcomes only) · safe-handling
rate (explicit correct failure reports — distinct from recovery) · verified
silent-failure rate · injection-failure rate · detector precision/recall/F1 ·
confusion matrices · runtime · LLM usage/cost when available · replay
stability. Every metric ships with its exact denominator definition.

## Reports

Professional HTML + Markdown with run metadata, oracle results, injected
faults, exact evidence step IDs, recovery outcomes, detector metrics,
uncertainty labels, and known limitations. Suspicion is never presented as
fact.

## Streamlit trace viewer

```bash
streamlit run ui/app.py
```

Run listing/filtering, timeline, tool I/O, injected faults, oracle outcomes,
run comparison, investigation reports, and recovery metrics — over the same
backend services, no duplicated logic.

## CI gate

Deterministic and configured: strict mode fails on critical findings, oracle
failures, or verified silent failures; report-only mode never blocks. LLM
judge output can **never** block or approve a build on its own; human-confirmed
findings are authoritative. Workflows: `.github/workflows/ci.yml`
(unit/integration/regression + failing→passing demonstration),
`report-only.yml` (evaluations, artifacts), `llm-judge.yml` (optional).

## Evaluation

```bash
python evals/detector_eval.py        # detector P/R/F1 on labelled fixtures
python evals/oracle_reliability_eval.py
python evals/fault_injection_eval.py
python evals/judge_comparison.py     # stub offline; --provider openai live
python evals/regression_eval.py      # failing→passing reproduction rate
python evals/adaptive_vs_fixed.py    # equal-budget adaptive comparison
python evals/mast/harness.py         # MAST mapping harness
```

MAST label mapping and its limitations: `docs/mast_mapping.md`. The external
MAST benchmark is **blocked** in this environment (dataset not accessible) —
no benchmark numbers are invented. Measured results from the scripts above
are in `evals/results/` and summarized in `docs/implementation_status.md`.

## Docker

```bash
docker build -t failsafe .
docker run --rm -v $PWD/data:/data failsafe evaluate --agent good
docker compose up viewer    # Streamlit UI on :8501 (optional)
```

The base project needs none of this — everything runs locally.

## Security

All traces, tool outputs, and generated test cases are untrusted input:
secret redaction on ingestion, prompt-injection containment in the judge
(evidence-only instructions), allowlisted investigation tools (no shell, no
arbitrary code, no filesystem writes), explicit replay authorization, strict
budgets against unbounded loops and API spend, and never any credentials in
the repository. See `docs/architecture.md#security`.

## Documentation

| Document | Contents |
|---|---|
| `docs/implementation_status.md` | Every component with its **verified** status |
| `docs/architecture.md` | Three layers, data flow, security model |
| `docs/detectors.md` | Detector contracts, thresholds, limitations |
| `docs/mast_mapping.md` | MAST label mapping + correspondence analysis |
| `docs/baselines.md` | Comparison with Langfuse, LangSmith, Phoenix, etc. |
| `.env.example` | All configuration, with the optional parts commented |

## License

Apache-2.0.
