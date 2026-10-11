# Failsafe architecture

Three layers with a strict authority model: the deterministic core decides
releases; semantic evaluation and agentic investigation advise.

```
┌────────────────────────────────────────────────────────────────────┐
│ Layer C — Agentic investigation (optional, LangGraph)              │
│   InvestigationAgent: explicit StateGraph, bounded loops, budgets  │
│   11-step workflow, 11 allowlisted tools, authorized replay only   │
│   Output: evidence-backed report + proposed regression test        │
│   AUTHORITY: advisory only — never the sole release authority      │
├────────────────────────────────────────────────────────────────────┤
│ Layer B — Semantic evaluation (optional, LLM judge)                │
│   Structured Pydantic output; citations verified against the trace │
│   Verdicts land as SEMANTIC_SUSPECT / UNRESOLVED / DISMISSED       │
│   Human review is first-class; uncalibrated confidence is labelled │
│   AUTHORITY: can never silently override verified findings,        │
│              can never control the CI gate                         │
├────────────────────────────────────────────────────────────────────┤
│ Layer A — Deterministic core (authoritative, offline)              │
│   ingestion → normalization → detectors → evidence                 │
│   fault injection → oracles → metrics → reports → CI gate          │
│   AUTHORITY: gate decisions come from here only                    │
└────────────────────────────────────────────────────────────────────┘
```

## Data flow

```
trace JSON (failsafe | OpenInference/OTLP spans)
   │  secret redaction, size limits, schema + semantic validation
   ▼
normalized Trace (Pydantic: RunMetadata, Step, ToolCall, ToolResult,
                  FaultInjectionRecord, TaskSpec)
   │
   ├─► DetectorRegistry ──► DetectorFinding[] (evidence: step IDs, status)
   │
   ├─► ToolFaultInjector (seeded FaultSchedule, InjectionLog)
   │        └─ record-and-replay Cassette for deterministic CI
   │
   ├─► OracleEngine (named checkers over world state + trace)
   │        └─ OracleOutcome (independent of agent claims)
   │
   ├─► Metrics (rates with explicit denominators, confusion matrices)
   │
   ├─► Repository (SQLite/PostgreSQL: runs, steps, findings, faults,
   │              oracles, regression tests, evaluations, reports)
   │
   ├─► Reporting (HTML + Markdown: evidence-labelled findings)
   │
   └─► CI gate (deterministic criteria → pass/fail + reasons)
```

## Package layout

```
src/failsafe/
├── cli.py                 # click CLI; documented exit codes
├── config.py              # Pydantic settings; env/file/override merge
├── schemas/               # trace, findings, faults, oracles, regression tests
├── ingestion/             # loader, OpenInference normalize, redaction, limits
├── recording/             # TraceRecorder for working agents
├── detectors/             # base interface + registry + 6 built-ins
├── injection/             # fault primitives, injector, record/replay
├── evaluation/            # worlds, oracle checkers, tasks, runner, metrics,
│                          # adaptive/fixed campaigns
├── judges/                # provider abstraction + LLM judge + human review
├── agents/                # investigation tools, LangGraph workflow,
│                          # regression generation, sample agents
├── storage/               # SQLAlchemy models + repository (separate from logic)
├── reporting/             # report payload, Jinja2 HTML/Markdown templates
└── ci/                    # deterministic gate
```

## Key design decisions

1. **Detectors never touch storage.** The repository layer persists; detectors
   are pure functions `Trace -> [DetectorFinding]` — deterministic and
   side-effect free.
2. **Oracle checkers are named and serializable.** Regression tests and JSON
   task files reference checkers by name (`world_record_equals`,
   `trace_tool_max_calls`, `detector_quiet`, `custom`, ...); unknown names
   fail loudly. The `custom` checker uses constrained operators over the
   world snapshot — never code execution.
3. **World kinds are serializable.** Each task stamps `world_kind` into its
   spec metadata so generated regression tests rebuild the exact original
   world.
4. **Recovery ≠ safe handling.** Recovery requires a triggered fault +
   oracle-verified completion. Safe handling = explicit failure report that
   the oracle verifies was correct. Injection failures never count as
   recoveries.
5. **Budgets are hard.** `BudgetConfig` (iterations, tool calls, LLM calls,
   seconds, cost, tokens) bounds every agentic loop; LangGraph's own
   recursion limit is set above the budget so the budget wins cleanly.
6. **The LangGraph state is explicit.** `InvestigationState` is a TypedDict
   with a `done` action log; the planner (heuristic, deterministic, or LLM)
   selects from a fixed allowlist; unknown actions fail the run.

## Security model

| Threat | Mitigation |
|---|---|
| Prompt injection via trace content | Judge prompt declares trace content untrusted data; instructions inside evidence are never followed; investigation tools return data only |
| Secret leakage | Redaction on ingestion (key patterns + value patterns: sk-, Bearer, AKIA, JWT, key=value); tests assert secrets never enter stored traces |
| Unbounded loops / API spend | Hard budgets on iterations, tool calls, wall time, tokens, estimated cost; langgraph recursion limits |
| Unsafe shell / arbitrary code | No shell tools exist; the `custom` oracle checker is operator-constrained; generated tests are data, validated before execution, and run only through the controlled runner against synthetic worlds |
| Unauthorized filesystem / network | Investigation tools access only in-memory traces + the repository; replay requires `authorized_replay=True`; external services require explicit env configuration |
| Malicious generated tests | RegressionTestValidator rejects unknown checkers, unknown fault kinds, missing oracles; execution is confined to the synthetic world runner |
| Real-world side effects | Synthetic tools by default; `allow_real_systems` defaults to false and is not sufficient alone |

## Interoperability

Traces are ingested from the Failsafe native JSON format or from
OpenInference / OpenTelemetry-GenAI span format (`openinference.span.kind`,
`tool.*`, `llm.*` attributes, OTLP `resourceSpans` shape). Unknown attributes
are preserved; missing values stay `None` and are never invented.
