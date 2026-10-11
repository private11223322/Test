# Baselines: what existing tools do and where Failsafe fits

Claims below reflect the tools' documented capabilities as of October 2026
(vendor docs and comparison surveys, e.g. Arize's LLM evaluation platform
comparison, GoodEye Labs' 2026 tool roundup, Latitude's observability
comparison). Verify against current vendor documentation before relying on
them; nothing here claims novelty for combining familiar libraries.

| Capability | Langfuse | LangSmith | Arize Phoenix | Promptfoo | DeepEval | Inspect AI | **Failsafe** |
|---|---|---|---|---|---|---|---|
| Trace capture / observability | OSS tracing, sessions | LangChain/LangGraph-native | OTel-native | Prompt/test runs | pytest-based | run logs | Normalized traces (native + OpenInference/OTLP ingest) |
| Offline LLM evals | manual/annotation workflows | datasets + evaluators | LLM-as-judge | prompt matrices, red teaming | 60+ metrics incl. agent metrics | scorers | Optional judge + deterministic detectors |
| Online evals on production traffic | partial | yes | limited | — | — | — | out of scope (local/CI tool) |
| **Deterministic failure detectors with evidence step IDs** | — | — | — | — | — | — | **6 detectors, registry-extensible** |
| **Controlled tool-fault injection (8 fault kinds, seeded)** | — | — | — | fault injection via prompts/red team | — | solvers can fail tools ad hoc | **seeded schedules + record/replay** |
| **Oracle-verified task outcomes (world state, not claims)** | scores | evaluators | evaluators | assertions | assertions | scorers | **executable oracle checkers + synthetic worlds** |
| **Recovery / silent-failure metrics with explicit denominators** | — | — | — | — | — | — | **recovery vs safe-handling vs silent-failure** |
| **Regression-test generation from observed failures** | — | — | — | test reuse | — | — | **validated tests, failing→passing measured** |
| CI gate (deterministic) | via API | yes | via OTel pipelines | `promptfoo eval` in CI | pytest in CI | `inspect eval` in CI | **gate that LLM output can never control** |
| Multi-agent investigation workflow | — | Insights/assistants | — | — | — | — | **bounded LangGraph agent (advisory)** |

(— = not a documented native capability; tools can often be extended to
approximate it.)

## Positioning

- **Langfuse / LangSmith / Phoenix** are observability and evaluation
  *platforms*: they excel at tracing production traffic and scoring outputs.
  Failsafe is a reliability *testing* tool: it deliberately injects failures,
  verifies outcomes against world state, and turns failures into regression
  tests. It complements those platforms (ingesting OpenInference spans from
  them) rather than replacing them.
- **Promptfoo** tests prompts/configurations across matrices with assertions;
  its red-teaming covers prompt-level faults. Failsafe's fault model targets
  the *tool layer* (timeouts, stale data, schema drift, rate limits) and
  measures recovery behavior.
- **DeepEval** offers metric breadth in pytest, including agent metrics;
  its judgments are metric scorers on outputs. Failsafe's detectors are
  structural and evidence-citing, and its gate never depends on an
  uncalibrated judge.
- **Inspect AI** (UK AISI) is a capable agent-evaluation harness (solvers,
  tools, scorers, sandboxes) used for model evaluations. Failsafe focuses on
  developer-facing reliability regression: seeded fault schedules,
  record/replay CI determinism, and generated regression tests that reproduce
  observed defects.

## What Failsafe actually contributes

Not novelty of parts — LangGraph, SQLAlchemy, Pydantic, Streamlit, and
LLM-as-judge are all established. The contribution is the **integration
around evidence-backed reliability testing**:

1. Deterministic failure detectors that cite exact trace step IDs and label
   their own uncertainty.
2. Controlled, seeded tool-fault injection with record/replay, wired to
   oracle-verified recovery measurement (recovery ≠ safe-handling ≠
   silent-failure).
3. Regression tests generated from observed failures, validated before
   acceptance, with a measured failing→passing reproduction rate.
4. A CI gate with an explicit authority model: deterministic criteria decide;
   LLM findings and the investigation agent are advisory; human review is
   authoritative.

Claims 1–4 are exercised by this repository's tests and evaluation scripts
(`evals/`); see `docs/implementation_status.md` for measured results.
