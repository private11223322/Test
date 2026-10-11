# Detector contracts

Every detector implements the documented interface in
`failsafe.detectors.base.Detector`:

```python
class Detector(ABC):
    id: str                    # registry key
    category: str              # finding category
    description: str           # what it catches
    known_limitations: list[str]

    def __init__(self, config: DetectorConfig | None = None): ...
    def detect(self, trace: Trace) -> list[DetectorFinding]: ...   # deterministic
```

Findings always carry **evidence with exact step IDs** and a status from:
`verified` · `rule_triggered` · `semantic_suspect` · `unresolved` ·
`dismissed` · `confirmed_by_human` · `rejected_by_human`.

New detectors are added by subclassing and `@registry.register` — no other
component changes.

Configuration lives in `DetectorConfig` (`failsafe.json` → `detectors`, or
`FAILSAFE_DETECTORS__*` env vars).

---

## A. `repeated_actions`

**Catches.** Tool calls repeated with normalized identical arguments
(sorted keys, collapsed whitespace).

**Thresholds.** `repeat_min_count` (default 3) — escalation threshold;
`repeat_normalize_args` (default true); `repeat_window` (default null =
whole run).

**Behavior.** Retries that directly follow a failed identical call are marked
`is_retry` and excluded from the escalation count (they belong to
`excessive_retries`) but remain in evidence. Below `repeat_min_count` the
finding is INFO-severity, rule-triggered — a duplicate alone is never a
confirmed failure.

**Tests.** Positive (5 identical calls → HIGH) · negative (distinct args;
single call) · boundary (2 calls = INFO; threshold config) · retry-after-fault
excluded from escalation.

**Limitations.** Legitimate intentional repeats are flagged for review;
semantically equivalent but structurally different arguments are not grouped.

## B. `excessive_retries`

**Catches.** Consecutive identical *failed* operations beyond `retry_budget`
(default 3).

**Thresholds.** `retry_budget`. Severity HIGH when failures ≥ budget + 2.

**Tests.** Positive (5 failures) · negative (3 failures within budget;
success resets the chain) · boundary (4 vs budget 3) · configurable budget.

**Limitations.** A retry with changed arguments is a different operation;
no cross-run tracing.

## C. `ignored_tool_errors`

**Catches.** Failed tool results not followed by a retry, fallback,
escalation, or explicit failure report.

**Thresholds.** `ignored_error_allow_actions` (default
retry/fallback/escalate/report_failure); `require_failure_report` (default
true); per-task override via `task.tool_policy`.

**Tests.** Positive (error then "everything is fine!") · negative (retry;
fallback tool; explicit failure report; policy disabled) · ambiguous case
emits RULE_TRIGGERED, not a confirmed failure.

**Limitations.** Heuristic recognition of fallback/escalation; unusual valid
strategies may be flagged. Not evaluable against MAST labels (no genuine
counterpart).

## D. `premature_termination`

**Catches.** Runs that end while `task.required_outcomes` lack structural
evidence. Deterministic outcome types: `tool_called`, `tool_called_with`,
`tool_success`, `response_contains`, `step_count_min`.

**Thresholds.** `premature_require_task_spec` (default true — without a task
spec the detector stays silent rather than guessing).

**Tests.** Positive (unmet outcome) · negative (all outcomes evidenced) ·
multiple unmet outcomes → one finding each · no-task-spec silence.

**Limitations.** Structural only; semantic completion needs the optional
judge. Unknown outcome types are treated as unsatisfiable (loudly).

## E. `unsupported_success`

**Catches.** Final responses that claim success without evidence: required
outcomes unsatisfied (RULE_TRIGGERED, HIGH), or unrecovered tool failures
(UNRESOLVED, LOW — needs review).

**Thresholds.** `success_claim_max_unverified`. Explicit
`metadata.claimed_success` overrides lexical detection.

**Negation handling.** An explicit failure report ("could not be completed",
"failed to update", "no changes were made") is never read as a success
claim. Failures later recovered by a successful retry with identical
arguments do not trigger review findings.

**Tests.** Positive (claim without evidence) · negative (honest failure
report; negation contexts) · ambiguous (claim after errors → UNRESOLVED) ·
recovered-failure suppression.

**Limitations.** Lexical claim detection; unusual phrasings may be missed.

## F. `lost_context`

**Catches.** (1) Exactly repeated LLM/retrieval requests
(`context_repeat_min_count`, default 2). (2) Structurally suspicious
re-asking → UNRESOLVED, LOW, for semantic judging.

**Tests.** Positive (identical prompts ×2) · negative (distinct prompts) ·
boundary (min count) · ambiguous re-ask → UNRESOLVED.

**Limitations.** Exact matching only for repeats; paraphrased repeats and
true context loss need the semantic judge. MAST correspondence is partial
(exact-match subset only).

---

## Determinism contract

Same trace + same `DetectorConfig` ⇒ byte-identical findings (ids, evidence,
severities). Enforced by `test_run_detectors_deterministic`. Detector output
never depends on LLM calls, wall-clock time, or storage state.
