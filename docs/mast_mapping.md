# MAST label mapping: source taxonomy → Failsafe detectors

## Source

**MAST (Multi-Agent System Failure Taxonomy)** from *"Why Do Multi-Agent LLM
Systems Fail?"* (arXiv:2503.13657, UC Berkeley RDI).  The taxonomy was derived
from manual annotation of 200+ execution traces from multiple MAS frameworks,
with 14 failure modes in 3 categories (inter-annotator Cohen's κ = 0.88):

| Category | Share of failures | Examples of modes (as named in the paper / summaries) |
|---|---|---|
| **Specification & system design** | ~41.8% | Disobey task specification; disobey role specification; repeat previously completed steps; lose conversation history; fail to recognize task completion (missing termination); task misinterpretation; poor task decomposition |
| **Inter-agent misalignment** | ~36.9% | Conversation reset; fail to ask for clarification; ignore another agent's input; information withholding; task derailment; reasoning–action mismatch |
| **Task verification & termination** | ~21.3% | Premature termination (~6.2%); no / incomplete verification (~8.2%); incorrect verification (~9.1%) |

> **Naming caveat.** Exact mode labels differ slightly across paper versions
> and secondary summaries. The mapping below uses the mode *definitions*, and
> each row states whether the correspondence is **direct** (genuine overlap
> with a Failsafe detector definition), **partial** (the detector sees a
> structural shadow of the mode), or **out of scope** (requires multi-agent
> semantics Failsafe's deterministic core does not model).

## External benchmark status: BLOCKED (environment)

The MAST dataset itself could **not** be downloaded in this environment
(network access is restricted to GitHub/npm/PyPI hosts; the dataset is
distributed with the paper's release).  **No benchmark numbers from the MAST
dataset are reported anywhere in this repository.**  The evaluation harness
(`evals/mast/harness.py`) runs on documented fixtures only and prints
`EXTERNAL_BENCHMARK_STATUS: BLOCKED` until real MAST traces are supplied by a
user with dataset access.  See `evals/mast/README.md` for how to plug the
real dataset in.

## Mapping table

| # | MAST mode (definition) | Category | Failsafe detector / component | Correspondence | Notes |
|---|---|---|---|---|---|
| 1 | Disobey task specification | Spec | `premature_termination` + oracle `required_outcomes` | **direct** | Unmet required outcomes are exactly "did not do what the task asked". |
| 2 | Disobey role specification | Spec | — | out of scope | Role constraints are multi-agent constructs; single-agent traces carry no role assignment to check. |
| 3 | Repeat previously completed steps / loops | Spec | `repeated_actions`, `excessive_retries`, `lost_context` (exact repeats) | **direct** | Exact repeated tool calls and repeated prompts with identical normalized arguments. |
| 4 | Lose conversation history / lost in conversation | Spec | `lost_context` (exact repeats only) | **partial** | Exact repeated *requests* are detected deterministically; paraphrased context loss needs the optional semantic judge (`judges/llm_judge.py`). |
| 5 | Fail to recognize task completion / missing termination | Spec | `premature_termination` (inverse), `unsupported_success` | **partial** | Failsafe sees "ended before outcomes were met" and "claimed success without evidence"; it does not model "kept going after done" as a distinct structural signal beyond `repeated_actions`. |
| 6 | Task misinterpretation / ambiguous specification | Spec | oracle failures + `unsupported_success` | **partial** | Misinterpretation is inferred from oracle divergence; the root cause (spec ambiguity) is not deterministically attributable. |
| 7 | Conversation reset | Inter-agent | `lost_context` (exact repeated requests) | **partial** | A reset typically re-sends identical earlier requests — that exact case is caught; subtle resets are not. |
| 8 | Fail to ask for clarification | Inter-agent | — | out of scope | Requires dialogue-act semantics and multi-agent context. |
| 9 | Ignore another agent's input | Inter-agent | — | out of scope | Multi-agent interaction is not modeled in the single-agent trace core. |
| 10 | Information withholding | Inter-agent | — | out of scope | Same as above. |
| 11 | Task derailment | Inter-agent | `premature_termination` + `ignored_tool_errors` | **partial** | Derailment that abandons required outcomes is caught as unmet outcomes / ignored errors; social derailment is not. |
| 12 | Reasoning–action mismatch | Inter-agent | `unsupported_success` (claims vs evidence), optional `llm_judge` | **partial** | Deterministic evidence-claim mismatch only; "said plan A, did B" needs semantic comparison (judge). |
| 13 | Premature termination | Verification | `premature_termination` | **direct** | Exact match to the detector definition. |
| 14 | No / incomplete verification | Verification | `unsupported_success`, oracle suite | **direct** (as *task-level* verification) | Failsafe *provides* the missing verification (oracles); detecting that a *system* skipped its own verifier is multi-agent scope. |
| 15 | Incorrect verification | Verification | oracle failure + `silent_failure` metric | **direct** (as measured outcome) | Failsafe measures "claimed/accepted success while the oracle verifies failure" = verified silent failure. |

> Row count: the paper reports 14 modes; several summaries group/split modes
> differently (e.g. separating "disobey task" vs "disobey role", or "no
> verification" vs "incorrect verification").  The table above is organized by
> **definition**, and every evaluation in `evals/mast/harness.py` states which
> rows it exercises.

## What is actually evaluated on what

| Detector | Evaluated against MAST labels? | Why |
|---|---|---|
| `repeated_actions` | only if dataset present (label: step repetition) | Direct correspondence. |
| `excessive_retries` | only if dataset present (label: step repetition subtype) | Partial: MAST does not separate retry budgets. |
| `ignored_tool_errors` | **no** | No MAST mode covers single-agent tool-error handling. |
| `premature_termination` | only if dataset present (label: premature termination) | Direct correspondence. |
| `unsupported_success` | only if dataset present (labels: incorrect verification, no verification) | Direct for verification labels. |
| `lost_context` | only if dataset present (labels: lost history, conversation reset) — exact-match subset only | Partial correspondence. |

**We do not claim that all six detectors can be evaluated on MAST.**  Two
detectors (`ignored_tool_errors`, `excessive_retries` as a budget concept)
have no genuine MAST label and are excluded from MAST scoring by design.

## Uncertainty statement

- Category shares (41.8 / 36.9 / 21.3%) and κ = 0.88 are from the paper's
  reported results, not re-measured here.
- Mode naming varies between paper versions and summaries; the mapping uses
  definitions, and rows are marked direct/partial/out-of-scope accordingly.
- No MAST dataset numbers appear in this repository because the dataset was
  not accessible in this environment.
