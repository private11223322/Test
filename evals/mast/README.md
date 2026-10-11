# MAST evaluation harness

Evaluates Failsafe detectors against MAST (Multi-Agent System Failure
Taxonomy) labels.  See `docs/mast_mapping.md` for the label mapping and
correspondence analysis.

## Status

- **External MAST benchmark: BLOCKED** — the dataset is not bundled and could
  not be downloaded in the build environment (restricted network).  The
  harness prints `EXTERNAL_BENCHMARK_STATUS: BLOCKED` and evaluates only on
  the documented fixtures in this directory.
- **Fixture evaluation: IMPLEMENTED** — demonstrates the full measurement
  pipeline (mapping, matching, confusion matrices) on hand-labelled cases.

## Running

```bash
python evals/mast/harness.py                 # fixtures only; reports BLOCKED
python evals/mast/harness.py --dataset PATH  # real MAST traces, when available
```

## Plugging in the real dataset

1. Obtain the MAST dataset from the paper's release (arXiv:2503.13657).
2. Convert each annotated trace to the fixture bundle format used here
   (`{"traces": [...], "expected_findings": [{"category": "...", "run_id":
   "...", "step_ids": [...]}]}`) using the trace converter in
   `failsafe.ingestion.normalize` for OpenInference-style spans.
3. Map the source label to the Failsafe category **only** where
   `docs/mast_mapping.md` marks the correspondence "direct" (or "partial" for
   the exact-match subset).
4. Run with `--dataset`.  The harness will report precision/recall/F1 per
   mapped label with sample counts and class imbalance.

## What is never done here

- No benchmark numbers are invented or extrapolated.
- Detectors without a genuine MAST counterpart are excluded from scoring.
- Unclear annotations are counted as `uncertain` and reported separately,
  not silently dropped.
