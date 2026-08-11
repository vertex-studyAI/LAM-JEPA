# ARC v5 negative-result analysis

This document describes a **read-only descriptive analysis** for the already-retained ARC-v5 repaired-validation rows.

It exists to help explain the repository's preserved `VALID_NEGATIVE_OR_INCONCLUSIVE_VALIDATION` outcome without reopening the frozen protocol, tuning thresholds on validation, or accessing the locked confirmatory ARC test split.

## Analyzer

```bash
python scripts/analysis/analyze_arc_v5_negative_result.py \
  --results path/to/arc-v5-repaired-validation.json \
  --out outputs/arc-v5-negative-result-slices.json
```

The input must be the retained result-package shape emitted by `scripts/benchmark/run_arc_v5_repaired_validation.py` and accepted by `scripts/ci/verify_arc_v5_repaired_validation.py`.

The analyzer fails closed unless:

- all four frozen conditions are present;
- every condition has the same seed set;
- every seed preserves the same row IDs and order;
- labels are identical across conditions and seeds;
- the result package records `test_accessed=false`;
- the result package records `research_complete=false`.

## Output

The generated JSON contains only descriptive diagnostics derived from retained IDs, labels, predictions, conditions, and seeds:

- per-seed accuracy recomputed from retained rows;
- prediction support and largest predicted-class share;
- per-true-label accuracy and prediction histograms;
- repaired-vs-legacy `fixed`, `regressed`, `both_correct`, and `both_wrong` transition counts;
- repaired-vs-no-quantizer transition counts;
- per-item repaired-model stability across seeds;
- hardest retained items ordered by the number of repaired seeds that answered them correctly.

These slices are intended for mechanism/error analysis. They are **not** new model-selection criteria and must not be used to weaken, reinterpret, or retroactively replace the frozen decision rules.

## Scientific boundary

Running this analyzer does **not**:

- change the frozen ARC-v5 protocol;
- change any success/failure threshold;
- authorize validation hyperparameter selection;
- authorize seed searching;
- access or authorize the locked confirmatory ARC test split;
- establish generalization, quantization benefit, model superiority, educational effectiveness, novelty, or `RESEARCH_COMPLETE`.

A negative or inconclusive slice remains a valid result. If the diagnostics motivate a genuinely new hypothesis, that hypothesis must be versioned and preregistered separately before new confirmatory data are accessed.

## Verification

The lightweight workflow `ARC V5 Negative Result Analysis QA` compiles the analyzer and runs fail-closed unit tests without downloading ARC data.

Local verification:

```bash
python -m py_compile \
  scripts/analysis/analyze_arc_v5_negative_result.py \
  tests/test_arc_v5_negative_result_analysis.py

python -m unittest tests/test_arc_v5_negative_result_analysis.py -v
```
