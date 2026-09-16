# LAM-JEPA seed-level ablation evidence addendum — 16 September 2026

## Scope

This addendum closes one paper-facing evidence gap identified by the external review: the planner and target-path mechanism summaries should be accompanied by seed-level integer outcomes, because five-seed bootstrap intervals can hide that an aggregate delta is driven by only one or two collapsed runs.

This document does **not** alter the frozen ARC protocol, seeds, thresholds, data, model, locked-test policy, or scientific conclusion. It derives only exact integer counts from retained five-seed validation accuracies over the frozen 295-row validation cohort.

Canonical scientific source: `760aa7f9a73a177d5ff4ba7eb470f7e68ace63cb`.

Publication branch base: `df9a9f8e6955326af2d0b879688eeca651ad73b3`.

Primary retained sources:

- `experiments/repro_wave_2026_08_13/INDEPENDENT_AUDIT.md`
- `experiments/repro_wave_2026_08_12/metrics.json`
- `paper/EXTERNAL_REVIEW_VQ_COLLAPSE_CORRECTION_20260831.md`
- `paper/main.tex`
- `CLAIM_LEDGER.md`

## Exact seed-level counts

Each seed is evaluated on 295 retained validation rows. The retained accuracies therefore map to exact integer correct counts:

| Seed | Full correct | `no_planner` correct | Full − `no_planner` | `no_target` correct | Full − `no_target` | Shuffled-label correct |
|---:|---:|---:|---:|---:|---:|---:|
| 1 | 71 | 71 | 0 | 71 | 0 | 78 |
| 2 | 78 | 78 | 0 | 78 | 0 | 71 |
| 3 | 78 | 71 | +7 | 83 | −5 | 83 |
| 4 | 71 | 71 | 0 | 71 | 0 | 78 |
| 5 | 78 | 78 | 0 | 83 | −5 | 78 |

The corresponding retained mean effects are exactly consistent with the manuscript summaries:

- planner contrast: `(+7) / (5 × 295) = +0.0047457627` mean accuracy;
- target-path contrast: `(-10) / (5 × 295) = -0.0067796610` mean accuracy, up to the stored floating-point representation;
- neither clears the frozen `+0.01` mean-effect requirement;
- both retained confidence intervals include zero at a boundary and therefore fail the frozen contribution criterion.

## Why the integer table matters

The external review established that the reviewed retained runs are constant classifiers: each condition/seed predicts one class for all 295 validation examples, and the examined quantized path collapses distinct pre-quantizer latents to one VQ code per run. Under that collapse state, the apparent planner/target differences above are not evidence of robust input-conditional mechanism use.

The planner mean delta is produced entirely by seed 3, where the full and `no_planner` conditions collapse onto classes whose validation base-rate counts differ by seven examples. Seeds 1, 2, 4, and 5 have zero full-minus-`no_planner` difference.

The target-path mean delta is produced by seeds 3 and 5, where `no_target` reaches a constant-class base-rate count five examples higher than full in each seed. Seeds 1, 2, and 4 have zero full-minus-`no_target` difference.

This seed-level decomposition strengthens the conservative interpretation already required by the external review:

> Planner and EMA-target contributions are unsupported under the frozen protocol. The retained deltas should not be interpreted as mechanism effects because the reviewed runs are collapsed constant predictors and the observed integer differences are sparse across seeds.

## Manuscript-ready insertion

Recommended insertion immediately after the existing mechanism-ablation table in `paper/main.tex`:

> Seed-level integer outcomes make the five-seed limitation explicit. Over the fixed 295-row validation set, full-minus-`no_planner` correct-count differences were `[0, 0, +7, 0, 0]`, while full-minus-`no_target` differences were `[0, 0, -5, 0, -5]`. Thus the planner aggregate is driven by a single seed and the target-path aggregate by two seeds. Because the externally reviewed retained runs are constant classifiers, these sparse count shifts are consistent with changes in the class selected under collapse rather than evidence of stable input-conditional planner or target-path benefit.

## Claim boundary

Allowed:

- exact seed-level count decomposition for the frozen validation cohort;
- statement that H2 and H3 remain unsupported;
- statement that the planner aggregate is driven by one seed and the target-path aggregate by two seeds;
- statement that the external review found constant-classifier collapse and single-code VQ collapse in the reviewed retained runs.

Not allowed:

- general claims that planners are useless;
- general claims that EMA target encoders are useless;
- general claims that vector quantization is harmful;
- use of the locked confirmatory ARC test;
- retuning or seed replacement to seek a positive mechanism result;
- treating seed-level sparsity as proof of causal mechanism absence outside this frozen study.

## Publication effect

This addendum does not change the headline result. It makes the negative mechanism result more transparent and directly addresses the external review request to pair five-seed bootstrap summaries with seed-level integer differences.
