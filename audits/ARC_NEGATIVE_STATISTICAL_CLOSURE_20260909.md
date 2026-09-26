# LAM-JEPA frozen ARC statistical closure — 9 September 2026

## Scope

This audit closes remaining **reporting/reproducibility** gaps in the already-frozen negative/inconclusive ARC-Challenge study. It does not retrain LAM-JEPA, access a new benchmark, inspect the locked ARC confirmatory test, tune a threshold, change a seed, or authorize the separately preregistered successor.

Canonical machine verifier:

```bash
python tools/verify_frozen_negative_arc_statistics.py \
  --out audits/arc_negative_statistical_closure_20260909.json
```

The verifier reads only three previously retained evidence files:

- `experiments/repro_wave_2026_08_13/independent_audit.json`;
- `audits/independent_artifact_audit_2026-08-13.json`;
- `experiments/repro_wave_2026_08_12/metrics.json`.

Scientific source remains `760aa7f9a73a177d5ff4ba7eb470f7e68ace63cb`.

## Benchmark semantics and all-row evaluation

The frozen task is AI2 ARC-Challenge **multiple-choice reasoning**, not the Abstraction and Reasoning Corpus. The protocol retains exactly-four-choice examples under a predeclared feature-only eligibility rule.

The independent verifier records:

- training source rows: 1,119;
- eligible training rows: 1,117;
- training rows actually used: **1,117 / 1,117 eligible**;
- validation source rows: 299;
- eligible validation rows: 295;
- validation rows actually used: **295 / 295 eligible**;
- seeds: `[1,2,3,4,5]`;
- 20 epochs, batch size 32;
- locked confirmatory test evaluated: **false**.

Therefore the reported validation result is an all-eligible-row evaluation. It is not based on a hand-picked validation subset. Excluded source rows are the four-choice eligibility exclusions already preserved by the frozen protocol; they are not post-outcome removals.

## Exact per-seed integer reconstruction

Because every condition is evaluated on the same 295 eligible validation examples, retained floating-point accuracies can be mapped back to exact correct-answer counts. The reconstruction is unique within the retained numerical precision:

| Seed | Full | No planner | No target | Shuffled-label control |
|---:|---:|---:|---:|---:|
| 1 | 71/295 | 71/295 | 71/295 | 78/295 |
| 2 | 78/295 | 78/295 | 78/295 | 71/295 |
| 3 | 78/295 | 71/295 | 83/295 | 83/295 |
| 4 | 71/295 | 71/295 | 71/295 | 78/295 |
| 5 | 78/295 | 78/295 | 83/295 | 78/295 |

The corresponding paired full-minus-ablation count deltas are:

- full minus `no_planner`: `[0, 0, +7, 0, 0]` correct answers;
- full minus `no_target`: `[0, 0, -5, 0, -5]` correct answers.

These reconstruct the retained aggregate effects exactly up to documented floating-point representation tolerance:

- full minus `no_planner`: mean accuracy delta `+0.0047457627`; retained bootstrap 95% interval `[0.0, 0.0142372881]`;
- full minus `no_target`: mean accuracy delta `-0.0067796610`; retained bootstrap 95% interval `[-0.0135593220, 0.0]`.

Both retained intervals include zero and both frozen mechanism criteria remain false. Nothing in this integer-level reconstruction upgrades the planner or EMA-target mechanism claims.

## Five-seed inference limitation

The independent experimental unit is the **training seed**, not the 295 validation questions. There are only five paired training seeds. Treating question-level rows as hundreds of independent model-training replicates would be pseudoreplication.

Accordingly:

- sample standard deviations and paired bootstrap intervals remain useful descriptive uncertainty summaries;
- the five-seed study does not establish broad benchmark-general statistical significance;
- no new p-value is introduced post hoc as a substitute success criterion;
- the locked test remains unavailable as a rescue set.

## Reproducibility closure

The existing retained evidence already establishes that all aggregate summaries and paired effects reproduce across independent workflow reruns, while some raw probabilities exhibit low-order floating-point drift. The audit classifies that drift as non-invalidating because the aggregate result, verifier verdict, and mechanism criteria are unchanged.

The machine verifier added in this audit now independently checks from committed evidence that:

1. all 1,117 eligible train rows and all 295 eligible validation rows were used;
2. the locked test remained unopened;
3. every per-seed retained accuracy corresponds to an integer correct count over 295 rows;
4. means and sample standard deviations recompute from those exact counts;
5. paired seed deltas recompute from the exact counts;
6. retained bootstrap intervals include zero for both mechanism comparisons;
7. no mechanism claim or `research_complete` state is authorized.

## Frozen verdict

`NEGATIVE_OR_INCONCLUSIVE_STATISTICS_RECOMPUTED_FROM_RETAINED_EVIDENCE`

The scientific conclusion is unchanged: this frozen LAM-JEPA configuration did not establish ARC-Challenge superiority, planner benefit, or EMA-target benefit. The capacity-matched supervised comparator remains adverse to the full model in the retained study. The historical ARC confirmatory test remains locked.

## Successor boundary

A successor line already exists as a **separately preregistered pre-outcome control plane**. Its model/runtime/evidence/budget gates remain explicitly unauthorized for scientific execution until their own blockers close. This audit neither modifies nor authorizes that successor and does not use it to reinterpret the frozen negative study.
