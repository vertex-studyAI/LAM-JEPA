# LAM successor study — ARC-contextual predictive representation protocol v1 (DRAFT)

**Status:** DRAFT / NOT FROZEN / NOT AUTHORIZED FOR HELD-OUT CLAIMS  
**Date:** 2026-08-27; primary decision rules and context/target construction frozen pre-outcome 2026-09-06  
**Scientific relationship to ARC-v5:** separate versioned study. This protocol does not modify, rescue, or reinterpret the frozen ARC-v5 negative/inconclusive result. The old locked confirmatory test remains untouched.

## 1. Why a successor study is justified

The frozen ARC-v5 line answered its own question negatively: the tested LAM-JEPA configuration did not outperform its gradient-active-parameter-matched supervised comparator, and its planner/EMA-target contribution criteria were not established. Source audit also showed that the frozen path used hashed whitespace tokens, mean pooling, and same-input EMA alignment rather than a contextual encoder predicting genuinely withheld information.

A successor is therefore permitted only as a **new hypothesis** that fixes the question rather than tuning the failed answer.

## 2. Primary research question

> With the encoder, classifier head, optimization budget, and supervised labels matched, does adding a genuinely context-to-distinct-target joint-embedding predictive objective improve held-out generalization or sample efficiency over ordinary contextual supervised training?

This protocol intentionally does **not** ask whether the old ARC-v5 architecture can be tuned until it wins.

## 3. Primary hypotheses and frozen decision rules

The primary statistical decision rules are now frozen pre-outcome in `protocols/arc_successor_v1_decision_rules.json` (SHA-256 `28d7158589091c2326154a30a4a14b1dac50c5d3049fae2b70298ac1a14dc625`). Freezing these fields resolves only `DELTA_PRIMARY`, `SEED_WIN_FRACTION`, and `UNCERTAINTY_RULE`; it does not authorize execution or resolve the remaining scientific blockers.

- **H1 — contextual predictive auxiliary value:** T1 must exceed B0 by a mean paired held-out accuracy difference of at least **0.02 absolute accuracy**, and the lower endpoint of the frozen 95% paired hierarchical bootstrap interval must be strictly greater than zero. The 0.02 practical margin deliberately carries forward the predecessor study's already-established threshold rather than selecting a more permissive successor-specific effect after prior negative evidence.
- **H2 — seed consistency:** at least **4 of the 5** frozen seeds (`11, 23, 37, 53, 71`) must have a strictly positive T1−B0 paired accuracy effect. Ties do not count as wins. Failed, divergent, or collapsed seeds may not be removed post hoc.
- **H3 — non-collapse:** T1 must clear every separately frozen representation-health gate. A numerically favorable classifier result accompanied by representation collapse does not count as mechanism success.

The uncertainty estimator is a **10,000-replicate paired hierarchical percentile bootstrap** with bootstrap seed `20260906`. Each replicate resamples frozen seeds with replacement and, within each selected seed, resamples confirmatory examples while preserving the paired T1/B0 correctness difference on each example. The reported statistic is the mean paired accuracy difference across the resampled seeds. Decision gates are evaluated on unrounded values; rounding is display-only.

H1–H3 must all pass for a positive primary mechanism result. Secondary metrics cannot rescue primary failure.

## 4. Freshness / leakage gate before any scientific run

Historical ARC-Challenge validation outcomes have already been observed in the project. Therefore the previous ARC validation split must **not** be relabeled as a clean confirmatory set for this successor.

Before freezing v1, maintain `DATA_FRESHNESS_AUDIT.md` recording:

1. every ARC split previously accessed by this project;
2. whether labels or aggregate outcomes were inspected;
3. every prompt/example used during architecture or hyperparameter development;
4. the candidate development and confirmatory datasets for v1;
5. why the confirmatory set is genuinely unobserved by the treatment-development process.

The audit exists, but its independent review remains a hard blocker. If a genuinely unobserved confirmatory set cannot be established, v1 may run only as a **development study**. It must not be described as confirmatory validation.

The old ARC-v5 locked test is never opened as part of this successor.

## 5. Data plan

### Development surface

Use only data explicitly marked development-safe by the freshness audit. The default candidate is ARC-Challenge training data with a newly frozen internal train/dev construction whose exact indices and hashes are recorded before treatment comparison.

### Confirmatory surface

`CONFIRMATORY_DATASET` remains unresolved and is a **hard blocker**. OpenBookQA is retained only as a separately sourced candidate dossier; it is not approved merely because its provenance metadata are pinned. A confirmatory dataset must satisfy all of the following before protocol freeze:

- not previously used to tune this treatment family;
- task-compatible with the frozen input/output contract;
- licensing and redistribution/use status recorded;
- immutable dataset version and exact local-byte hash retained;
- labels hidden from model/hyperparameter development;
- overlap/contamination review completed to the extent mechanically and manually reviewable;
- exact development/confirmatory split policy frozen;
- one-shot evaluation rule documented and independently approved pre-outcome.

No scientific treatment run is authorized while `CONFIRMATORY_DATASET` is unresolved.

## 6. Model families

Every learned primary comparison uses the same contextual encoder family and classifier head dimensions unless an explicitly documented parameter-matching adjustment is required.

### B0 — contextual supervised baseline

- contextual text encoder `ENCODER_FAMILY`;
- multiple-choice classifier head;
- supervised cross-entropy only;
- same token budget, labels, optimizer family, training steps, early-stopping rule, and augmentation policy as T1.

### B1 — reconstruction/control auxiliary baseline

Same B0 backbone plus a conventional masked-token or reconstruction auxiliary objective. This tests whether any extra self-supervised signal helps, rather than attributing generic auxiliary-training gains to JEPA.

### T1 — distinct-target JEPA auxiliary

- online/context encoder receives a deliberately incomplete context view;
- EMA target encoder receives a **distinct withheld target view** unavailable to the online branch;
- predictor maps context representations to target representations;
- target representation is stop-gradient;
- supervised classifier remains matched to B0;
- no target token/sample leakage into the context branch;
- masking/view construction is deterministic from the frozen seed/config.

### T2 — T1 plus anti-collapse regularization

Same T1 design plus predeclared variance/covariance regularization. T2 is a secondary treatment motivated by recent JEPA collapse literature. It may not replace T1 after an unfavorable T1 outcome.

### VQ variants — deferred secondary ablation

Vector quantization is **not** part of the primary headline treatment. A quantized T1/T2 variant may be evaluated only after the non-quantized primary protocol is frozen and its VQ-specific diagnostics and budgets are preregistered. This prevents the old quantization path from becoming an uncontrolled rescue knob.

## 7. Context/target construction

`CONTEXT_TARGET_CONSTRUCTION` is now resolved by the pre-outcome artifact `protocols/arc_successor_v1_context_target.json` (SHA-256 `6d828536b9983f84beb85e29c1db56dfd01a23aa22f3840a88272867bd3f5d6b`) and the executable constructor `tools/arc_successor_context_target.py`. This resolution does **not** authorize a scientific run and does not resolve the encoder, data, collapse, budget, environment, or independent-review blockers.

The frozen construction is label-blind and deterministic. It accepts only a stable example identifier, the question stem, and ordered choice texts. Answer keys, gold labels, correct-choice indices, and equivalent label-bearing fields are not inputs to view construction. One contiguous span is selected from the **question stem only** using whitespace-token positions, a fixed view seed `20260906`, a 20% span fraction, a minimum of one token, and a maximum of eight tokens. The start position is deterministically derived from the example identifier and frozen view seed. The same example therefore receives the same view across epochs and model seeds.

The T1 auxiliary context replaces exactly that withheld question span with `[WITHHELD_SPAN]` and preserves all ordered choice texts. The auxiliary target contains only the withheld question span; it receives no answer label, choice text, or remaining context. B0 and T1 retain the same full label-free question-plus-choices serialization for the supervised classifier path, so this artifact changes the auxiliary predictive view rather than granting T1 different supervised input.

Post-outcome changes to mask rate, span rule, view seed, example-specific masking, or named-confirmatory-example special cases are prohibited. The constructor and its verifier/tests are experiment-enabling controls only; they are not evidence of an accuracy benefit or of non-collapse.

## 8. Matched-budget contract

For B0/B1/T1/T2 record and match, within a preregistered tolerance:

- trainable parameter count;
- encoder family and initialization checkpoint/revision;
- maximum sequence length and tokenizer;
- number of supervised examples and label exposures;
- number of optimizer steps;
- batch size / gradient accumulation;
- optimizer and learning-rate schedule;
- regularization unrelated to the treatment;
- early-stopping rule;
- hyperparameter-search budget;
- device class and precision;
- wall-clock and accelerator time.

If exact parameter equality is impossible, freeze an allowed parameter-count ratio and report it. Treatment-specific predictor/target components must never be hidden from the count. `PARAMETER_MATCH_TOLERANCE` and `MAX_COMPUTE_RATIO` remain unresolved hard blockers.

## 9. Seeds

Frozen primary seed set: `11, 23, 37, 53, 71`.

Failed, divergent, or collapsed seeds are retained in the aggregate unless a preregistered mechanical exclusion rule applies equally to all systems. Under the frozen primary rule, at least four seeds must have a strictly positive T1−B0 accuracy difference; zero-difference ties are not wins.

## 10. Primary and secondary metrics

### Primary

- multiple-choice held-out accuracy on the declared confirmatory evaluation surface.

The frozen primary material-effect gate is mean paired T1−B0 accuracy `>= 0.02`, with the paired hierarchical-bootstrap 95% lower endpoint `> 0` and at least four of five strictly positive seed effects.

### Secondary

- negative log-likelihood;
- expected calibration error or a predeclared calibration metric;
- low-data/sample-efficiency curves at frozen label fractions;
- paired per-seed B0/T1 effects;
- compute-normalized performance;
- representation-health metrics in Section 11.

A secondary metric cannot rescue failure on the frozen primary criterion.

## 11. Mandatory collapse diagnostics

Every training run must record at fixed checkpoints:

- per-dimension latent variance;
- latent covariance eigenvalue spectrum;
- effective rank / normalized effective rank;
- mean pairwise cosine similarity on a fixed probe batch;
- representation norms for online and target encoders;
- predictor output norm and target-prediction error;
- online-target alignment statistics;
- gradient norms for encoder, predictor, and classifier blocks.

For any VQ variant also record:

- active-code count;
- code-assignment histogram;
- entropy / perplexity of assignments;
- dead-code count and persistence;
- quantization error;
- nearest-code margin distribution;
- per-seed code-switch/utilization trajectory;
- encoder-distribution drift relative to codebook movement.

Exact collapse thresholds remain `COLLAPSE_THRESHOLDS_TBD` and must be fixed from train-only diagnostics or literature-supported values **before** treatment held-out outcomes are inspected.

## 12. Success, null, and kill criteria

Three primary decision fields are already frozen: `DELTA_PRIMARY=0.02`, `SEED_WIN_FRACTION=0.8` (4/5 strictly positive seeds), and the 10,000-replicate paired hierarchical `UNCERTAINTY_RULE` described above. `CONTEXT_TARGET_CONSTRUCTION` is also frozen as described in Section 7.

The study remains intentionally non-executable until these remaining blockers are resolved in evidence-backed artifacts:

- `DATA_FRESHNESS_AUDIT` independent review;
- `CONFIRMATORY_DATASET` independent approval and exact byte receipt;
- `ENCODER_FAMILY_AND_REVISION`;
- `COLLAPSE_THRESHOLDS`;
- `PARAMETER_MATCH_TOLERANCE`;
- `MAX_COMPUTE_RATIO`;
- `EXACT_REPRODUCE_COMMAND` and environment/source bindings.

### Success

H1–H3 must all pass. A favorable mean with failed seed consistency, a bootstrap lower bound at or below zero, or a collapse gate is not a positive mechanism result.

### Null / negative

If T1 does not clear the material-margin criterion against B0 under the frozen protocol, report the result as negative/inconclusive and stop primary architecture search.

### Kill

Kill the primary JEPA-specific claim if any occurs:

- matched B0 is equal or better under the frozen primary gate;
- benefit vanishes after equalizing label, step, context, or search budgets;
- performance requires excluding failed/collapsed seeds post hoc;
- predictive target construction is found to leak withheld information;
- treatment clears accuracy only while violating frozen representation-health gates;
- treatment exceeds the frozen compute ratio without the required benefit;
- confirmatory freshness cannot be established.

## 13. Hyperparameter selection

Hyperparameter search must occur only on the designated development surface with a fixed per-system trial budget. Search spaces are frozen before any treatment-specific held-out evaluation.

B0, B1, T1, and T2 receive the same number of tuning trials or equivalent predeclared compute budget. No architecture may receive extra trials because early results look promising.

## 14. Analysis plan

For every seed and system retain raw predictions and loss curves. Report:

1. per-seed metric table;
2. mean and sample standard deviation;
3. paired T1−B0 effects;
4. the frozen 10,000-replicate paired hierarchical percentile-bootstrap 95% interval;
5. all failed/divergent/collapsed runs;
6. representation-health curves;
7. parameter, label-exposure, step, runtime, and search-budget tables;
8. exact data and source hashes.

The bootstrap resamples seeds with replacement and then paired confirmatory examples within each sampled seed, preserving each example's T1/B0 correctness difference. The primary gate uses unrounded values. No p-hacking across multiple metrics is permitted.

## 15. Ablation order

Ablations are not allowed to rescue a failed primary treatment. If T1 passes its primary gate, run in this order:

1. T1 without EMA target update / shared-target control;
2. distinct-target construction variants predeclared before ablation execution;
3. predictor removal or linear-predictor control;
4. T2 variance/covariance regularization;
5. VQ on/off under matched non-quantized control;
6. if VQ is used, staged/warm-start codebook treatment as a separately frozen factor.

If T1 fails, archive the primary study and design any new treatment as v2.

## 16. Reproducibility package

A scientifically admissible run must produce:

- immutable source SHA;
- environment lockfile/container digest;
- dataset identifiers and hashes;
- exact split indices/hashes;
- exact tokenizer/checkpoint revision;
- machine-readable config;
- seed;
- parameter counts;
- raw predictions;
- optimizer and representation diagnostics;
- metric JSON;
- stdout/stderr log;
- artifact hashes;
- wall-clock/device metadata;
- one exact reproduce command;
- claim ledger mapping every reported number to its raw artifact.

## 17. Freeze checklist

Do **not** run held-out treatment evaluation until all are true:

- [ ] `DATA_FRESHNESS_AUDIT.md` independently reviewed.
- [ ] confirmatory dataset/split proven unobserved and frozen, or study explicitly downgraded to development-only.
- [ ] encoder/checkpoint/tokenizer frozen.
- [x] context/target visibility rules frozen (`protocols/arc_successor_v1_context_target.json`, SHA-256 `6d828536b9983f84beb85e29c1db56dfd01a23aa22f3840a88272867bd3f5d6b`).
- [ ] B0/B1/T1/T2 definitions and implementation identities frozen.
- [ ] parameter/search/compute tolerances frozen.
- [x] seeds frozen (`11, 23, 37, 53, 71`).
- [x] primary metric, 0.02 practical-effect threshold, 4/5 seed-consistency gate, and hierarchical-bootstrap estimator frozen pre-outcome.
- [ ] collapse thresholds frozen without held-out treatment inspection.
- [ ] exact commands/environment/source identities frozen.
- [ ] raw artifact schema and claim-ledger paths frozen.

The checked boxes above are necessary but not sufficient for execution authorization.

## 18. Publication boundary

Allowed before results:

> We preregister a successor study testing whether genuinely distinct-target predictive representation learning adds value over a matched contextual supervised baseline, with a pre-outcome 2 percentage-point practical-effect gate, four-of-five seed consistency requirement, and paired hierarchical bootstrap uncertainty rule.

Not allowed before evidence:

- “LAM-JEPA v2 improves ARC reasoning.”
- “JEPA beats supervised learning.”
- “quantization improves reasoning.”
- “the prior negative result was repaired.”
- any claim that the old locked test validated the successor.

## 19. Next implementation tasks

1. obtain independent review of the data-freshness audit and OpenBookQA candidate provenance/licensing/overlap/split policy; retain `CONFIRMATORY_DATASET` as unresolved until that review and exact-byte receipt exist;
2. select and pin the contextual encoder/checkpoint/tokenizer;
3. integrate the frozen context/target constructor into the B0/T1 data path and keep the label/visibility/leakage regressions green;
4. freeze collapse thresholds from literature/train-only diagnostics plus parameter-match and compute-ratio tolerances;
5. implement B0 first and verify deterministic data plumbing;
6. add the representation-health logger;
7. implement T1 only after B0 and leakage tests are green;
8. freeze the exact environment/source/reproduce command in a new evidence-backed commit;
9. only then authorize the development/confirmatory plan permitted by the independently reviewed freshness boundary.
