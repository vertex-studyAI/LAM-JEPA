# LAM successor study — ARC-contextual predictive representation protocol v1 (DRAFT)

**Status:** DRAFT / NOT FROZEN / NOT AUTHORIZED FOR HELD-OUT CLAIMS  
**Date:** 2026-08-27  
**Scientific relationship to ARC-v5:** separate versioned study. This protocol does not modify, rescue, or reinterpret the frozen ARC-v5 negative/inconclusive result. The old locked confirmatory test remains untouched.

## 1. Why a successor study is justified

The frozen ARC-v5 line answered its own question negatively: the tested LAM-JEPA configuration did not outperform its gradient-active-parameter-matched supervised comparator, and its planner/EMA-target contribution criteria were not established. Source audit also showed that the frozen path used hashed whitespace tokens, mean pooling, and same-input EMA alignment rather than a contextual encoder predicting genuinely withheld information.

A successor is therefore permitted only as a **new hypothesis** that fixes the question rather than tuning the failed answer.

## 2. Primary research question

> With the encoder, classifier head, optimization budget, and supervised labels matched, does adding a genuinely context-to-distinct-target joint-embedding predictive objective improve held-out generalization or sample efficiency over ordinary contextual supervised training?

This protocol intentionally does **not** ask whether the old ARC-v5 architecture can be tuned until it wins.

## 3. Primary hypotheses

- **H1 — contextual predictive auxiliary value:** treatment T1 outperforms baseline B0 on the frozen primary metric by at least the preregistered material margin `DELTA_PRIMARY`.
- **H2 — seed consistency:** the paired T1−B0 effect is positive for at least `SEED_WIN_FRACTION` of frozen seeds and its uncertainty interval satisfies the preregistered directional criterion.
- **H3 — non-collapse:** T1 clears every frozen representation-health gate. A numerically favorable classifier result accompanied by representation collapse does not count as mechanism success.

Secondary hypotheses may be added before freeze, but no secondary result can rescue failed H1–H3.

## 4. Freshness / leakage gate before any scientific run

Historical ARC-Challenge validation outcomes have already been observed in the project. Therefore the previous ARC validation split must **not** be relabeled as a clean confirmatory set for this successor.

Before freezing v1, create `DATA_FRESHNESS_AUDIT.md` that records:

1. every ARC split previously accessed by this project;
2. whether labels or aggregate outcomes were inspected;
3. every prompt/example used during architecture or hyperparameter development;
4. the candidate development and confirmatory datasets for v1;
5. why the confirmatory set is genuinely unobserved by the treatment-development process.

If a genuinely unobserved confirmatory set cannot be established, v1 may run only as a **development study**. It must not be described as confirmatory validation.

The old ARC-v5 locked test is never opened as part of this successor.

## 5. Data plan

### Development surface

Use only data explicitly marked development-safe by the freshness audit. The default candidate is ARC-Challenge training data with a newly frozen internal train/dev construction whose exact indices and hashes are recorded before treatment comparison.

### Confirmatory surface

`CONFIRMATORY_DATASET` is unresolved at draft time and is a **hard blocker**. It must satisfy all of the following before protocol freeze:

- not previously used to tune this treatment family;
- task-compatible with the frozen input/output contract;
- licensing and redistribution status recorded;
- immutable dataset version/hash retained;
- labels hidden from model/hyperparameter development;
- one-shot evaluation rule documented.

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

The context and target must encode genuinely different information. Accepted v1 constructions are restricted to one choice frozen before outcomes:

- masked contiguous semantic spans;
- withheld sentence/question substructure that is not visible in the context view;
- another explicitly defined non-overlapping representation target.

The target encoder may not receive the same full serialized input as the context encoder and then be described as predictive target learning.

The exact mask rate/distribution, span construction, randomization, and visibility rules are freeze-time fields.

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

If exact parameter equality is impossible, freeze an allowed parameter-count ratio and report it. Treatment-specific predictor/target components must never be hidden from the count.

## 9. Seeds

Default frozen seed proposal: `11, 23, 37, 53, 71`.

These seed values may be changed only before protocol freeze. Once frozen, failed, divergent, or collapsed seeds are retained in the aggregate unless a preregistered mechanical exclusion rule applies equally to all systems.

## 10. Primary and secondary metrics

### Primary

- multiple-choice held-out accuracy on the declared evaluation surface.

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

Exact collapse thresholds are `COLLAPSE_THRESHOLDS_TBD` at draft time and must be fixed from train-only diagnostics or literature-supported values **before** treatment held-out outcomes are inspected.

## 12. Success, null, and kill criteria

This draft is intentionally non-executable until the numerical fields below are frozen:

- `DELTA_PRIMARY` — minimum material paired accuracy gain;
- `SEED_WIN_FRACTION` — minimum fraction of seeds with positive paired effect;
- `UNCERTAINTY_RULE` — e.g. bootstrap interval condition;
- `COLLAPSE_THRESHOLDS`;
- `PARAMETER_MATCH_TOLERANCE`;
- `MAX_COMPUTE_RATIO`;
- `CONFIRMATORY_DATASET`.

### Success

H1–H3 must all pass. A favorable mean with failed seed consistency or a collapse gate is not a positive mechanism result.

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
4. a preregistered paired bootstrap interval or other frozen uncertainty estimator;
5. all failed/divergent/collapsed runs;
6. representation-health curves;
7. parameter, label-exposure, step, runtime, and search-budget tables;
8. exact data and source hashes.

No p-hacking across multiple metrics. The primary metric and uncertainty rule are frozen first.

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

- [ ] `DATA_FRESHNESS_AUDIT.md` complete.
- [ ] confirmatory dataset/split proven unobserved and frozen, or study explicitly downgraded to development-only.
- [ ] encoder/checkpoint/tokenizer frozen.
- [ ] context/target visibility rules frozen.
- [ ] B0/B1/T1/T2 definitions frozen.
- [ ] parameter/search/compute tolerances frozen.
- [ ] seeds frozen.
- [ ] primary metric and uncertainty estimator frozen.
- [ ] numerical success/kill thresholds frozen.
- [ ] collapse thresholds frozen without held-out treatment inspection.
- [ ] exact commands/environment frozen.
- [ ] raw artifact schema and claim ledger paths frozen.

## 18. Publication boundary

Allowed before results:

> We preregister a successor study testing whether genuinely distinct-target predictive representation learning adds value over a matched contextual supervised baseline.

Not allowed before evidence:

- “LAM-JEPA v2 improves ARC reasoning.”
- “JEPA beats supervised learning.”
- “quantization improves reasoning.”
- “the prior negative result was repaired.”
- any claim that the old locked test validated the successor.

## 19. Next implementation tasks

1. produce the data-freshness audit;
2. select and pin the contextual encoder/checkpoint;
3. implement B0 first and verify deterministic data plumbing;
4. implement context/target visibility tests that fail on leakage;
5. add the representation-health logger;
6. implement T1 only after B0 and leakage tests are green;
7. fill every `TBD` field and convert this file to a frozen protocol in a new commit;
8. only then run the development/confirmatory plan permitted by the freshness audit.
