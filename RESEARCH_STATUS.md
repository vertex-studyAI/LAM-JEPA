# LAM-JEPA Research Status

**Evidence cutoff:** 13 August 2026  
**Current documentation baseline:** `6c6f5c10e8610239ce6c72a4fa7f549659662014`  
**Frozen full-controls scientific source:** `760aa7f9a73a177d5ff4ba7eb470f7e68ace63cb`  
**Classification:** `GREEN_REPRODUCIBLE_NEGATIVE_RESULT / ARC SUPERIORITY HYPOTHESIS UNSUPPORTED / RESEARCH_COMPLETE_FALSE`

## Executive result

LAM-JEPA now has a reproducible negative/inconclusive ARC-Challenge evidence package. The frozen superiority and mechanism hypotheses are unsupported. This is a legitimate GREEN state for **reproducible scientific falsification**, not a claim of benchmark superiority, external generalization, publication readiness, or research completion.

The locked ARC confirmatory test must not be used to rescue the failed validation hypothesis.

## What is reproducibly established

The repository contains and has executed:

- checksum-pinned ARC-Challenge train/validation acquisition with the test split absent;
- a five-seed, 20-epoch frozen full-controls validation;
- a gradient-active-parameter-matched supervised baseline;
- `no_planner` and `no_target` mechanism ablations;
- a deterministic shuffled-label negative control;
- a pinned DeBERTa development comparator path;
- a separately frozen repaired-v5 trainability/validation line;
- independent scientific reruns with retained artifact digests;
- a deterministic-training seed-order repair with the pre-fix failure preserved.

## Frozen ARC full-controls result

Protocol budget:

- seeds `[1, 2, 3, 4, 5]`;
- 20 epochs;
- batch size 32;
- learning rate `0.0003`;
- model steps 1;
- 1,117 eligible train rows;
- 295 eligible validation rows;
- CPU execution;
- locked test not evaluated.

| Condition | Mean accuracy | Sample SD | n |
|---|---:|---:|---:|
| Full LAM-JEPA | 0.2549152493 | 0.0129968006 | 5 |
| `no_planner` | 0.2501694888 | 0.0129968006 | 5 |
| `no_target` | 0.2616949081 | 0.0203953938 | 5 |
| Shuffled-label control | 0.2630508393 | 0.0145011803 | 5 |

Paired mechanism effects:

- full − `no_planner`: `+0.0047457606`, bootstrap 95% CI `[0.0, 0.0142372817]`, criterion not met;
- full − `no_target`: `-0.0067796588`, bootstrap 95% CI `[-0.0135593176, 0.0]`, criterion not met.

The separately retained capacity-matched supervised baseline has higher mean validation accuracy than LAM-JEPA. The bounded pinned DeBERTa development comparison is also adverse to LAM-JEPA and remains characterization evidence rather than a broad inferiority theorem.

**Frozen verdict:** `ARC_SUPERIORITY_AND_MECHANISM_HYPOTHESES_UNSUPPORTED`.

## Independent full scientific reruns

Frozen scientific revision: `760aa7f9a73a177d5ff4ba7eb470f7e68ace63cb`.

Attempt 2:

- workflow run `31203337502`;
- job `94178988063`;
- artifact `9149336081`;
- digest `sha256:c45710b5dae6a767ccb6bab7f6e3d8e9578752d8cf9b79fd82a65ae824dded1b`;
- success.

Attempt 3:

- workflow run `31203337502`;
- job `94291056903`;
- artifact `9162165932`;
- digest `sha256:caa898f1ff046a337db9b5ddbffe1b332943a732868e2fd809abeda8ee89c30b`;
- success.

The aggregate model scores, paired effects, negative-control summary, verifier summary, and strict verifier verdict are exact across the two attempts. Low-level per-example probabilities show floating-point drift with maximum observed numeric difference approximately `5.9186e-4`; 8/10 retained files are byte-identical.

Correct wording: **scientific aggregate reproducibility**, not byte-identical full-artifact reproducibility.

## Deterministic replay repair

Pre-fix defect: `train_single.py` instantiated `LAMJEPA` before applying the requested seed. The differing nominal same-seed losses are retained as invalidated reproducibility evidence.

Repair:

- PR #61;
- merged fix `b72a97a99769b278eb8ec75bc5eab62dc9599f29`;
- no scientific protocol change.

Six independently verified replay attempts show exact within-attempt model state/metrics/RNG state and exact cross-attempt final loss/accuracy. Secondary floats and serialized checkpoint bytes are not claimed byte-identical.

## Repaired-v5 line

The bounded v5 trainability repair restored its declared train-only gate. Its separately frozen validation remained `VALID_NEGATIVE_OR_INCONCLUSIVE_VALIDATION` and did not establish generalization or quantization benefit.

Do not use this engineering repair to retroactively validate the original hard-VQ mechanism.

## Preserved stale-evidence discrepancy

The frozen raw full-controls payload contains a stale `protocol.claim_boundary` sentence saying the invocation is not the final five-seed/20-epoch protocol. The executable arguments, protocol fields, and independent verifier confirm that the final five-seed/20-epoch budget was executed.

This is classified as `NON_INVALIDATING_REPORTING_METADATA_DEFECT`. The raw evidence is preserved unchanged; the discrepancy is documented rather than rewritten after observing the result.

## Supported claims

The repository may state that:

1. the documented training/evaluation pipeline executes reproducibly;
2. ARC-Challenge external-benchmark plumbing is implemented with retained eligibility evidence;
3. the frozen multi-seed full-controls validation was executed;
4. aggregate scientific conclusions reproduce across independent reruns;
5. the capacity-matched comparison does not support LAM-JEPA superiority;
6. planner and target mechanism criteria are not met;
7. the bounded v5 repair improves trainability under its own train-only gate but does not rescue validation;
8. adverse and failed runs are retained rather than tuned away.

## Unsupported claims

Do not claim:

- LAM-JEPA superiority on ARC;
- planner benefit on ARC;
- target-path benefit on ARC;
- repaired quantization benefit on ARC validation;
- general benchmark superiority;
- externally validated educational effectiveness;
- confirmatory-test success;
- AGI/general-intelligence capability;
- `RESEARCH_COMPLETE`;
- byte-identical independent-run checkpoints or raw predictions.

## Closure state

### GREEN

- frozen negative scientific conclusion;
- multi-seed full-controls execution;
- capacity-matched/ablation evidence;
- independent aggregate result reproduction;
- deterministic replay semantics;
- preserved failed/pre-fix runs;
- explicit scientific stop rule;
- executable reproduction commands in `REPRODUCE.md`;
- machine-readable experiment metadata.

### Still not submission-ready

- owner-approved root license and citation metadata remain unresolved;
- related-work bibliography must be source-verified before insertion;
- the manuscript still requires final publication formatting/reviewer pass;
- no new hypothesis may inherit the failed ARC line without a separate preregistration.

## Scientific stop rule

The locked ARC confirmatory test must not be used to rescue the failed validation hypothesis. Do not retune the failed validation line. Any future architecture repair, dataset, or scientific hypothesis must be versioned separately, frozen before its validation evidence is observed, and reported independently from this negative result.
