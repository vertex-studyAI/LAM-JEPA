# LAM-JEPA Results Ledger

**Reproducibility wave:** 2026-08-12 to 2026-08-13  
**Frozen scientific source revision:** `760aa7f9a73a177d5ff4ba7eb470f7e68ace63cb` for the full ARC-v3 controls validation  
**Seed-order reproducibility repair:** `b72a97a99769b278eb8ec75bc5eab62dc9599f29`  
**Scientific status:** negative / inconclusive on the frozen ARC-Challenge superiority and mechanism hypotheses  
**Confirmatory test status:** LOCKED; do not use it to rescue the failed validation hypothesis.

## Research question

Under the frozen ARC-Challenge train/validation protocol, does LAM-JEPA improve validation accuracy over a gradient-active-parameter-matched supervised baseline, and do the planner, target/EMA path, or repaired quantized latent mechanism contribute a reproducible validation benefit?

## Dataset and task

AI2 ARC-Challenge multiple-choice reasoning. Protocol v3 retains exactly four-choice rows, preserves source order, and uses checksum-addressed train and validation data only. The locked ARC test is not downloaded or evaluated for this failed hypothesis line.

Frozen full-controls budget:

- seeds: `1 2 3 4 5`;
- epochs: `20`;
- batch size: `32`;
- learning rate: `0.0003`;
- model steps: `1`;
- eligible train rows: `1117`;
- eligible validation rows: `295`;
- device: CPU.

## Baselines and controls

- capacity-matched supervised baseline using gradient-active parameter matching;
- pinned `microsoft/deberta-v3-xsmall` comparator for bounded development characterization;
- deterministic shuffled-label negative control;
- `no_planner` and `no_target` ablations.

## Canonical scientific result

| System / control | Validation accuracy, mean ± sample SD | n | Interpretation |
|---|---:|---:|---|
| Full LAM-JEPA | 0.2549152493 ± 0.0129968006 | 5 | Proposed model |
| Capacity-matched supervised | 0.2664406780 ± 0.0154600058 | 5 | Matched baseline; stronger mean |
| `no_planner` | 0.2501694888 ± 0.0129968006 | 5 | Planner ablation |
| `no_target` | 0.2616949081 ± 0.0203953938 | 5 | Target-path ablation |
| Shuffled-label control | 0.2630508393 ± 0.0145011803 | 5 | Below frozen 0.35 failure threshold |

Paired mechanism effects from the frozen full-controls run:

- full − `no_planner`: mean `+0.0047457606`, sample SD `0.0106118432`, bootstrap 95% CI `[0.0, 0.0142372817]`;
- full − `no_target`: mean `-0.0067796588`, sample SD `0.0092834301`, bootstrap 95% CI `[-0.0135593176, 0.0]`.

Neither predeclared mechanism criterion was met. No statistical-significance claim is made.

The separately retained capacity-matched comparison remains adverse to LAM-JEPA (`0.2549152542 ± 0.0129968064` versus `0.2664406780 ± 0.0154600058`; paired LAM minus matched `-0.0115254237 ± 0.0140994131`). A bounded development comparison against the pinned pretrained comparator was also adverse (`0.15625` vs `0.21875`).

## Independent full scientific reruns

The frozen workflow `.github/workflows/arc-protocol-v3-full-controls-validation.yml` was rerun unchanged on the scientific SHA `760aa7f9a73a177d5ff4ba7eb470f7e68ace63cb`.

### Attempt 2

- Actions run: `31203337502`, attempt `2`;
- job: `94178988063`;
- artifact: `9149336081`;
- artifact digest: `sha256:c45710b5dae6a767ccb6bab7f6e3d8e9578752d8cf9b79fd82a65ae824dded1b`;
- verdict: success.

### Attempt 3

- Actions run: `31203337502`, attempt `3`;
- job: `94291056903`;
- artifact: `9162165932`;
- artifact digest: `sha256:caa898f1ff046a337db9b5ddbffe1b332943a732868e2fd809abeda8ee89c30b`;
- verdict: success.

Attempt 3 again completed protocol verification, checksum-addressed train/validation acquisition, the full five-seed twenty-epoch run, independent verification, frozen-budget assertions, and evidence upload. The locked test remained absent.

## Attempt-2 vs attempt-3 artifact comparison

Both artifacts contain the same 10 retained files. Eight files are byte-identical, including:

- ARC download manifest;
- human-readable benchmark output;
- strict verifier report;
- verifier summary JSON;
- verifier console output;
- protocol verification JSON;
- train and validation parquet files.

The raw full-results JSON and its normalized-input copy differ in low-order floating-point prediction probabilities. Across the raw result trees, 35,526 numeric leaves differ while no non-numeric leaf differs. The maximum observed numeric drift was approximately `5.9186e-4` in one negative-control per-example class probability.

Critically, the following structures are **exactly equal** between attempts 2 and 3:

- full-model aggregate accuracy mean and SD;
- `no_planner` aggregate accuracy mean and SD;
- `no_target` aggregate accuracy mean and SD;
- every paired seed-level mechanism delta;
- paired means, SDs and bootstrap intervals;
- negative-control aggregate accuracy mean and SD;
- negative-control pass/fail verdict;
- independent verifier verdict and strict report.

Therefore the defensible reproducibility claim is exact replication of the frozen scientific aggregate conclusion and verifier decision, with low-level floating-point probability drift below the level needed to alter the scientific result. Byte-exact full raw JSON identity across independent runners is not claimed.

## Metadata inconsistency discovered in attempt 3

The frozen runner's raw `protocol.claim_boundary` string says the control script is “not the final five-seed/20-epoch protocol.” That text is stale for this invocation: the workflow explicitly passes seeds `1..5`, 20 epochs, batch size 32, full eligible train/validation sets, and the independent verifier confirms `final_five_seed_20_epoch_protocol_executed = true`.

This is classified as a **non-invalidating reporting-metadata defect**, not a scientific protocol failure. The executed arguments, retained data, aggregate metrics, verifier, and locked-test policy all agree on the actual run. The frozen artifact is preserved unchanged; it is not rewritten after observation.

## Seed-order reproducibility bug and repair

A separate reproducibility defect was previously found in `train_single.py`: model initialization occurred before the requested seed was applied. Under nominally identical SHA / CLI / seed / CPU execution, one-step losses differed (`10.853294372558594` vs `10.34877872467041`). This pre-fix evidence remains preserved.

PR #61 applied the smallest software repair: seed before `LAMJEPA(cfg)` construction while retaining trainer-side seeding for the subsequent stream. No ARC split, seed set, scientific threshold, metric, architecture, or locked-test policy changed.

The repaired deterministic replay metadata records six independently verified replay attempts. Within each runner attempt, model state, final metrics and RNG state are exact; across attempts, final loss `11.704492568969727` and final accuracy `0.0` remain exact while some secondary floating-point values drift and PyTorch checkpoint bytes are not identical.

This software reproducibility repair does not rescue the negative ARC scientific result.

## Repaired ARC-v5 line

The separate train-only quantizer repair `arc-v5-stable-ema-residual-0.03125` restored its bounded trainability gate, but repaired validation remained `VALID_NEGATIVE_OR_INCONCLUSIVE_VALIDATION`. The generalization and quantization-benefit gates were not supported.

## Result

The defensible conclusion is **not** that LAM-JEPA beats ARC baselines. The full five-seed ARC controls result has now survived another independent frozen rerun with exactly matching aggregate scientific conclusions and verifier outputs. LAM-JEPA remains below its capacity-matched supervised baseline, and the planner/target mechanism criteria remain unsupported.

## Limitations

- Five validation seeds do not justify broad benchmark-general significance claims.
- ARC-Challenge is one benchmark family and the locked test remains intentionally unused for this failed hypothesis line.
- Independent runners exhibit low-order floating-point drift in per-example probabilities.
- The stale raw claim-boundary string is a reporting defect and should not be read as the executed budget.
- The pretrained comparator is bounded development characterization, not a full matched confirmatory trial.
- No claim of educational effectiveness, general benchmark superiority, AGI, or general intelligence is supported.

## Stop rule

Do not tune the current architecture or thresholds against the locked ARC test. Any new architectural repair, benchmark, or scientific hypothesis must receive a new versioned protocol before its validation evidence is observed.
