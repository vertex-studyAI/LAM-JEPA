# LAM-JEPA reproducibility wave — 2026-08-12 to 2026-08-13

## Executive scientific status

The current reproducibility evidence supports a **negative/inconclusive ARC conclusion**, not a superiority claim.

Under the frozen ARC-Challenge protocol v3, using five seeds, 20 epochs, batch size 32, learning rate `0.0003`, all 1,117 eligible train rows and all 295 eligible validation rows on CPU, the locked ARC test split was not downloaded or evaluated.

| Condition | Mean validation accuracy | Sample SD | n |
|---|---:|---:|---:|
| full LAM-JEPA | 0.2549152493 | 0.0129968006 | 5 |
| no planner | 0.2501694888 | 0.0129968006 | 5 |
| no target | 0.2616949081 | 0.0203953938 | 5 |
| shuffled-label negative control | 0.2630508393 | 0.0145011803 | 5 |

Paired mechanism effects:

- full minus `no_planner`: mean `+0.0047457606`, sample SD `0.0106118432`, bootstrap 95% CI `[0.0, 0.0142372817]`; preregistered criterion **not met**.
- full minus `no_target`: mean `-0.0067796588`, sample SD `0.0092834301`, bootstrap 95% CI `[-0.0135593176, 0.0]`; preregistered criterion **not met**.
- the shuffled-label control remained below the frozen `0.35` ceiling, but that does not rescue the planner, target, superiority, or quantization hypotheses.

**Verdict:** `ARC_SUPERIORITY_AND_MECHANISM_HYPOTHESES_UNSUPPORTED`.

## Independent full-scientific rerun

The full frozen controls were independently rerun from scientific source SHA `760aa7f9a73a177d5ff4ba7eb470f7e68ace63cb` in workflow run `31203337502`.

Retained successful attempts:

- attempt 2: job `94178988063`, artifact `9149336081`, digest `sha256:c45710b5dae6a767ccb6bab7f6e3d8e9578752d8cf9b79fd82a65ae824dded1b`;
- attempt 3: job `94291056903`, artifact `9162165932`, digest `sha256:caa898f1ff046a337db9b5ddbffe1b332943a732868e2fd809abeda8ee89c30b`.

Each artifact contains 10 retained files. Eight files were byte-identical across the two attempts. Raw result JSON and normalized input JSON were not byte-identical because of low-order floating-point drift: 35,526 numeric leaf values differed, with maximum observed drift `0.0005918592214584351`. There were no non-numeric leaf differences.

Despite that low-level drift, the following were exact across the independent attempts:

- aggregate full accuracy;
- aggregate `no_planner` accuracy;
- aggregate `no_target` accuracy;
- paired mechanism effects;
- negative-control summary;
- verifier summary;
- strict verifier report;
- final scientific conclusion.

The defensible reproducibility claim is therefore **exact aggregate/verifier reproduction with bounded low-level floating-point drift**, not byte-for-byte identity of every probability or serialized artifact.

## Frozen ARC-v5 repaired-validation result

The separate ARC-v5 repaired-validation line remains **`VALID_NEGATIVE_OR_INCONCLUSIVE_VALIDATION`**. It is retained because the trainability repair is a legitimate software/mechanism investigation, but it does not authorize a positive reinterpretation.

Frozen five-seed validation summary (ARC-Challenge validation, test split not accessed):

| Condition | Mean accuracy | SD | Bootstrap 95% CI | n |
|---|---:|---:|---:|---:|
| legacy CE | 0.2616949081 | 0.0203953938 | [0.2454237282, 0.2779660881] | 5 |
| repaired v5 CE | 0.2657627106 | 0.0219162368 | [0.2474576294, 0.2820338905] | 5 |
| no quantizer CE | 0.2623728782 | 0.0183489654 | [0.2481355965, 0.2766101599] | 5 |
| repaired v5 shuffled labels | 0.2501694858 | 0.0231652156 | [0.2332203358, 0.2698304981] | 5 |

Paired repaired-minus-legacy mean = `0.0040678024`, SD = `0.0234119207`, bootstrap 95% CI = `[-0.0135593116, 0.0216949165]`.

Paired repaired-minus-no-quantizer mean = `0.0033898324`, SD = `0.0075798873`, bootstrap 95% CI = `[-0.0027118593, 0.0094915211]`.

Predeclared decisions: negative control valid = true; collapse rejected = false; generalization supported with limitations = false; quantization benefit supported = false.

## Reproducibility defect discovered and repaired

Rerunning exact pre-fix main SHA `2f59b4297e5978d4ce769ebe95adb363e1e75d7a` with the same seed/CLI/CPU workflow produced different one-step training losses: `10.853294372558594` and `10.34877872467041`.

The non-determinism was isolated to `train_single.py`: `LAMJEPA(cfg)` was instantiated before the requested seed was applied by `Trainer.__init__`, so model initialization was not governed by `--seed`.

The merged fix is SHA `b72a97a99769b278eb8ec75bc5eab62dc9599f29`, PR #61. The fix changes reproducibility plumbing, not the frozen scientific protocol.

## Deterministic replay boundary

The repaired same-seed workflow has been independently replayed six times. Latest retained replay: Actions run `31641305854`, attempt 6, artifact `9160533550`, digest `sha256:84ef6f4a9c4274441a8e8a4b959620551cd37ae6fbd29a0efd07510553359354`, head `96ddbe4433f514aeeede87e734085a9c8a9313e9`.

Within an attempt, exact equality is verified for model state, metrics, semantic metadata, and RNG state. Across independent CPU runners, primary one-step outputs remain exact:

- final loss: `11.704492568969727`;
- final accuracy: `0.0`.

Checkpoint bytes and every secondary floating-point quantity are not claimed byte-identical across independent runners.

## Reporting metadata defect

The frozen raw full-controls result includes one stale claim-boundary sentence stating that the invocation is not the final five-seed/20-epoch protocol. The actual workflow arguments and independent verifier confirm five seeds, 20 epochs, the full eligible train/validation rows, and `final_five_seed_20_epoch_protocol_executed=true`.

This is classified as a **non-invalidating reporting metadata defect**. The frozen raw artifact is preserved unchanged; executable arguments and independent verifier output are authoritative for the executed budget.

## Limitations and claim boundary

Do not claim:

- LAM-JEPA superiority on ARC;
- validated planner benefit;
- validated target-path benefit;
- validated quantization benefit;
- educational effectiveness;
- AGI/general-intelligence capability;
- research completeness.

The ARC confirmatory test remains locked for the failed hypothesis. Future architectural changes or new hypotheses should be versioned separately and preregistered before observing their validation evidence.
