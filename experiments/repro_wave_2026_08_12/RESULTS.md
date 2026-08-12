# LAM-JEPA reproducibility wave — 2026-08-12

## Scientific status

The frozen ARC-v5 validation result remains **`VALID_NEGATIVE_OR_INCONCLUSIVE_VALIDATION`**. This reproducibility repair does not authorize a positive reinterpretation.

Frozen five-seed validation summary (ARC-Challenge validation, test split not accessed):

| Condition | Mean accuracy | SD | Bootstrap 95% CI | n |
|---|---:|---:|---:|---:|
| legacy CE | 0.2616949081 | 0.0203953938 | [0.2454237282, 0.2779660881] | 5 |
| repaired v5 CE | 0.2657627106 | 0.0219162368 | [0.2474576294, 0.2820338905] | 5 |
| no quantizer CE | 0.2623728782 | 0.0183489654 | [0.2481355965, 0.2766101599] | 5 |
| repaired v5 shuffled labels | 0.2501694858 | 0.0231652156 | [0.2332203358, 0.2698304981] | 5 |

Paired repaired-minus-legacy mean = 0.0040678024, SD = 0.0234119207, bootstrap 95% CI = [-0.0135593116, 0.0216949165].

Paired repaired-minus-no-quantizer mean = 0.0033898324, SD = 0.0075798873, bootstrap 95% CI = [-0.0027118593, 0.0094915211].

Predeclared decisions: negative control valid = true; collapse rejected = false; generalization supported with limitations = false; quantization benefit supported = false.

## Reproducibility defect discovered

Rerunning the exact `main` SHA `2f59b4297e5978d4ce769ebe95adb363e1e75d7a` with the same seed/CLI/CPU workflow produced different one-step training losses: `10.853294372558594` and `10.34877872467041`.

The external ARC smoke, multi-seed benchmark package, matched baselines, paper manifest, and paired ablations remained byte-stable. The non-determinism was isolated to the `train_single.py` entry point.

Root cause: `LAMJEPA(cfg)` was instantiated before the requested seed was applied by `Trainer.__init__`, so model initialization was not governed by `--seed`.

## Repair and rerun

The repair seeds before model construction and adds an exact same-seed replay gate. On branch `repro-wave-2026-08-12`, SHA `fcd9af58c8e9fd4a8f8ac622a63d089f251c009e`, GitHub Actions run `31617532593` passed:

- seed: 1
- step: 1
- final loss: 11.704492568969727
- final accuracy: 0.0
- model tensors compared: 178
- model state exact: true
- metrics exact: true
- semantic metadata exact: true
- RNG state exact: true

Artifact ID: `9149892758`; artifact digest: `sha256:c08fd6a086fb83dd1b9a8c3b97c5f992bf1211c5a8c2dcd632dde175270808ba`.

The first version of the replay verifier failed after the training outputs already matched exactly because it compared serialized RNG tensor objects rather than their values. That verifier bug was preserved in Actions history, corrected to structural/value equality, and rerun to green. No scientific configuration was changed.

## Independent workflow-attempt replay boundary

After the seed-order repair was merged to `main` at SHA `b72a97a99769b278eb8ec75bc5eab62dc9599f29`, the exact-same-seed workflow was run twice as separate GitHub Actions attempts. Both attempts independently passed the within-job verifier: model state, metrics, semantic metadata, and RNG state matched exactly between the two clean output paths inside each attempt.

Across the two independent workflow attempts, the primary one-step outputs were stable:

- final loss: `11.704492568969727` in both attempts;
- final accuracy: `0.0` in both attempts;
- replay verifier JSON: byte-identical SHA-256 `1080efccc40d7a931451ec3fa5094113e877d54b4c16739cfe1861e22292f4af`.

However, a few floating-point submetrics differed at approximately the `1e-6` to `1e-7` level across attempts (for example `uni`, `conf`, `rub`, `plan`, and `eval_confidence`), and serialized PyTorch checkpoint bytes differed. Therefore the defensible claim is **semantic same-seed reproducibility within a fixed runner attempt plus numerically stable primary outputs across independent CPU CI attempts**, not byte-for-byte checkpoint reproducibility across separate runners.

This cross-attempt numerical drift does not alter the frozen ARC-v5 scientific result, but it is now retained as an explicit reproducibility limitation.
