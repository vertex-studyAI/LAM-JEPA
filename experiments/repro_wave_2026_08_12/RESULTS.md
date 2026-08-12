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

The first version of the replay verifier failed after the training outputs already matched exactly because it compared serialized RNG tensor objects rather than their values. That verifier bug was preserved in Actions history, corrected to structural/value equality, and rerun to green. No scientific configuration was changed.

## Independent workflow-attempt replay boundary

The repaired exact-same-seed workflow has now been independently replayed **five times**. The latest replay is Actions run `31631032761`, attempt 5, artifact `9158902481`, digest `sha256:9385a4d87fc8794379d51b61496f94e2ecc386a698166cc137ff34670cd59b89`, on head `6aceefa1f4afb0e01869eda2734744753965c976`.

Within each attempt, the verifier passes exact same-seed semantic replay for model state, metrics, semantic metadata, and RNG state. Across attempts 4 and 5, the primary one-step outputs remain exact:

- final loss: `11.704492568969727` in both attempts;
- final accuracy: `0.0` in both attempts;
- artifact file count: `35` in both attempts.

The artifact archives themselves are not byte-identical. Direct file comparison of attempts 4 and 5 again found low-order floating-point drift in secondary metrics and raw probabilities, typically around `1e-8` to `1e-6`, plus non-identical PyTorch checkpoint serialization. For example, the final training `cov` term moved from `22.98987579345703` to `22.989877700805664`, while final loss and accuracy did not change.

Therefore the defensible claim remains **semantic same-seed reproducibility within a runner attempt plus numerically stable primary outputs across independent CPU CI attempts**, not byte-for-byte checkpoint or full-float identity across runners.

This cross-attempt numerical drift does not alter the frozen ARC-v5 scientific result and is retained as an explicit reproducibility limitation.
