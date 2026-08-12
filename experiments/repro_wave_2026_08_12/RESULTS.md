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
