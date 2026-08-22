# ARC-v3 Prediction-Collapse Diagnostic — 2026-08-22

## Scope

This is a post-hoc diagnostic of the already-frozen negative/inconclusive ARC-v3 validation artifact. It does **not** alter the model, dataset, split, seed set, threshold, locked-test state, or scientific outcome. It strengthens the interpretation of the retained negative evidence and identifies the next falsification test.

Evidence source:

- scientific source SHA: `760aa7f9a73a177d5ff4ba7eb470f7e68ace63cb`;
- workflow run: `31203337502`, attempt `3`;
- retained artifact ID: `9162165932`;
- retained artifact ZIP SHA-256: `caa898f1ff046a337db9b5ddbffe1b332943a732868e2fd809abeda8ee89c30b`;
- raw full-results JSON SHA-256: `76aad8b1327e21470aeed137bac341b75b4fcf1f37e5394047642d395e8070f8`;
- diagnostic script: `scripts/analysis/analyze_arc_prediction_collapse.py`.

The locked ARC test remains unused.

## Recomputed validation label distribution

The 295 eligible validation examples have positional-label counts:

| Label | Count | Frequency |
|---|---:|---:|
| 0 | 63 | 0.2135593220 |
| 1 | 71 | 0.2406779661 |
| 2 | 78 | 0.2644067797 |
| 3 | 83 | 0.2813559322 |

These frequencies become important because the retained model outputs collapse to constant positional predictions.

## Primary finding: every retained run is a constant-class predictor

Across all 20 retained validation runs (full, `no_planner`, `no_target`, and shuffled-label negative control; five seeds each), each run predicts exactly one answer class for **all 295 validation examples**.

### Full LAM-JEPA

| Seed | Chosen class for all 295 examples | Accuracy | Exact matching validation-label frequency |
|---:|---:|---:|---:|
| 1 | 1 | 0.2406779661 | 0.2406779661 |
| 2 | 2 | 0.2644067797 | 0.2644067797 |
| 3 | 2 | 0.2644067797 | 0.2644067797 |
| 4 | 1 | 0.2406779661 | 0.2406779661 |
| 5 | 2 | 0.2644067797 | 0.2644067797 |

The full-model mean `0.2549152542` is therefore the mean accuracy of seed-dependent constant-class choices, not item-dependent ARC decisions.

The same collapse occurs in every `no_planner`, `no_target`, and shuffled-label-control seed.

## Probability-level input invariance

The collapse is stronger than a shared argmax. Within each run, the four-class probability vector is nearly invariant across all 295 different validation examples.

Across the 15 full/ablation runs, the maximum per-class probability range across examples is at most approximately `1.49e-7`. This is far below the diagnostic tolerance `1e-6`.

Thus the retained classifier is not merely making the same final choice frequently; its reported probability vector is effectively constant over the validation set at this numerical scale.

## Choice-reversal falsification

The retained artifact also contains choice-reversal predictions for the three scientific variants. Reversing the order of the four answer choices changes the correct positional label, so an item-sensitive positional classifier should generally respond to that transformation.

Instead, for all 15 full/ablation seed runs:

- the argmax class is preserved on **100% of examples** after choice reversal;
- the maximum probability change between original and reversed-choice evaluation is at most approximately `8.94e-8`.

This is direct retained-artifact evidence of near-complete insensitivity to the choice-order intervention.

## Mechanism-effect reinterpretation

The reported seed-level planner and target ablation deltas are exactly explained by switches between constant positional classes and the validation-set class frequencies.

### Full minus `no_planner`

Only seed 3 differs: full predicts class 2 for all examples while `no_planner` predicts class 1 for all examples.

`78/295 - 71/295 = 7/295 = 0.023728813559...`

This exactly equals the retained seed-3 full-minus-`no_planner` accuracy delta. The other four seed deltas are zero because the full and ablated runs choose the same constant class.

### Full minus `no_target`

Seeds 3 and 5 differ: full predicts class 2 while `no_target` predicts class 3.

`78/295 - 83/295 = -5/295 = -0.016949152542...`

This exactly equals each retained nonzero full-minus-`no_target` seed delta. The other three deltas are zero.

Therefore the observed mechanism deltas do not demonstrate item-level changes in reasoning quality. In this artifact they are fully accounted for by seed-dependent constant-class selection.

## Source-level clue, not yet a root-cause claim

The frozen source uses a very weak token mixing path: `TokenEncoder.encoder` is `nn.Identity()`, followed by tokenwise `LayerNorm` and mean pooling. The ARC benchmark then passes this representation through the LAM-JEPA backbone and a four-choice head.

The retained artifact does **not** record intermediate encoder variance, quantizer code indices, post-quantization variance, or final latent variance. Therefore this audit must not claim whether collapse originates in token encoding, quantization/codebook occupancy, memory/planner dynamics, the choice head, or their interaction.

## Claim boundary

### Supported

- The frozen ARC-v3 result remains negative/inconclusive.
- Every retained scientific/control validation run in attempt 3 is a constant-class predictor over all 295 eligible validation examples.
- Within-run output probabilities are nearly input-invariant at `1e-6` tolerance.
- Choice reversal preserves every scientific-variant argmax and changes probabilities only at ~`1e-7` scale.
- The retained planner/target accuracy deltas are exactly explained by constant-class label-frequency shifts.

### Not supported

- LAM-JEPA superiority on ARC.
- Planner benefit, target/EMA benefit, or repaired quantization/generalization benefit.
- Item-level ARC reasoning by this frozen model.
- A root-cause attribution to any single module before intermediate-state instrumentation.
- Any use of the locked ARC test to rescue this line.

## Next falsification test

Run a **diagnostic-only, no-retuning collapse-localization pass** on the frozen scientific source and existing train/validation split. Record, per seed and variant:

1. pre-quantization encoder/projector representation variance across examples;
2. quantizer code-index occupancy and number of unique codes;
3. post-quantization representation variance;
4. memory-corrected and final latent variance;
5. four-choice logit variance and argmax diversity;
6. the same quantities under choice reversal.

The test should identify the earliest stage at which example-dependent information disappears. It must not change model hyperparameters or access the locked test.

## Paper implication

If independently reviewed and reproduced from the retained artifact, the manuscript should describe the ARC failure more precisely as a **degenerate, near-input-invariant constant-class collapse under the frozen protocol**, rather than only reporting near-chance validation accuracy and unsupported mechanism effects.
