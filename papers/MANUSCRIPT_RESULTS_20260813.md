# Manuscript-Ready Results Text — LAM-JEPA ARC-Challenge

This text is intended for a Results / Reproducibility section. It deliberately preserves the negative/inconclusive outcome and should not be edited into a superiority claim without new preregistered evidence.

## Main ARC-Challenge result

We evaluated LAM-JEPA under a frozen ARC-Challenge development-validation protocol using the checksum-addressed training and validation splits, a predeclared exactly-four-choice eligibility rule, five paired seeds, 20 epochs, batch size 32, learning rate `3e-4`, and one model planning step. The final eligible sample contained 1,117 training examples and 295 validation examples. The locked ARC test split was not accessed for this hypothesis line.

LAM-JEPA achieved mean validation accuracy `0.2549 ± 0.0130` across five seeds. A gradient-active-parameter-matched supervised baseline achieved `0.2664 ± 0.0155`, with a paired LAM-JEPA-minus-baseline difference of `−0.0115 ± 0.0141`. Thus, the frozen validation evidence does not support a LAM-JEPA superiority claim on ARC-Challenge.

## Mechanism ablations

The full LAM-JEPA model achieved `0.2549 ± 0.0130` validation accuracy, compared with `0.2502 ± 0.0130` for the `no_planner` ablation and `0.2617 ± 0.0204` for the `no_target` ablation. The paired full-minus-`no_planner` effect was `+0.00475` with a retained bootstrap 95% interval `[0.0, 0.01424]`; the paired full-minus-`no_target` effect was `−0.00678` with interval `[−0.01356, 0.0]`. Neither predeclared mechanism criterion was met. Accordingly, the current ARC evidence does not establish a reproducible planner or target-path benefit.

A deterministic shuffled-label control achieved `0.2631 ± 0.0145`, below the frozen control-failure ceiling of `0.35`. Passing this control gate does not rescue the main mechanism hypothesis; it only indicates that the particular predeclared failure condition was not triggered.

## Strong comparator characterization

A separately bounded development comparison against pinned `microsoft/deberta-v3-xsmall` was also adverse to LAM-JEPA (`0.15625` versus `0.21875`, paired difference `−0.0625`). Because this comparison was bounded and not a full matched confirmatory trial, we treat it as characterization evidence rather than a standalone inferiority test. It nevertheless reinforces the conclusion that the current data do not support a superiority claim.

## Independent reproduction

We reran the frozen full-controls workflow twice on independent GitHub-hosted runner attempts without changing the scientific source or protocol. Both attempts completed the same five-seed, 20-epoch validation and independent verification successfully. The aggregate means, standard deviations, paired seed-level mechanism deltas, bootstrap intervals, negative-control verdict, and strict verifier outputs were exactly equal between the two reruns.

The raw per-example probability payloads were not byte-identical across independent runners: 35,526 numeric leaves differed at low order, with a maximum observed difference of approximately `5.9e-4`, while no non-numeric leaf differed. These differences did not alter any aggregate metric or scientific verdict. We therefore claim reproducibility of the aggregate scientific conclusion and verifier decision, not byte-exact floating-point identity of all raw probabilities.

## Reproducibility defect and repair

During the reproducibility audit, we identified a software defect in the deterministic training path: model initialization occurred before applying the requested seed. Before repair, nominally identical one-step runs with the same requested seed produced different losses (`10.8533` versus `10.3488`). The smallest versioned repair applied the seed before model construction while preserving subsequent trainer-side seeding.

After repair, six independently verified replay attempts preserved exact final loss (`11.704492568969727`) and final accuracy (`0.0`) across attempts, although secondary floating-point quantities and serialized checkpoint bytes could still differ across independent runners. We treat the seed-order change as a software reproducibility repair rather than a new scientific intervention; it does not change the frozen ARC result or support the original superiority hypothesis.

## Repaired quantized-latent line

A separate train-only investigation localized a trainability failure to the quantized latent path and introduced the narrowly scoped `arc-v5-stable-ema-residual-0.03125` repair. Although this repair restored its predeclared bounded trainability gate, the subsequently frozen validation remained `VALID_NEGATIVE_OR_INCONCLUSIVE_VALIDATION` and did not satisfy the generalization or quantization-benefit criteria. The repair therefore does not rescue the original hard-vector-quantization mechanism claim.

## Interpretation

The primary contribution of the current ARC study is a reproducible negative result and an auditable failure analysis rather than a positive performance result. The pipeline supports checksum-addressed external benchmark execution, matched-capacity comparison, strong-comparator characterization, multi-seed controls, mechanism ablations, independent verification, and retained adverse evidence. Under that protocol, however, LAM-JEPA did not outperform the matched supervised baseline and the planner, target, and repaired quantization mechanisms did not receive validation support.

## Limitations

The result is limited to ARC-Challenge development validation and five seeds. The locked test split remains intentionally unused for this failed hypothesis line. The pretrained comparison is bounded rather than a full confirmatory matched trial, and independent runners show low-order floating-point drift in individual probabilities. No claim is made about educational effectiveness, general benchmark superiority, or general intelligence.

## Recommended table caption

**Table X. Frozen ARC-Challenge validation results across five paired seeds.** Values are mean accuracy ± sample standard deviation. LAM-JEPA does not exceed the gradient-active-parameter-matched supervised baseline in mean accuracy. Component ablations do not satisfy the predeclared planner or target-path benefit criteria. The shuffled-label control remains below its frozen failure ceiling. The locked test split was not accessed.

## Recommended figure caption

**Figure X. Paired mechanism-effect estimates under the frozen ARC-Challenge validation protocol.** Points show the mean seed-paired change in validation accuracy for the full model relative to each ablation; bars show the retained bootstrap 95% intervals. Both intervals include zero at a boundary, and neither predeclared mechanism criterion is met. The figure is descriptive of the frozen validation evidence and is not a significance claim.
