# Development revision: scale-invariant linear CKA

This is a prospective engineering correction to the reusable representation
diagnostic. It does not revise the closed negative ARC study, its frozen
analysis, its manuscript, or the unopened confirmatory test.

## Observation and discriminating check

At parent commit `d40a4c4a3a9490a252265b1c54229506dd6a7b3a` (PR #203),
`linear_cka` converted every representation to float32 and applied an absolute
floor to covariance norms. A representation could therefore have self-CKA
near zero after a small change of units, or NaN after a large change of units.
Large float64 offsets erased otherwise representable differences. A constant
representation returned zero although its CKA denominator is undefined.

The mathematical prediction is specific: independent nonzero rescaling of the
two inputs preserves linear CKA, and an informative representation has
self-CKA one. The retained new suite failed 27 of 31 cases against that parent
source; the unmodified failure output is in the verification directory.

## Change and resulting contract

The implementation removes a common offset before centering, scales before
covariance reductions, and computes in detached float64 on the input device.
It uses equivalent feature-space or sample-space formulas according to their
intermediate storage size. Both paths are checked against an independent small
NumPy definition. Analytic fixtures cover orthogonal sample patterns, feature
rotations, large offsets, and finite scales from `1e-300` to `1e300`.

Undefined constant, nonfinite, complex, malformed, or unpaired inputs now raise
`ValueError`. The retained `eps` parameter is a nonnegative tolerance for final
roundoff at the [0, 1] boundary, not an absolute covariance regularizer. This is
an intentional compatibility change for future callers. Inputs, gradients, and
the global PyTorch RNG state are preserved.

## Verification and limits

All 31 new tests pass. Focused lint, syntax compilation, and source whitespace checks
pass. Exact commands, versions, file hashes, and raw output are retained in
`research/verification/cka_scale_20261010/receipt.json`.

Verification was CPU-only. Devices without float64 support are not established
as compatible; no accelerator performance or memory benchmark was run. Scaling
cannot recover distinctions already lost in the caller's input dtype. This is
the biased linear CKA statistic, not an unbiased estimator or a test of model
efficacy. No training, protected evaluation, dataset access, or paid compute was
used, and no independent reviewer is claimed for this revision.

Next action: review the explicit invalid-input and `eps` compatibility changes,
then integrate after PR #203. Any future scientific result using this utility
must identify this prospective code revision.
