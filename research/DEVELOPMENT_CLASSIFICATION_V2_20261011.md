# Current-source admission of optional classification evaluator v2

## Outcome and scope

The existing private near-tie repair is now admitted against the actual Python source and all unit tests at `vertex-studyAI/LAM-JEPA@08010fb91df08f480e860029b9454ffc551cd132` (PR #203). The final local CPU run passed **89 tests**, with **zero failures, errors or skips**, under warnings-as-errors. The preserved older additive evaluator fails the exact near-tie fixture. No model was fitted and no scientific dataset, protected ARC outcome, retained study result or manuscript was loaded or rescored.

The completed ARC study remains **NEGATIVE and closed**, limited to its frozen tested configuration. The confirmatory ARC test remains unopened. This engineering result does not change submission status, establish a successor scientific hypothesis or advance a scientific checkpoint.

Current main and open PR heads #201–205 were inspected before this integration; the optional `classification_v2.py` module was absent. This change therefore brings an existing repaired utility into current source. It does not repeat the repair or replace the original evaluator. The inherited paired statistical report changes remain under their own PR #203 history.

## Failure, mechanism and implementation

The older additive scorer selected a class using the largest rounded softmax probability. Two represented float64 logits can be distinct while their probabilities round to the same value. With a logit gap of `2**-60`, the retained two-row regression has two correct logit winners but the old scorer reports accuracy `0.5` after first-index tie breaking on probabilities.

The existing private repair selects the class from the validated float64 logits and reads that class's probability for confidence. True logit ties continue to choose the first class. The repaired source and its 52-case test file were adopted byte-for-byte; this session adds three actual-current-model integration tests and the evidence/state records.

The optional entry points are:

```python
from lam_jepa.eval.classification_v2 import (
    ClassificationAccumulator,
    evaluate_classification_v2,
)
```

No existing caller or import export changes. The utility is available only when explicitly imported and used.

## Evaluation contract

The accumulator accepts finite, dense, floating logits with exact shape `(batch, classes)`, at least one example and two classes. Labels must be a dense integer tensor with exact shape `(batch,)` and values in the declared class range. It rejects broadcasting, a changing class count, empty evaluation, missing/invalid inputs, unrepresentable log probabilities and exceeded resource limits.

For example `i`, let `p_i = softmax(x_i)`, `a_i = argmax(x_i)`, and `y_i` be its label. The result contains sample-weighted accuracy, mean top-class probability, negative log likelihood `-log p_i[y_i]`, and unscaled multiclass Brier score `sum_c (p_i[c] - 1[c=y_i])**2`. Reductions use validated CPU float64 data. Sample-weighted mean updates avoid treating a short final batch as an independent full-size batch.

Top-label ECE uses fixed equal-width confidence bins: `[j/B, (j+1)/B)`, with the final bin also containing `1`. It is `sum_b |correct_b - sum_confidence_b| / N`. Empty bins have null accuracy/confidence. ECE is descriptive and depends on binning; a small ECE does not establish calibration or independent experimental units. The utility does not estimate uncertainty or authorize a scientific claim. Every summary includes `scientific_clearance: false`.

Successful accumulator updates retain scalar aggregates and `O(B)` bin records, rather than per-example predictions or autograd graphs. An invalid update leaves the previous aggregate unchanged. Defaults are 15 bins, 10,000,000 total examples and 10,000,000 elements per batch; bins are capped at 10,000. Temporary evaluation storage still scales with the current batch and class count. These checks do not bound total model/process memory.

The model wrapper requires an explicit loader and calls the current LAM forward contract with `sample_rollout=False` and `noise_std=0.0`. It validates token and label shapes before forward execution where possible. It uses evaluation mode and restores every module's original mode, including deliberately mixed modes, on success or failure. It neither moves nor restores model weights. An arbitrary custom model or loader that mutates buffers or RNG is not rolled back. The current LAM model's state/RNG preservation is tested on generated inputs in this admission; it is not a universal guarantee about other models.

## Retained evidence

| Attempt | Result | Evidence |
|---|---|---|
| Old additive source, distinct-logit witness | 1 expected failure; 51 deselected | `research/verification/classification_v2_20261011/near_tie_baseline_witness.log` and receipt; old scorer source retained in `baseline/` |
| Current-source acceptance 1 | Collection error, no tests executed: declared `tqdm` dependency absent | `current_unit_acceptance.log`, XML and receipt |
| Dependency repair | Isolated local install of `tqdm==4.70.1`; shared runtime unchanged | `dependency_install_receipt.json` |
| Current-source acceptance 2 | 89 passed; 0 failures/errors/skips | `current_unit_acceptance_02.log`, XML and receipt |

The 89 passes comprise all **34** current upstream unit cases, the **52** preserved optional evaluator cases, and **3** new current-model integration cases. The latter compare sample-weighted outputs with an independent direct forward/loss reference, reject broadcastable labels before any model forward, and propagate a constructed loader interruption while preserving the current model's modes, state and RNG. The unit file with `arc` in its name constructs a new model; it does not load ARC data.

The accepted pytest run took 2.38 seconds; the measured child command took 2.883 seconds wall time and 2.878 seconds CPU time, with observed child peak RSS 317,260 KiB. These are bounded engineering-run measurements, not training/inference benchmarks or forecasts of full-study cost. The environment used Python 3.12.14, PyTorch 2.14.1+cpu, pytest 9.1.1, NumPy 2.3.5 and tqdm 4.70.1, with one numeric thread and third-party pytest autoload disabled. The environment receipt records the exact observed versions.

## Lineage and reproducibility

The input private package SHA256 is `627a87141758eb5c6fbcfe8e2bdb5396a847eca6be542914eb243ba7b984c024`. The exact older additive scorer and failed run remain retained; failure is not overwritten by the passing run. The source manifest authenticates 143 retrieved files against Git blob identities and SHA256. The local materialization included all Python source, all unit tests and relevant metadata; it was not a complete repository checkout. Any remote draft must inherit the full original Git tree, preserving every unmodified path, mode and blob.

All upstream Python source and unit-test bytes remain unchanged. The canonical-state update appends this maintenance revision and retains the prior state snapshot. The new module SHA256 is `25a4e236033ffc12d9e2bb9b8293b97036a8177d16f90fa389adef198bb5c927`; preserved private tests SHA256 is `02ac9e48ea5b171ae26d3daca5a0292322d36a99e76e4f020cb6f9a80a5e6109`; actual-model tests SHA256 is `8b1ce152868bfaf21f734a1b5a37ff01d11a9085de7b0b3582b22588d06aaae2`.

In a clean checkout of the reviewed draft, install the repository requirements in an isolated environment, using the recorded versions to match this receipt. Reproduce the accepted scope with:

```bash
PYTHONPATH=src PYTHONDONTWRITEBYTECODE=1 PYTEST_DISABLE_PLUGIN_AUTOLOAD=1 OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1 python -m pytest tests/unit -q --maxfail=1 -p no:cacheprovider -W error --junitxml=/tmp/lam-classification-current-unit.xml
```

This is a complete run of the authenticated current unit directory, not the full repository CI, integration benchmark matrix or scientific protocol. No advantage in accuracy, calibration, runtime or architecture efficacy is claimed. The next engineering action is independent review of this optional utility. A new scientific architecture claim requires a separately versioned successor study, a new falsifier and a protected protocol.
