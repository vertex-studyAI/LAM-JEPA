# LAM-JEPA Results Ledger

**Reproducibility wave:** 12 August 2026  
**Repository head audited:** `2f59b4297e5978d4ce769ebe95adb363e1e75d7a`  
**Scientific evidence base retained by the repository:** `05c039fcc02c09c0aa1c1487596dcdd741ee6d51`  
**Current scientific verdict:** negative/inconclusive on the frozen ARC validation line; superiority and mechanism claims are unsupported.

## Research question

Under a frozen ARC-Challenge protocol and matched comparison budget, does LAM-JEPA improve validation accuracy over a capacity-matched supervised baseline, and do the planner and target mechanisms contribute positive measurable effects?

## Hypothesis and falsification rule

The superiority/mechanism hypothesis is supported only if the frozen validation protocol shows a reproducible positive advantage over the matched baseline and the predeclared mechanism criteria are met. Failure to meet those gates falsifies the current ARC superiority/mechanism claim. The locked ARC confirmatory test must not be used to rescue a failed validation hypothesis.

## Dataset / task

- ARC-Challenge train/validation under the repository's frozen eligibility protocol.
- Eligible rows: 1,117 / 1,119 train and 295 / 299 validation.
- Excluded rows remain retained as evidence and source ordering is preserved.
- Locked ARC test is not evidence for the failed validation claim and remains unused for rescue.

## Compared systems

- **Proposed:** full LAM-JEPA.
- **Simple/standard matched baseline:** supervised model matched on ARC-objective gradient-active parameter count.
- **Mechanism ablations:** `no_planner`, `no_target`.
- **Negative control:** deterministic shuffled-label control.
- **Strong pretrained characterization comparator:** pinned `microsoft/deberta-v3-xsmall` at revision `14809e4f1fe1895fcba8b258271a940c6ca45ec4`.

### Capacity match

- LAM-JEPA gradient-active parameters: `86,372`.
- Matched supervised gradient-active parameters: `86,644`.
- Ratio: `1.0031491687`.

## Frozen validation protocol

- Seeds: 5.
- Epochs: 20.
- Batch size: 32.
- Learning rate: `0.0003`.
- Model steps: 1.
- Train rows: all 1,117 eligible rows.
- Validation rows: all 295 eligible rows.

## Retained results

All values below are retained repository evidence; they are **not** claimed as a fresh local rerun in the 12 August reproducibility sandbox.

| Comparison | Mean validation accuracy | Dispersion / paired effect |
|---|---:|---:|
| Full LAM-JEPA | 0.2549152542 | ± 0.0129968064 |
| Matched supervised | 0.2664406780 | ± 0.0154600058 |
| Paired LAM − matched | — | -0.0115254237 ± 0.0140994131 |
| `no_planner` | 0.2501694915 | ± 0.0129968064 |
| `no_target` | 0.2616949153 | ± 0.0203954020 |
| Shuffled-label control | 0.2630508475 | ± 0.0145011862 |

Paired mechanism effects retained by the repository:

- full minus `no_planner`: `+0.0047457627`, 95% bootstrap CI `[0.0, 0.0142372881]`;
- full minus `no_target`: `-0.0067796610`, 95% bootstrap CI `[-0.0135593220, 0.0]`.

The shuffled-label control remained below the frozen `0.35` failure threshold, but neither required mechanism criterion was met.

### Strong pretrained comparator characterization

A bounded development comparison retained by the repository was adverse to LAM-JEPA:

- LAM-JEPA: `0.15625`;
- DeBERTa: `0.21875`;
- paired LAM − DeBERTa: `-0.0625`.

This is characterization evidence only and is not elevated into a standalone inferiority theorem.

## ARC-v5 trainability repair

The train-only repair `arc-v5-stable-ema-residual-0.03125` restored the repository's declared bounded trainability gate. It does not rescue the original hard-VQ mechanism claim. The repaired validation protocol was frozen before execution and the independent recomputation verdict remained:

`VALID_NEGATIVE_OR_INCONCLUSIVE_VALIDATION`

The repaired validation did not meet the predeclared generalization or quantization-benefit gates.

## Uncertainty and statistics

- Primary frozen validation uses five seeds and reports aggregate dispersion.
- Planner and target mechanism comparisons retain paired bootstrap confidence intervals.
- No significance claim is made beyond the reported analysis.
- A five-seed negative/inconclusive result is evidence against the frozen claim, but it is not proof that every possible LAM-JEPA variant is inferior on every task.

## Compute / environment provenance

The current repository contains reproducibility and container CI. The 12 August 2026 audit environment could inspect the connected GitHub repository but its execution sandbox had no outbound GitHub/dataset network, so it did **not** perform a fresh ARC dataset retrain. Fresh-execution status must therefore remain distinct from retained repository evidence.

Current head `2f59b4297e5978d4ce769ebe95adb363e1e75d7a` has successful push-triggered repository workflows for the research-claim boundary, ARC protocol QA, reproducibility CI, and container smoke packaging. Those checks strengthen execution/package provenance; they do not change the negative/inconclusive scientific conclusion.

## Supported conclusion

The defensible conclusion is that the repository implements an auditable LAM-JEPA experimental pipeline and that the current frozen ARC validation evidence does **not** support superiority over the matched supervised baseline or a validated positive planner/target contribution.

## Limitations

1. The central external benchmark line is ARC-specific.
2. Five seeds limit precision relative to a much larger repeated-run campaign.
3. The strong pretrained comparison is bounded characterization rather than the primary matched-capacity test.
4. The repaired quantizer establishes trainability under a declared train-only gate, not generalization superiority.
5. This reproducibility-wave audit did not freshly download ARC or retrain the full model because outbound dataset/network access was unavailable in the execution sandbox.
6. Future architectural hypotheses must be versioned and preregistered rather than tuned against the locked confirmatory test.
