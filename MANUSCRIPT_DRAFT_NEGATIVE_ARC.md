# LAM-JEPA on ARC-Challenge: A Reproducible Falsification-First Evaluation

**Manuscript status:** evidence-backed working draft; not submission-ready.  
**Scientific claim boundary:** the current ARC superiority and mechanism hypotheses are unsupported. This draft treats the negative/inconclusive result as the result rather than attempting to rescue it with the locked confirmatory test.

## Abstract

We evaluate LAM-JEPA on ARC-Challenge using a preregistered external-benchmark pipeline, a capacity-matched supervised comparator, mechanism ablations, a shuffled-label control, and a pinned pretrained comparator. The current evidence does not support a LAM-JEPA superiority claim. Across five frozen validation seeds, LAM-JEPA achieved accuracy `0.2549152542 ± 0.0129968064`, while the gradient-active-parameter-matched supervised model achieved `0.2664406780 ± 0.0154600058`, for a paired LAM-minus-matched difference of `-0.0115254237 ± 0.0140994131`. The planner and target-mechanism ablations likewise failed their preregistered contribution criteria: full minus `no_planner` was `+0.0047457627` with a 95% bootstrap interval `[0.0, 0.0142372881]`, while full minus `no_target` was `-0.0067796610` with interval `[-0.0135593220, 0.0]`. A bounded development comparison against pinned DeBERTa-v3-xsmall was also adverse (`0.15625` vs `0.21875`), although that comparison is characterization evidence rather than a standalone inferiority claim. A later trainability repair restored a bounded train-only gate but repaired validation remained negative/inconclusive. We preserve these outcomes, keep the ARC confirmatory test locked for this failed hypothesis, and document the experiment as a reproducible falsification case study. The results emphasize the value of matched baselines, frozen controls, adverse-result retention, and explicit stop rules in small-model representation-learning research.

## 1. Introduction

Representation-learning projects are vulnerable to a common failure mode: an interesting mechanism is treated as validated because the code runs, a favorable development slice exists, or a later repair improves trainability. This work instead asks a narrower question: under a frozen ARC-Challenge validation protocol, does the current LAM-JEPA implementation outperform an appropriate capacity-matched supervised baseline, and do its planner and target mechanisms contribute measurably under the preregistered criteria?

The answer on the current evidence is no. The purpose of this manuscript is therefore not to report a superiority result. It is to document a reproducible negative/inconclusive evaluation with enough protocol detail, controls, raw evidence, and claim boundaries to make the falsification scientifically useful.

The current contributions are:

1. a reproducible ARC-Challenge evaluation path with retained eligibility and exclusion evidence;
2. a gradient-active-parameter-matched supervised comparison;
3. five-seed mechanism ablations and a deterministic shuffled-label control;
4. a pinned pretrained-comparator path;
5. a documented trainability repair whose later validation did not rescue the original claim;
6. an explicit scientific stop rule that forbids use of the locked confirmatory test to rescue the failed hypothesis.

No claim of ARC superiority, planner benefit, target-mechanism benefit, quantization benefit, general benchmark superiority, or research completeness is made.

## 2. Related Work

**Citation gate:** references must be verified before insertion. Do not invent citations.

The final related-work section should cover four narrowly relevant areas:

- joint-embedding predictive architectures and latent prediction;
- representation learning on multiple-choice or reasoning benchmarks;
- matched-capacity and ablation methodology for neural architectures;
- reproducibility, preregistration, and negative-result reporting in machine learning.

This draft intentionally leaves bibliographic entries unresolved until each source is checked against the final claim wording.

## 3. Method

### 3.1 LAM-JEPA configuration

The manuscript must describe only architecture components that are verified against the implementation and frozen configuration. The current evidence package identifies planner and target-path mechanisms as required ablation targets, but this draft does not reconstruct additional architecture details from project naming alone.

**TODO before submission:** extract the exact module graph, tensor shapes, objective terms, quantization path, optimizer configuration, and inference path directly from the source/config files and lock them to the manuscript commit.

### 3.2 Capacity matching

The supervised comparator is matched using gradient-active parameter count under the ARC objective rather than nominal total parameter count. The frozen counts are:

| System | Gradient-active parameters |
|---|---:|
| LAM-JEPA | 86,372 |
| Matched supervised | 86,644 |

The ratio is `1.0031491687`.

### 3.3 Mechanism controls

The required validation controls include the full model, `no_planner`, `no_target`, and a deterministic shuffled-label control. These are evaluated under the same frozen five-seed validation budget.

### 3.4 Trainability repair

A train-only investigation localized a failure to the quantized latent path. The opt-in repair `arc-v5-stable-ema-residual-0.03125` passed its bounded trainability gate and was independently reproduced before repaired validation. It must be treated as a new repaired configuration, not retroactive evidence for the original hard-VQ mechanism.

## 4. Experimental Setup

### 4.1 Dataset and eligibility

The ARC-Challenge protocol preserves source order and uses a feature-only eligibility rule frozen before confirmatory access. The recorded eligible counts are:

- train: 1,117 / 1,119 rows;
- validation: 295 / 299 rows.

Excluded rows are retained as evidence. The locked ARC test was not used to adjudicate the failed superiority claim.

### 4.2 Frozen validation budget

The full-controls validation uses:

- five seeds;
- 20 epochs;
- batch size 32;
- learning rate `0.0003`;
- model steps 1;
- all 1,117 eligible training rows;
- all 295 eligible validation rows.

### 4.3 Pretrained comparator

The pinned comparator path uses `microsoft/deberta-v3-xsmall` at immutable revision `14809e4f1fe1895fcba8b258271a940c6ca45ec4`.

### 4.4 Hardware

**UNKNOWN IN THIS DRAFT.** Hardware must be copied from retained environment evidence or rerun metadata. Do not infer GPU/CPU type.

## 5. Results

### 5.1 Capacity-matched result

| System | Validation accuracy |
|---|---:|
| LAM-JEPA | 0.2549152542 ± 0.0129968064 |
| Matched supervised | 0.2664406780 ± 0.0154600058 |
| Paired LAM − matched | -0.0115254237 ± 0.0140994131 |

The frozen validation does not support superiority over the capacity-matched supervised baseline.

### 5.2 Mechanism ablations

| Configuration | Validation accuracy |
|---|---:|
| Full LAM-JEPA | 0.2549152542 ± 0.0129968064 |
| `no_planner` | 0.2501694915 ± 0.0129968064 |
| `no_target` | 0.2616949153 ± 0.0203954020 |
| Shuffled-label control | 0.2630508475 ± 0.0145011862 |

Paired effects:

- full minus `no_planner`: `+0.0047457627`, 95% bootstrap CI `[0.0, 0.0142372881]`;
- full minus `no_target`: `-0.0067796610`, 95% bootstrap CI `[-0.0135593220, 0.0]`.

Neither required mechanism criterion is supported by the frozen evidence.

### 5.3 Bounded pretrained comparison

On the recorded bounded development comparison:

- LAM-JEPA: `0.15625`;
- DeBERTa-v3-xsmall: `0.21875`;
- paired difference: `-0.0625`.

This is characterization evidence and should not be promoted into a final broad inferiority claim.

### 5.4 Repaired validation

The independent recomputation verdict for the repaired ARC-v5 validation is:

`VALID_NEGATIVE_OR_INCONCLUSIVE_VALIDATION`

The repair improved the declared trainability gate but did not establish the preregistered generalization or quantization-benefit claims.

## 6. Ablations

The current manuscript-ready ablations are `no_planner` and `no_target`. The final paper should not add post-hoc ablations and present them as preregistered. Any new architecture repair, benchmark, or mechanism variant must be versioned as a separate hypothesis with a new validation protocol.

The shuffled-label control should remain visible even though its numerical outcome is not favorable to an intuitive narrative. Its purpose is diagnostic, not rhetorical.

## 7. Limitations

1. The current scientific conclusion is specific to the frozen ARC line and tested configurations.
2. Failure to show superiority does not prove the architecture can never be useful on another task.
3. The bounded DeBERTa comparison is not a full final comparator study.
4. The repaired quantization path changes trainability but does not validate the original mechanism claim.
5. Hardware/environment details still need to be tied explicitly to retained run metadata in this manuscript.
6. Related-work references and publication metadata remain unresolved and must be verified rather than invented.
7. The locked confirmatory test cannot ethically be used as a rescue set after the validation hypothesis failed.

## 8. Discussion

The most useful result is methodological: the evidence pipeline prevented an executable research prototype from being mislabeled as a successful research result. Capacity matching removed one easy confound. Mechanism ablations prevented the full-model score from being attributed automatically to the planner or target path. The trainability repair demonstrated why engineering recovery and scientific validation must remain separate: a repair can make optimization behave better without producing the expected generalization advantage.

This makes the project a stronger candidate for a falsification-first technical report or reproducibility/negative-results submission than for a superiority paper in its current form.

## 9. Conclusion

Under the frozen ARC-Challenge validation protocol, the current LAM-JEPA evidence does not support superiority over a capacity-matched supervised baseline and does not validate the planner or target mechanisms. A later trainability repair also failed to produce a positive repaired-validation verdict. These negative/inconclusive results are preserved as first-class artifacts, and the confirmatory test remains locked for the failed line. The next scientific step should be a genuinely new preregistered hypothesis, not post-hoc tuning against the same validation evidence.

## References

**TO BE VERIFIED.** No references are inserted in this draft until each citation is checked.

## Appendix A — Reproducibility package gate

Before submission, the artifact package must pin:

- source commit;
- environment and dependency versions;
- exact seed list;
- frozen configs;
- data acquisition and eligibility commands;
- capacity-matched baseline command;
- ablation commands;
- pretrained-comparator command;
- evaluation/recomputation command;
- raw per-seed outputs;
- aggregate tables and bootstrap calculation;
- hardware metadata;
- license and citation metadata approved by the owner.

Current source-level publication packaging remains incomplete until license/citation/provenance work is closed.

## Appendix B — Claim table

| Claim | Status |
|---|---|
| Reproducible documented pipeline executes | Supported |
| ARC external-benchmark plumbing implemented | Supported |
| Five-seed frozen ARC validation executed | Supported |
| Capacity-matched baseline comparison executed | Supported |
| Planner improves ARC | Unsupported |
| Target mechanism improves ARC | Unsupported |
| LAM-JEPA beats matched supervised baseline | Unsupported |
| Repaired quantization improves generalization | Unsupported |
| LAM-JEPA is research-complete | False |
