# LAM-JEPA on ARC-Challenge: A Reproducible Falsification-First Evaluation

**Manuscript status:** evidence-backed working draft; not submission-ready.  
**Scientific claim boundary:** the current ARC superiority and mechanism hypotheses are unsupported. This draft treats the negative/inconclusive result as the result rather than attempting to rescue it with the locked confirmatory test.  
**Provenance companion:** `MANUSCRIPT_PROVENANCE.md`.

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

LAM-JEPA sits at the intersection of several established directions rather than introducing each ingredient from first principles. Joint-embedding predictive architectures predict representations of target content from context representations rather than reconstructing raw observations; I-JEPA established this approach for image representation learning [1]. Latent-action learning is also an established direction: LAPA learns discrete latent actions with a vector-quantized objective for action-model pretraining from video [2]. More directly relevant to the architectural combination, V-JEPA 2 couples JEPA-style video representations with an action-conditioned latent world model for planning [3], while subsequent latent-action world-model work studies learned action spaces and planning from action-free video [4]. These works mean that JEPA-style latent prediction, vector-quantized latent actions, and latent world-model planning should not be treated as individually novel contributions of the present study.

Our empirical question is narrower. We test whether one small LAM-JEPA configuration provides measurable benefit on ARC-Challenge under a frozen protocol with a capacity-matched supervised baseline and mechanism ablations. ARC itself was introduced by Clark et al. as a natural grade-school science question-answering challenge intended to stress reasoning beyond earlier benchmarks [5]. We do not claim benchmark novelty.

The present contribution is therefore best understood as a falsification-first evaluation of a specific combined architecture. The current evidence does not establish planner or target-path benefit, so the architecture ancestry and the empirical contribution must remain separate. The value of the package lies in the frozen controls, matched comparison, adverse-result retention, repair provenance, independent reruns, and the explicit decision not to unlock the confirmatory test after validation failure.

## 3. Method

### 3.1 Source-grounded LAM-JEPA configuration

The implementation used as the architectural source of truth defines a token/numeric multi-view encoder followed by a linear latent projector. The token branch embeds tokens with learned positional parameters, normalizes the token sequence and mean-pools it; an optional numeric input is projected separately. The two views are concatenated, fused by an MLP and normalized before projection into the latent space.

When enabled, an EMA-updated vector quantizer maps the online latent to a codebook vector using nearest-code assignment and a straight-through estimator. Its repository objective exposes commitment and codebook penalties. The resulting latent can pass through a sparse-memory retrieval module with a gated correction. When the planner is enabled, a latent-action policy selects a discrete action and a residual transition model predicts a mean and log-variance for the next latent state; the frozen ARC controls use a one-step rollout (`model_steps=1`). Disabling the planner removes this rollout path.

The target path contains a separately instantiated encoder and projector initialized from the online networks and updated by exponential moving average. When `use_target=False`, the implementation instead uses a detached online latent as the target representation. The model also exposes output-decoder, value, confidence, verifier, rubric, uncertainty and latent-summary heads. Their existence in the general package is not treated as evidence that each head contributes to ARC performance.

The general repository loss combines supervised cross-entropy with weighted latent alignment, variance, covariance, uniformity, geodesic, confidence-calibration, verifier, trajectory-consistency, rubric and quantization terms. For the paper, benchmark-specific runner/config behavior takes precedence over the generic library objective. `MANUSCRIPT_PROVENANCE.md` records this source boundary and requires the final Method lock to remain tied to the frozen ARC runner rather than to architecture naming.

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

- five seeds (`1, 2, 3, 4, 5`);
- 20 epochs;
- batch size 32;
- learning rate `0.0003`;
- model steps 1;
- all 1,117 eligible training rows;
- all 295 eligible validation rows.

### 4.3 Pretrained comparator

The pinned comparator path uses `microsoft/deberta-v3-xsmall` at immutable revision `14809e4f1fe1895fcba8b258271a940c6ca45ec4`.

### 4.4 Hardware and runtime boundary

The frozen full-controls runs were executed on GitHub-hosted Ubuntu runners using Python 3.11 and CPU execution, as recorded in `EVIDENCE_AUDIT_20260813.md` and the frozen workflow. The manuscript does not infer a specific CPU model because the retained evidence does not establish one. Independent reruns reproduced the aggregate scientific metrics and strict verifier output, while low-order floating-point differences remained in raw probability-bearing payloads; byte-exact raw floating-point identity is therefore not claimed.

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

The shuffled-label control is numerically competitive with the full configuration while still remaining below its separately frozen failure ceiling. We retain this adverse diagnostic rather than treating the ceiling pass as evidence for the representation mechanism. In a skeptical interpretation, the result increases concern that the current ARC signal is weak relative to optimization/noise effects.

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
5. The frozen run metadata establishes GitHub-hosted Ubuntu, Python 3.11 and CPU execution but not a specific CPU model.
6. Publication metadata, owner-approved licensing/citation information, and independent external review remain unresolved.
7. The locked confirmatory test cannot ethically be used as a rescue set after the validation hypothesis failed.
8. The architectural ingredients substantially overlap established JEPA, latent-action and world-model planning directions; current evidence does not establish a novel mechanism.

## 8. Discussion

The most useful result is methodological: the evidence pipeline prevented an executable research prototype from being mislabeled as a successful research result. Capacity matching removed one easy confound. Mechanism ablations prevented the full-model score from being attributed automatically to the planner or target path. The trainability repair demonstrated why engineering recovery and scientific validation must remain separate: a repair can make optimization behave better without producing the expected generalization advantage.

The originality audit also changes the paper framing. By 2026, JEPA-style representation learning, latent actions and latent world-model planning are established directions. Because the current ablations do not validate the proposed mechanism, the paper should not claim novelty from their combination alone. Its stronger contribution is a traceable falsification of a concrete small-model hypothesis under matched controls and explicit stop rules.

This makes the project a stronger candidate for a falsification-first technical report or reproducibility/negative-results submission than for a superiority paper in its current form.

## 9. Conclusion

Under the frozen ARC-Challenge validation protocol, the current LAM-JEPA evidence does not support superiority over a capacity-matched supervised baseline and does not validate the planner or target mechanisms. A later trainability repair also failed to produce a positive repaired-validation verdict. These negative/inconclusive results are preserved as first-class artifacts, and the confirmatory test remains locked for the failed line. The next scientific step should be a genuinely new preregistered hypothesis, not post-hoc tuning against the same validation evidence.

## References

[1] Mahmoud Assran, Quentin Duval, Ishan Misra, Piotr Bojanowski, Pascal Vincent, Michael Rabbat, Yann LeCun, Nicolas Ballas. *Self-Supervised Learning from Images with a Joint-Embedding Predictive Architecture*. arXiv:2301.08243, 2023.

[2] Seonghyeon Ye, Joel Jang, Byeongguk Jeon, Sejune Joo, Jianwei Yang, Baolin Peng, Ajay Mandlekar, Reuben Tan, Yu-Wei Chao, Bill Yuchen Lin, Lars Liden, Kimin Lee, Jianfeng Gao, Luke Zettlemoyer, Dieter Fox, Minjoon Seo. *Latent Action Pretraining from Videos*. arXiv:2410.11758, 2024.

[3] Mido Assran et al. *V-JEPA 2: Self-Supervised Video Models Enable Understanding, Prediction and Planning*. arXiv:2506.09985, 2025.

[4] Quentin Garrido, Tushar Nagarajan, Basile Terver, Nicolas Ballas, Yann LeCun, Michael Rabbat. *Learning Latent Action World Models In The Wild*. arXiv:2601.05230, 2026.

[5] Peter Clark, Isaac Cowhey, Oren Etzioni, Tushar Khot, Ashish Sabharwal, Carissa Schoenick, Oyvind Tafjord. *Think you have Solved Question Answering? Try ARC, the AI2 Reasoning Challenge*. arXiv:1803.05457, 2018.

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
- hardware/runtime metadata at the granularity actually retained;
- claim/table/figure provenance via `MANUSCRIPT_PROVENANCE.md`;
- license and citation metadata approved by the owner.

Current source-level publication packaging remains incomplete until license/citation/provenance work is closed.

## Appendix B — Claim table

| Claim | Status |
|---|---|
| Reproducible documented pipeline executes | Supported |
| ARC external-benchmark plumbing implemented | Supported |
| Five-seed frozen ARC validation executed | Supported |
| Capacity-matched baseline comparison executed | Supported |
| Aggregate scientific conclusion independently reproduced | Supported |
| Planner improves ARC | Unsupported |
| Target mechanism improves ARC | Unsupported |
| LAM-JEPA beats matched supervised baseline | Unsupported |
| Repaired quantization improves generalization | Unsupported |
| LAM-JEPA introduces JEPA/latent actions/planning as new general techniques | Unsupported |
| LAM-JEPA is research-complete | False |
