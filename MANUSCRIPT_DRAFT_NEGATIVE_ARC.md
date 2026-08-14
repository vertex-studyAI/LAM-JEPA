# LAM-JEPA on ARC-Challenge: A Reproducible Falsification-First Evaluation

**Manuscript status:** evidence-backed working draft; internally close to external-review packaging, but not publication-ready.  
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

### 2.1 Joint-embedding predictive architectures

Joint-Embedding Predictive Architectures predict target representations from context representations instead of directly reconstructing the original input. I-JEPA provides a canonical image-domain instantiation with a learned target encoder updated by exponential moving average. LAM-JEPA shares the broad representation-space prediction and online/target asymmetry, so neither latent-space prediction nor EMA target networks are claimed as novel here.

More recent JEPA work further narrows the novelty boundary. V-JEPA 2 extends self-supervised video representations to action-conditioned latent world modeling and planning. Current 2026 work also studies latent-action world models, physical-state grounding for JEPA world models, and joint-embedding world modeling for vision-language-action policies. Consequently, the combination of predictive latent representations, latent actions, grounding, and planning cannot by itself support a mechanism-novelty claim for LAM-JEPA. The present manuscript instead evaluates a particular reasoning-oriented configuration and reports what its frozen controls do and do not support.

### 2.2 Discrete latent representations

Vector-quantized latent learning predates this project; VQ-VAE is a foundational example. LAM-JEPA's quantizer therefore constitutes an implementation/design choice rather than a novel discrete-representation mechanism. This distinction matters because the later ARC-v5 engineering repair altered trainability around the quantized path but did not establish a quantization generalization benefit.

### 2.3 ARC and pretrained language-model characterization

The AI2 Reasoning Challenge (ARC) was introduced as a grade-school science question-answering benchmark with separate Challenge and Easy sets. This study uses the frozen ARC-Challenge train/validation path and deliberately keeps the confirmatory test unavailable after the validation hypothesis fails. A pinned DeBERTa-v3-xsmall checkpoint is retained as a bounded pretrained characterization comparator. Its adverse comparison is not promoted into a broad statement that LAM-JEPA is inferior to all pretrained language models.

### 2.4 Ablation, falsification, and reproducibility

The methodological emphasis in this paper is conservative rather than novel: match a competent baseline, freeze the protocol, preserve adverse controls, keep paired seed structure, distinguish engineering repairs from scientific validation, and stop rather than opening a locked test set after a failed validation hypothesis. Gradient-active parameter matching is used here to reduce one capacity confound; it is not claimed as a new experimental principle.

The verified bibliography and remaining literature tasks are maintained in `RELATED_WORK_TODO.md`. Bibliographic entries below are restricted to sources whose metadata has been checked; a fuller manuscript-specific related-work sweep remains required before submission.

## 3. Method

### 3.1 Source-grounded architecture

The current repository implementation receives token inputs and optional numeric features through a multi-view encoder. The token path embeds tokens with learned positional parameters, normalizes them, and mean-pools across the sequence; the numeric path projects a fixed-width numeric vector. The two representations are concatenated, fused by an MLP, normalized, and projected into the model's latent dimension.

When enabled, an EMA-updated vector quantizer maps the online latent to a codebook vector with straight-through gradients and a commitment/codebook penalty. An optional sparse-memory module retrieves a memory vector and applies a gated correction. The planner is implemented by a latent-action module: a policy selects one of a finite set of latent actions and a residual transition network predicts a mean and log-variance for the next latent state. The frozen ARC full-controls experiment uses `model_steps=1`; disabling the planner removes this rollout path.

The target path contains a separate encoder and projector initialized from the online networks and updated by exponential moving average using the configured momentum. When `use_target=False`, the target representation is replaced by a detached online representation. The model also exposes output-decoder, value, confidence, verifier, rubric, uncertainty, and latent-summary heads. The ARC paper makes claims only about components actually exercised and controlled by its frozen benchmark path.

### 3.2 General repository objective and benchmark boundary

The general source implementation defines a composite objective containing supervised cross-entropy plus weighted latent alignment, variance, covariance, uniformity, geodesic, confidence-calibration, verifier, trajectory-consistency, rubric, and quantization terms. The benchmark paper does not infer that every repository objective term is scientifically validated. Any ARC-specific simplification or override in the frozen benchmark runner takes precedence for describing the experiment, and the final method lock must remain traceable to the frozen runner/config rather than to project naming.

### 3.3 Capacity matching

The supervised comparator is matched using gradient-active parameter count under the ARC objective rather than nominal total parameter count. The frozen counts are:

| System | Gradient-active parameters |
|---|---:|
| LAM-JEPA | 86,372 |
| Matched supervised | 86,644 |

The ratio is `1.0031491687`.

### 3.4 Mechanism controls

The required validation controls include the full model, `no_planner`, `no_target`, and a deterministic shuffled-label control. These are evaluated under the same frozen five-seed validation budget.

### 3.5 Trainability repair

A train-only investigation localized a failure to the quantized latent path. The opt-in repair `arc-v5-stable-ema-residual-0.03125` passed its bounded trainability gate and was independently reproduced before repaired validation. It is treated as a new repaired configuration, not retroactive evidence for the original hard-VQ mechanism.

## 4. Experimental Setup

### 4.1 Dataset and eligibility

The ARC-Challenge protocol preserves source order and uses a feature-only eligibility rule frozen before confirmatory access. The recorded eligible counts are:

- train: 1,117 / 1,119 rows;
- validation: 295 / 299 rows.

Excluded rows are retained as evidence. The locked ARC test was not downloaded or used to adjudicate the failed superiority claim.

### 4.2 Frozen validation budget

The full-controls validation uses:

- seeds `1, 2, 3, 4, 5`;
- 20 epochs;
- batch size 32;
- learning rate `0.0003`;
- model steps 1;
- all 1,117 eligible training rows;
- all 295 eligible validation rows;
- CPU execution.

The workflow independently verifies this budget and asserts that the locked test was not evaluated.

### 4.3 Pretrained comparator

The pinned comparator path uses `microsoft/deberta-v3-xsmall` at immutable revision `14809e4f1fe1895fcba8b258271a940c6ca45ec4`.

### 4.4 Execution environment

The retained full-controls workflow executes on a GitHub-hosted `ubuntu-latest` runner, configures Python `3.11`, installs the CPU PyTorch wheel, and invokes the benchmark with `--device cpu`. The exact physical CPU model of the hosted runner is not retained in the current evidence and is therefore not claimed.

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

## 6. Ablations and robustness checks

The manuscript-ready architecture ablations are `no_planner` and `no_target`; the shuffled-label control is retained as a diagnostic negative control. The final paper does not relabel post-hoc experiments as preregistered. Any new architecture repair, benchmark, mechanism variant, or causal diagnostic must be versioned as a separate hypothesis with a new protocol.

Independent reruns reproduce the aggregate scientific conclusion and strict verifier output, but the raw probability-bearing payloads are not byte-identical across independent hosted runners. The retained comparison found low-order numerical drift while preserving non-numeric content and the declared conclusion. This is reported as the reproducibility boundary rather than hidden.

## 7. Failure Analysis

Three failure boundaries matter for interpretation.

First, the full model does not outperform the capacity-matched supervised comparator under the frozen validation. Second, removing the planner does not produce a sufficiently adverse change to support a planner-benefit claim, while removing the target path has a numerically higher mean accuracy than the full model. Third, a trainability repair around the quantized path improves a bounded optimization gate but does not convert the later validation into positive generalization evidence.

These outcomes are consistent with several possible explanations—optimization difficulty, prediction-support collapse, component non-use, confounding among objective terms, or a mismatch between the architecture and the benchmark—but the current experiment does not identify one of these as a proven causal explanation. Mechanism language is therefore deliberately weaker than outcome language.

## 8. Limitations

1. The current scientific conclusion is specific to the frozen ARC line and tested configurations.
2. Failure to show superiority does not prove the architecture can never be useful on another task.
3. The bounded DeBERTa comparison is not a full final comparator study.
4. The repaired quantization path changes trainability but does not validate the original mechanism claim.
5. Hosted-runner OS/Python/CPU execution is recorded, but exact CPU model and full hardware microarchitecture are not retained.
6. The related-work audit is substantially improved but still requires a final manuscript-specific sweep before submission.
7. The locked confirmatory test cannot be used as a rescue set after the validation hypothesis failed.
8. The current ablations establish lack of measured component benefit under this protocol; they do not prove why the components failed.

## 9. Reproducibility

The frozen full-controls workflow downloads only checksum-addressed ARC train and validation splits, asserts the absence of the test parquet, runs the fixed five-seed/20-epoch controls on CPU, and independently verifies the resulting budget and claim boundary. The evidence audit retains two independent full scientific reruns, artifact digests, the pre-fix deterministic seed-order defect, the minimal seed-before-model-construction repair, and the unchanged negative/inconclusive conclusion.

`MANUSCRIPT_PROVENANCE.md` records the claim-to-artifact chain and the allowed figure/table sources. `REPRODUCE.md`, `RELEASE_PROVENANCE.md`, and the frozen protocol/workflow files define the executable package. A final public release still requires an immutable release revision plus owner-approved legal and citation metadata.

## 10. Broader Impact and Research Practice

This work does not demonstrate an educationally effective or generally superior reasoning system. Its broader value, if any, is methodological: preserving failed hypotheses, strong controls, negative ablations, exact provenance, and stop rules can reduce false positive research narratives. Negative results are only useful when the tested question and its limitations remain narrow enough that others can tell what was actually falsified.

## 11. Discussion

The evidence pipeline prevented an executable research prototype from being mislabeled as a successful research result. Capacity matching removed one easy confound. Mechanism ablations prevented the full-model score from being attributed automatically to the planner or target path. The trainability repair demonstrated why engineering recovery and scientific validation must remain separate: a repair can make optimization behave better without producing the expected generalization advantage.

The literature boundary reinforces that framing. Representation-space prediction, EMA targets, vector quantization, latent-action modeling, and planning all have substantial prior art. The current package is therefore stronger as a falsification-first technical report or reproducibility/negative-results submission than as a claimed new JEPA mechanism.

## 12. Conclusion

Under the frozen ARC-Challenge validation protocol, the current LAM-JEPA evidence does not support superiority over a capacity-matched supervised baseline and does not validate the planner or target mechanisms. A later trainability repair also failed to produce a positive repaired-validation verdict. These negative/inconclusive results are preserved as first-class artifacts, and the confirmatory test remains locked for the failed line. The next scientific step, if any, should be a genuinely new preregistered hypothesis rather than post-hoc tuning against the same validation evidence.

## References — verified core anchors

1. Assran, M., Duval, Q., Misra, I., Bojanowski, P., Vincent, P., Rabbat, M., LeCun, Y., & Ballas, N. *Self-Supervised Learning from Images with a Joint-Embedding Predictive Architecture.* arXiv:2301.08243, 2023.
2. Assran, M. et al. *V-JEPA 2: Self-Supervised Video Models Enable Understanding, Prediction and Planning.* arXiv:2506.09985, 2025.
3. van den Oord, A., Vinyals, O., & Kavukcuoglu, K. *Neural Discrete Representation Learning.* arXiv:1711.00937, 2017/2018.
4. Clark, P., Cowhey, I., Etzioni, O., Khot, T., Sabharwal, A., Schoenick, C., & Tafjord, O. *Think you have Solved Question Answering? Try ARC, the AI2 Reasoning Challenge.* arXiv:1803.05457, 2018.
5. He, P., Gao, J., & Chen, W. *DeBERTaV3: Improving DeBERTa using ELECTRA-Style Pre-Training with Gradient-Disentangled Embedding Sharing.* arXiv:2111.09543; ICLR 2023.

**Related-work completion note:** `RELATED_WORK_TODO.md` records additional current JEPA/action-world-model sources and the remaining ablation/reproducibility literature checks. The final submission bibliography must be generated from fully verified metadata rather than extended from memory.

## Appendix A — Reproducibility package gate

Before public release/submission, the artifact package must pin:

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
- hardware boundary (`ubuntu-latest`, Python 3.11, CPU; exact CPU model unclaimed);
- figure/table generation commands tied to retained machine-readable sources;
- license and citation metadata approved by the owner.

Current scientific reproduction is strong; public release packaging remains incomplete until legal/citation and external-review gates are closed.

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
| Locked ARC test used to rescue failed validation | False / prohibited |
| LAM-JEPA is research-complete | False |
| LAM-JEPA is publication-ready | Not yet; legal/bibliographic/external-review gates remain |
