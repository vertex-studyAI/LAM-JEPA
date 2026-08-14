# LAM-JEPA on ARC-Challenge: A Reproducible Falsification-First Evaluation

**Manuscript status:** evidence-backed working draft; method source-locked, but not submission-ready.  
**Scientific execution commit:** `760aa7f9a73a177d5ff4ba7eb470f7e68ace63cb`.  
**Scientific claim boundary:** the current ARC superiority and mechanism hypotheses are unsupported. This draft treats the negative/inconclusive result as the result rather than attempting to rescue it with the locked confirmatory test.

## Abstract

We evaluate LAM-JEPA on ARC-Challenge using a frozen external-benchmark pipeline, a gradient-active-parameter-matched supervised comparator, mechanism ablations, a shuffled-label control, and a pinned pretrained comparator. The current evidence does not support a LAM-JEPA superiority claim. Across five frozen validation seeds, LAM-JEPA achieved accuracy `0.2549152542 ± 0.0129968064`, while the gradient-active-parameter-matched supervised model achieved `0.2664406780 ± 0.0154600058`, for a paired LAM-minus-matched difference of `-0.0115254237 ± 0.0140994131`. The planner and target-path ablations likewise failed their predeclared contribution criteria: full minus `no_planner` was `+0.0047457627` with a 95% bootstrap interval `[0.0, 0.0142372881]`, while full minus `no_target` was `-0.0067796610` with interval `[-0.0135593220, 0.0]`. A bounded development comparison against pinned DeBERTa-v3-xsmall was also adverse (`0.15625` vs `0.21875`), although that comparison is characterization evidence rather than a standalone inferiority claim. A later trainability repair restored a bounded train-only gate but repaired validation remained negative/inconclusive. We preserve these outcomes, keep the ARC confirmatory test locked for this failed hypothesis line, and document the experiment as a reproducible falsification case study. The results emphasize the value of matched baselines, frozen controls, adverse-result retention, and explicit stop rules in small-model representation-learning research.

## 1. Introduction

Representation-learning projects are vulnerable to a common failure mode: an interesting mechanism is treated as validated because the code runs, a favorable development slice exists, or a later repair improves trainability. This work instead asks a narrower question: under a frozen ARC-Challenge validation protocol, does the current LAM-JEPA implementation outperform an appropriate capacity-matched supervised baseline, and do its planner and target mechanisms contribute measurably under the predeclared criteria?

The answer on the current evidence is no. The purpose of this manuscript is therefore not to report a superiority result. It is to document a reproducible negative/inconclusive evaluation with enough protocol detail, controls, raw evidence, and claim boundaries to make the falsification scientifically useful.

The current contributions are:

1. a reproducible ARC-Challenge evaluation path with retained eligibility and exclusion evidence;
2. a gradient-active-parameter-matched supervised comparison;
3. five-seed planner/target mechanism ablations and a deterministic shuffled-label control;
4. a pinned pretrained-comparator characterization path;
5. a documented trainability repair whose later validation did not rescue the original claim;
6. independent workflow reruns that reproduce the aggregate conclusion while exposing low-order floating-point drift; and
7. an explicit scientific stop rule that forbids use of the locked confirmatory test to rescue the failed hypothesis.

No claim of ARC superiority, planner benefit, target-mechanism benefit, quantization benefit, general benchmark superiority, educational effectiveness, or research completeness is made.

## 2. Related Work and Novelty Boundary

Joint-embedding predictive architectures, moving-average target networks, vector quantization, latent-action learning, and action-conditioned predictive world models are established research directions [1–7]. ARC itself is an established reasoning benchmark [8], and the pinned strong comparator comes from the DeBERTaV3 family [9]. The present study therefore does not claim to introduce latent prediction, EMA targets, vector-quantized latents, latent actions, or JEPA for language.

Our empirical question is narrower: does one small, frozen LAM-JEPA configuration provide measurable benefit on ARC-Challenge under a gradient-active-capacity-matched supervised comparison and predeclared mechanism ablations? The current answer is negative/inconclusive. The defensible contribution is therefore a scoped empirical falsification and reproducibility package, not a substantial new architecture or theory claim. `ORIGINALITY_AUDIT.md` records the closest-work map, prohibited novelty claims, and the required final literature-refresh gate.

## 3. Method

### 3.1 Source-locked ARC input path

This section describes the implementation at scientific execution commit `760aa7f9a73a177d5ff4ba7eb470f7e68ace63cb`, not a later repository head or the broader aspirational architecture described in historical documents.

Each ARC row is formatted as the question followed by indexed answer choices. Text is lower-cased, whitespace-split, truncated to at most 96 tokens, and each token is deterministically mapped into a vocabulary of size 256 using a BLAKE2b-based hash; shorter sequences are zero-padded. The ARC numeric input is a one-dimensional zero vector, which the encoder pads to the configured numeric input width of 32.

The token path uses a learned `256 × 32` embedding table and learned positional embeddings, followed by layer normalization and mean pooling. Critically, `TokenEncoder.encoder` at the scientific commit is `nn.Identity()`: despite configuration fields named `num_layers` and `num_heads`, the frozen ARC token path contains **no Transformer encoder block**. The numeric branch is a linear projection into 32 dimensions. Token and numeric features are concatenated, fused by an MLP with hidden width 64, and normalized to produce a 32-dimensional representation. A linear projector maps that representation to a 32-dimensional planning space.

This architectural fact sharply limits the scope of the result. The experiment evaluates this small hashing-based representation model and must not be generalized to modern pretrained language encoders or JEPA-for-language systems broadly.

### 3.2 Quantization, retrieval, latent action, and target path

With quantization enabled, the projected state is assigned to the nearest of 32 learned 32-dimensional code vectors by Euclidean distance. During training, codebook counts and weights are updated with EMA decay `0.99`; the quantization loss is the sum of commitment and codebook mean-squared errors, and the returned quantized state uses a straight-through estimator.

With memory enabled, the model queries a learned key/value memory of capacity 64. The retrieval implementation uses the top 16 similarity scores by default, applies a softmax over the selected values, and passes the retrieval through a learned gated correction. The frozen ARC experiment does not establish that this retrieval is useful episodic memory or that it contributes causally to accuracy.

The latent-action module has eight discrete action indices. A linear policy produces action probabilities; an action embedding and the current state, retrieved memory, and zero uncertainty/context vector are concatenated and passed through learned transition heads for a residual latent-state update. The frozen ARC budget uses exactly one model step. Training uses the model's stochastic action/transition path; evaluation uses deterministic one-step inference with rollout noise set to zero. The action indices are not externally annotated as human-interpretable reasoning operations and are therefore treated only as learned latent categories.

The target encoder and projector begin synchronized with the online encoder/projector and are updated after optimizer steps with EMA momentum `0.996`. When the target path is enabled, the target representation of the same ARC input is computed under `torch.no_grad()`. In the `no_target` ablation, the target is the detached online projected state. Thus the ablation tests the EMA-target path in this implementation; it does not remove every alignment-related term from the objective.

After the one-step latent rollout, a 32-dimensional MLP `latent_summary_head` produces the representation consumed by a dedicated four-choice ARC classification head. The general `LAMJEPA` module also computes value, confidence, verifier, rubric, uncertainty, and vocabulary-decoder outputs, but these auxiliary heads are not terms in the frozen ARC classification objective. Capacity matching therefore uses gradient-active parameters under the actual ARC objective rather than every nominal parameter in the general model.

### 3.3 Frozen ARC objective and optimization

For ARC, the optimized LAM-JEPA loss is:

`cross_entropy(choice_logits, labels) + 0.5 * cosine_alignment(z_q, target_z) + 0.25 * quant_loss + 0.25 * trajectory_loss`.

The trajectory term is mean squared error between rolled-out latent states and the detached quantized state when a rollout exists. This ARC-specific objective is narrower than the repository's generic multi-task `total_loss`; the generic auxiliary loss must not be presented as the loss used for the frozen ARC result.

Training uses AdamW with learning rate `3e-4`, seeds `[1,2,3,4,5]`, 20 epochs, batch size 32, one model step, gradient clipping at norm `1.0`, and an EMA target update after each optimizer step. Evaluation uses deterministic one-step inference and softmax over the four answer choices.

### 3.4 Capacity matching

The supervised comparator is matched using gradient-active parameter count under the ARC objective rather than nominal total parameter count. The frozen counts are:

| System | Gradient-active parameters |
|---|---:|
| LAM-JEPA | 86,372 |
| Matched supervised | 86,644 |

The ratio is `1.0031491687`, within the frozen `[0.99, 1.01]` allowance. The matched baseline excludes the EMA target encoder, JEPA alignment machinery, latent-action planner, sparse memory, and vector quantizer and is trained on the same eligible rows, epochs, batch size, seed set, optimizer family, and evaluation passes.

### 3.5 Mechanism controls

The required validation controls include the full model, `no_planner`, `no_target`, and a deterministic shuffled-training-label control. The planner and target contribution rule requires a paired full-minus-ablation mean accuracy of at least `0.01` with a paired bootstrap 95% interval excluding zero. The shuffled-label failure ceiling is `0.35` validation accuracy. Passing the negative-control ceiling is diagnostic only and cannot rescue a failed primary hypothesis.

### 3.6 Trainability repair

A separate train-only investigation localized a failure to the quantized latent path. The opt-in repair `arc-v5-stable-ema-residual-0.03125` passed its bounded trainability gate and was independently reproduced before repaired validation. It is a separately versioned repaired configuration, not retroactive evidence for the original hard-VQ mechanism.

## 4. Experimental Setup

### 4.1 Dataset and eligibility

The frozen ARC-Challenge protocol preserves source order and applies a feature-only eligibility rule: retain a row if and only if it has exactly four choices. Eligibility decisions are prohibited from using answer labels, predictions, model outputs, or performance. The recorded eligible counts are:

- train: 1,117 / 1,119 rows;
- validation: 295 / 299 rows.

Excluded row IDs, counts, and ordered-ID digests are retained as evidence. The locked ARC test was not downloaded or evaluated to adjudicate this failed hypothesis line.

### 4.2 Frozen validation budget

The full-controls validation uses five seeds `[1,2,3,4,5]`, 20 epochs, batch size 32, learning rate `0.0003`, one model step, all 1,117 eligible training rows, all 295 eligible validation rows, and CPU execution. The primary metric is multiple-choice accuracy. The frozen superiority rule requires a mean absolute gain of at least `0.02` and a paired seed-level bootstrap 95% interval excluding zero; any headline superiority claim must also survive the strongest trained non-JEPA baseline rather than only a weak reference.

### 4.3 Pretrained comparator

The pinned comparator path uses `microsoft/deberta-v3-xsmall` at immutable revision `14809e4f1fe1895fcba8b258271a940c6ca45ec4` and learning rate `2e-5`. It is not parameter-matched to LAM-JEPA and is reported only as a bounded strong-comparator characterization.

### 4.4 Compute environment

The retained full-controls workflow specifies a GitHub-hosted `ubuntu-latest` runner, Python 3.11, CPU-only PyTorch from the PyTorch CPU wheel index, a 60-minute job timeout, and explicit `--device cpu` execution. The repository did **not** retain the exact underlying CPU model for the scientific runs, so this manuscript does not invent or infer a processor SKU. That missing hardware granularity remains a reproducibility limitation.

## 5. Results

### 5.1 Capacity-matched result

| System | Validation accuracy |
|---|---:|
| LAM-JEPA | 0.2549152542 ± 0.0129968064 |
| Matched supervised | 0.2664406780 ± 0.0154600058 |
| Paired LAM − matched | -0.0115254237 ± 0.0140994131 |

The frozen validation does not support superiority over the gradient-active-parameter-matched supervised baseline.

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

Neither required mechanism criterion is supported by the frozen evidence. The `no_target` ablation has a higher mean accuracy than the full model in this five-seed result; this adverse observation remains visible rather than being explained away post hoc.

The shuffled-label control is numerically competitive with the full configuration while still remaining below its separately frozen failure ceiling. This is an adverse diagnostic, not evidence for the representation mechanism.

### 5.3 Bounded pretrained comparison

On the recorded bounded development comparison:

- LAM-JEPA: `0.15625`;
- DeBERTa-v3-xsmall: `0.21875`;
- paired difference: `-0.0625`.

This is characterization evidence and is not promoted into a broad final inferiority claim.

### 5.4 Independent reproduction

The frozen full-controls workflow was rerun on independent GitHub-hosted runner attempts without changing the scientific source or protocol. Retained attempts reproduced the aggregate model/ablation/negative-control summaries and strict verifier decision.

Raw per-example probabilities were not byte-identical across independent runners. The retained comparison records low-order numerical drift with a maximum observed numeric difference of approximately `5.9186e-4`, while no aggregate scientific verdict changed. We therefore claim reproducibility of the aggregate conclusion and verifier decision, not byte-exact identity of every floating-point probability or serialized checkpoint.

### 5.5 Reproducibility defect and repair

A separate audit found that an earlier deterministic-training path instantiated the model before applying the requested seed. Before repair, nominally identical one-step runs produced losses `10.853294372558594` and `10.34877872467041`. The narrow repair moved requested seeding before model construction while retaining trainer-side seeding. It changed no ARC split, scientific seed set, threshold, architecture, or locked-test policy. The pre-fix failure remains preserved as evidence.

### 5.6 Repaired quantized-latent validation

The independent recomputation verdict for the repaired ARC-v5 validation is:

`VALID_NEGATIVE_OR_INCONCLUSIVE_VALIDATION`

The repair improved the declared trainability gate but did not establish the preregistered generalization or quantization-benefit claims.

## 6. Ablations and Robustness

The manuscript-ready preregistered mechanism ablations are `no_planner` and `no_target`. The final paper must not add post-hoc ablations and present them as preregistered. Any new architecture repair, benchmark, or mechanism variant requires a new versioned hypothesis and protocol.

The shuffled-label control remains visible even though its outcome is not favorable to an intuitive positive narrative. Its role is diagnostic, not rhetorical. Any additional robustness statistic included in a final paper must trace to a retained raw artifact rather than being reconstructed from prose.

## 7. Failure Analysis

Three primary failure boundaries are supported.

First, **overall performance**: the full model does not exceed the gradient-active-parameter-matched supervised comparator in mean validation accuracy.

Second, **planner mechanism**: removing the planner changes mean accuracy by only about `+0.00475` in favor of the full model, below the frozen `0.01` mechanism threshold, with an interval touching zero.

Third, **target mechanism**: the `no_target` ablation is numerically better than the full model in mean accuracy, so the frozen evidence cannot support an EMA-target benefit on this task.

The repaired quantization study adds a fourth boundary: fixing a trainability problem did not produce the expected validation benefit. This separates engineering recovery from mechanism validation.

The experiment does not isolate one universal cause of failure. Plausible explanations include weak input representation, task/model mismatch, unnecessary composite machinery, limited statistical power, or optimization interactions; these remain hypotheses rather than established causes. In particular, because the frozen token encoder is a small hashing-based mean-pooled embedding model with no Transformer block, the result should not be generalized to modern pretrained language encoders or JEPA methods broadly.

## 8. Limitations

1. The conclusion is specific to the frozen ARC-Challenge development-validation line and tested configurations.
2. Failure to show superiority does not prove the architecture or JEPA family can never be useful on another task.
3. The locked ARC test remains intentionally unused for this failed hypothesis line, so this is not a leaderboard or final test-set result.
4. Five seeds constrain precision; dispersion and bootstrap intervals should be interpreted accordingly.
5. The bounded DeBERTa comparison is not a full final matched comparator study.
6. The input encoder is a small hashed-token mean-pooled representation, not a pretrained language model or Transformer.
7. The latent action indices have no external semantic annotations.
8. The learned retrieval path is not evidence of useful episodic memory or causal retrieval benefit.
9. The repaired quantization path changes trainability but does not validate the original mechanism claim.
10. Independent runners reproduce aggregate conclusions but show low-order floating-point drift in raw probabilities.
11. The exact physical CPU model used by GitHub-hosted runners was not retained.
12. The literature audit is conservative but not exhaustive; a final venue-specific search is required before submission.
13. License, authorship order, citation metadata, and immutable public release metadata require owner approval and remain external release gates.

## 9. Discussion

The most useful result is methodological and empirical rather than architectural: the evidence pipeline prevented an executable composite model from being mislabeled as a successful research result. Gradient-active capacity matching removed one easy confound. Mechanism ablations prevented the full-model score from being attributed automatically to the planner or target path. Independent reruns distinguished reproducibility of the scientific conclusion from byte-exact floating-point determinism. The trainability repair demonstrated why engineering recovery and scientific validation must remain separate.

The originality audit further narrows the contribution. JEPA-style representation prediction, EMA targets, vector quantization, latent actions, and action-conditioned world models are established directions [1–7]. The defensible novelty lies, if anywhere, in the **specific controlled falsification package and its provenance discipline**, not in renaming those ingredients as a new mechanism.

This makes the project a stronger candidate for a falsification-first technical report, reproducibility venue, or negative-results/methodology workshop than for a positive architecture-superiority paper in its current form.

## 10. Conclusion

Under the frozen ARC-Challenge validation protocol, the current LAM-JEPA evidence does not support superiority over a capacity-matched supervised baseline and does not validate the planner or EMA-target mechanisms. A later trainability repair also failed to produce a positive repaired-validation verdict. These negative/inconclusive results are preserved as first-class artifacts, and the confirmatory test remains locked for the failed line. Any next scientific step must be a genuinely new, versioned, preregistered hypothesis rather than post-hoc tuning against the same validation evidence.

## References

[1] Mahmoud Assran et al. *Self-Supervised Learning from Images with a Joint-Embedding Predictive Architecture*. arXiv:2301.08243, 2023.

[2] Adrien Bardes et al. *Revisiting Feature Prediction for Learning Visual Representations from Video*. arXiv:2404.08471, 2024.

[3] Jean-Bastien Grill et al. *Bootstrap Your Own Latent: A New Approach to Self-Supervised Learning*. arXiv:2006.07733, 2020.

[4] Aaron van den Oord, Oriol Vinyals, Koray Kavukcuoglu. *Neural Discrete Representation Learning*. arXiv:1711.00937, 2017.

[5] Seonghyeon Ye et al. *Latent Action Pretraining from Videos*. arXiv:2410.11758, 2024.

[6] Hafez Ghaemi, Eilif Benjamin Muller, Shahab Bakhtiari. *seq-JEPA: Autoregressive Predictive Learning of Invariant-Equivariant World Models*. NeurIPS 2025, OpenReview `GKt3VRaCU1`.

[7] Hai Huang, Yann LeCun, Randall Balestriero. *LLM-JEPA: Large Language Models Meet Joint Embedding Predictive Architectures*. OpenReview `meGygz3CkM`, 2025/2026 venue record.

[8] Peter Clark et al. *Think you have Solved Question Answering? Try ARC, the AI2 Reasoning Challenge*. arXiv:1803.05457, 2018.

[9] Pengcheng He, Jianfeng Gao, Weizhu Chen. *DeBERTaV3: Improving DeBERTa using ELECTRA-Style Pre-Training with Gradient-Disentangled Embedding Sharing*. arXiv:2111.09543, 2021.

**Reference gate:** `ORIGINALITY_AUDIT.md` is the live conservative novelty ledger. Before submission, verify every final bibliographic field against the current primary record and add newly published closer work if found; a closer paper should narrow the framing, not trigger a cosmetic rename.

## Appendix A — Reproducibility package gate

Before submission, the artifact package must pin:

- scientific source commit `760aa7f9a73a177d5ff4ba7eb470f7e68ace63cb`;
- any later software-reproducibility repair commit separately;
- Python/dependency versions and CPU-only environment;
- exact seed list;
- frozen configs and protocol file;
- data hashes, acquisition command, eligibility command, excluded-row evidence, and locked-test non-access assertion;
- capacity-matched baseline command and gradient-active parameter report;
- planner/target ablation and shuffled-label-control commands;
- pretrained-comparator command and its bounded characterization label;
- evaluation/recomputation/verifier commands;
- raw per-seed outputs;
- aggregate tables and bootstrap calculation;
- workflow/run/job/artifact identifiers and artifact digests;
- the fact that exact runner CPU silicon was not retained;
- owner-approved license, authorship, and citation metadata.

Current publication packaging remains incomplete until the exact retained matched-baseline raw artifact is linked into the figure/table manifest, release metadata are owner-approved, and final independent release QA succeeds.

## Appendix B — Claim table

| Claim | Status |
|---|---|
| Reproducible documented pipeline executes | Supported |
| ARC external-benchmark plumbing implemented | Supported |
| Five-seed frozen ARC validation executed | Supported |
| Gradient-active-capacity-matched baseline comparison executed | Supported |
| Aggregate scientific conclusion independently reruns | Supported |
| Exact raw probability/checkpoint bytes reproduce across independent runners | Unsupported |
| Planner improves ARC | Unsupported |
| EMA target mechanism improves ARC | Unsupported |
| LAM-JEPA beats matched supervised baseline | Unsupported |
| Repaired quantization improves validation generalization | Unsupported |
| Latent actions are interpretable reasoning operations | Unsupported |
| Learned retrieval improves ARC causally | Unsupported |
| LAM-JEPA establishes educational effectiveness | Unsupported |
| Result generalizes to pretrained language models or JEPA broadly | Unsupported |
| LAM-JEPA is research-complete or publication-complete | False |
