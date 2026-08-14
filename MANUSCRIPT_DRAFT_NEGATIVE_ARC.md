# LAM-JEPA on ARC-Challenge: A Reproducible Falsification-First Evaluation of a Latent-Action Rollout and EMA Target Path

**Manuscript status:** evidence-backed working draft; not submission-ready.  
**Scientific claim boundary:** the current ARC superiority and mechanism hypotheses are unsupported. The locked ARC test remains unused for this failed hypothesis line.  
**Canonical paper line:** this negative-result manuscript, not the broad architecture-introduction narrative in `paper.tex`.

## Abstract

We evaluate a small LAM-JEPA configuration on AI2 ARC-Challenge under a frozen development-validation protocol designed to distinguish implementation success from scientific support. The evaluated model combines a token/numeric encoder, a projected latent representation, vector quantization, sparse-memory correction, a one-step latent-action rollout, an exponential-moving-average target path, and a four-choice classifier. We compare it with a gradient-active-parameter-matched supervised model, preregistered `no_planner` and `no_target` ablations, a deterministic shuffled-label control, and a bounded pinned DeBERTa-v3-xsmall characterization. Across five frozen validation seeds, LAM-JEPA achieved accuracy `0.254915 ± 0.012997`, while the matched supervised model achieved `0.266441 ± 0.015460`; the paired LAM-minus-matched mean was `-0.011525 ± 0.014099`. The full-minus-`no_planner` effect was `+0.004746` with retained bootstrap 95% interval `[0.0, 0.014237]`, and full-minus-`no_target` was `-0.006780` with interval `[-0.013559, 0.0]`; neither met the preregistered mechanism criterion. A separately versioned quantized-latent trainability repair passed a bounded train-only gate but remained negative/inconclusive on frozen validation. Independent runner attempts reproduced the aggregate scientific conclusion and strict verifier output while retaining low-order per-example floating-point drift. We therefore report a falsification result for this exact configuration rather than a superiority claim. The study illustrates why matched controls, source-level mechanism definitions, versioned repairs, adverse-result retention, and explicit stop rules matter in small-model representation-learning experiments.

## 1. Introduction

A research architecture can accumulate persuasive-looking evidence without establishing its central scientific claim. Code may execute end to end, a development slice may improve after a repair, or a named component may appear intuitively useful, while strong matched controls and ablations tell a different story.

LAM-JEPA began as a broad design combining joint-embedding predictive ideas, latent transitions, discrete representations, memory, verification and educational heads. The present paper does **not** attempt to validate that full vision. It asks a narrower frozen question:

> On the eligible four-choice ARC-Challenge development-validation split, does the tested LAM-JEPA configuration outperform a gradient-active-parameter-matched supervised comparator, and do its one-step latent-action rollout and EMA target path satisfy preregistered contribution criteria?

The current evidence answers **no**. That result is retained rather than repaired into a positive narrative.

The paper contributes:

1. a checksum-addressed ARC-Challenge train/validation pipeline with a frozen, feature-only eligibility rule and locked confirmatory test;
2. a supervised comparator matched by gradient-active parameter count under the exact ARC objective;
3. five-seed `no_planner` and `no_target` controls plus a deterministic shuffled-label control;
4. a bounded pinned pretrained-comparator path;
5. independent reruns with artifact-level traceability;
6. a documented seed-order reproducibility bug and narrow software repair that does not rewrite the scientific result;
7. a separately versioned quantized-latent trainability repair whose validation still fails the promotion rule;
8. an explicit stop rule preventing use of the locked ARC test as a rescue set.

No claim is made that latent actions, JEPA, planning, EMA targets, vector quantization, verification, memory or grokking are new concepts. No claim is made that JEPA broadly fails on reasoning.

## 2. Related Work and Originality Boundary

The architecture combines several established research directions.

**Joint-embedding prediction.** I-JEPA predicts target-block representations from context representations rather than reconstructing pixels [1]. V-JEPA extends feature prediction to video [2]. These works establish latent/feature prediction as prior art; the current contribution is not the invention of JEPA-style prediction.

**Action-conditioned latent world models.** MuZero showed that learned latent dynamics can support planning without reconstructing the full environment [3]. V-JEPA 2 later coupled a self-supervised video representation with an action-conditioned latent world model used for robot planning [4]. LAPA learns discrete latent actions from video using a VQ-based objective [5], and later latent-action world-model work extends this direction to broader video settings. LAM-JEPA differs in using internal reasoning-action abstractions for a small multiple-choice model, but the general concepts of latent actions and latent planning are established.

**Target networks and discrete representations.** BYOL uses a slowly moving target network [6], while VQ-VAE introduced widely used neural vector-quantized representations [7]. These are architectural ingredients rather than novel LAM-JEPA mechanisms.

**Reasoning search and verification.** Tree of Thoughts explores multiple reasoning trajectories with explicit evaluation/search [8], and process-verification work shows the value of supervising intermediate reasoning [9]. The frozen ARC experiment here does not evaluate the beam/tree search algorithm depicted in the historical conceptual manuscript; it evaluates a one-step latent-action rollout.

**Grokking.** Delayed generalization on small algorithmic datasets is established by prior grokking work [10]. The present ARC result is not a grokking result.

The conservative originality classification is therefore **useful engineering/reproducibility contribution + bounded novel empirical negative result**, with possible combination novelty but **no established mechanism or theoretical novelty**. `ORIGINALITY_AUDIT.md` contains the detailed mapping.

## 3. Problem Formulation and Frozen Hypotheses

The v3 protocol retains exactly-four-choice AI2 ARC-Challenge examples using a feature-only rule frozen before confirmatory access. The scientific source revision for the full controls is `760aa7f9a73a177d5ff4ba7eb470f7e68ace63cb`.

Let `A_full(s)` denote validation accuracy for LAM-JEPA at seed `s`, and `A_base(s)` the matched supervised accuracy.

### H1 — Superiority

LAM-JEPA may be described as outperforming a baseline only if:

- mean absolute gain is at least `0.02`; and
- the paired 95% bootstrap interval for `A_full - A_base` excludes zero.

Any headline superiority claim must hold against the strongest trained non-JEPA baseline used for that claim.

### H2 — One-step latent-action-rollout contribution

The configuration named `no_planner` disables the model's latent-action rollout. A mechanism contribution requires:

- mean paired `A_full - A_no_planner >= 0.01`; and
- the paired 95% bootstrap interval excludes zero.

This is **not** a test of tree search, beam search, or planning in general.

### H3 — EMA target-path contribution

The `no_target` configuration disables the EMA target path used by the alignment term. A target-path contribution requires:

- mean paired `A_full - A_no_target >= 0.01`; and
- the paired 95% bootstrap interval excludes zero.

### H4 — Negative-control validity

A deterministic training-label permutation is used as a leakage/shortcut diagnostic. If shuffled-label validation accuracy exceeds `0.35`, the control fails and the run must be investigated before any confirmatory test use.

The `0.35` ceiling is a failure detector, not a claim that any result below it is scientifically strong.

## 4. Exact Evaluated Model

This section is derived from the frozen implementation rather than the historical architecture diagram.

### 4.1 Input encoder

The ARC path uses a `MultiViewEncoder`:

- token IDs are embedded;
- learned positional vectors are added;
- the current `TokenEncoder` applies normalization and mean pooling; its internal sequence `encoder` is `nn.Identity` in the frozen implementation;
- numeric features are projected through a linear layer;
- token and numeric representations are concatenated, fused by an MLP and normalized.

The fused representation is projected to the planning/latent dimension.

### 4.2 Quantized latent path

When enabled, an EMA-updated vector quantizer assigns the projected latent vector to a codebook entry and uses a straight-through quantized representation `z_q`. The ARC loss includes the retained quantization loss term.

The current paper does **not** claim that this quantizer is novel. A later residual/EMA repair is treated separately in Section 8.

### 4.3 Sparse memory

The frozen full configuration retrieves from `SparseMemory` and applies a gated correction before the latent transition. Memory is present in the full model, but this paper makes no memory-contribution claim because the manuscript-ready preregistered mechanism evidence focuses on `no_planner` and `no_target`.

### 4.4 One-step latent-action rollout

The `LatentActionModel` contains:

- a linear policy over discrete action IDs;
- an action embedding;
- MLPs predicting transition mean and log-variance from the current latent, action embedding, retrieved memory signal and optional uncertainty/context signal;
- a residual latent update followed by layer normalization.

The frozen ARC budget sets `model_steps = 1`. Training samples a one-step rollout; deterministic validation selects the highest-probability action. The ARC evaluator does **not** expand a candidate beam/tree or run the conceptual top-K search procedure shown in `paper.tex`.

Accordingly, the term **one-step latent-action rollout** is preferred in this manuscript. We retain the literal configuration name `no_planner` only because it is part of the frozen protocol/artifact lineage.

### 4.5 EMA target path

A target encoder and target projector are initialized from the online path and updated by exponential moving average. The target representation appears in the alignment objective. `no_target` replaces the target representation with the online projected representation detached from gradient flow.

### 4.6 ARC classifier and objective

The final latent state passes through a `latent_summary_head`, then a dedicated four-choice linear answer head.

For the frozen v3 ARC full-controls line, the LAM objective is:

`cross_entropy(choice_logits, labels)`  
`+ 0.5 × cosine_alignment(z_q, target_z)`  
`+ 0.25 × quantization_loss`  
`+ 0.25 × trajectory_loss`.

With one model step, the trajectory term compares the rolled latent state with the detached quantized representation according to the frozen implementation.

The architecture also contains value, confidence, verifier, rubric, uncertainty and decoder heads. Their existence is **not evidence that they improve ARC reasoning, verification, calibration, grading or tutoring**, and this paper does not claim those capabilities were validated by the ARC experiment.

## 5. Baselines and Capacity Matching

### 5.1 Gradient-active-parameter-matched supervised baseline

The non-JEPA comparator uses the same eligible rows, epochs, batch size, training seeds, optimizer family and evaluation budget. Capacity is matched by counting parameters that receive gradients under the exact ARC training objective, rather than nominal parameters that may be inactive.

| System | Gradient-active parameters |
|---|---:|
| LAM-JEPA | 86,372 |
| Matched supervised | 86,644 |

Parameter-count ratio: `1.0031491687`, inside the frozen `[0.99, 1.01]` band.

### 5.2 Shuffled-label negative control

Training labels are deterministically permuted while validation labels remain untouched. This control detects obvious leakage/shortcut behavior under the predeclared `0.35` ceiling.

### 5.3 Pinned pretrained characterization

A bounded development comparison uses `microsoft/deberta-v3-xsmall` at immutable revision `14809e4f1fe1895fcba8b258271a940c6ca45ec4`. It is **not** parameter matched and is not treated as a full confirmatory strong-baseline trial.

## 6. Experimental Setup

### Dataset

AI2 ARC-Challenge. The protocol retains only examples with exactly four answer choices, preserves source order, records exclusions, and forbids label/model-output-based eligibility changes.

- train: 1,117 eligible of 1,119 source rows;
- validation: 295 eligible of 299 source rows;
- test: intentionally not accessed for this failed hypothesis line.

### Frozen v3 full-controls budget

- seeds: `1, 2, 3, 4, 5`;
- epochs: `20`;
- batch size: `32`;
- learning rate: `0.0003` for LAM-JEPA and matched baseline;
- optimizer: AdamW;
- gradient clipping: retained in the implementation;
- model steps: `1`;
- no confirmatory-test use.

### Environment

Retained evidence records the scientific reruns as GitHub-hosted Ubuntu, Python 3.11, CPU execution. The exact CPU model is not claimed because it is not pinned in the current manuscript evidence. Exact workflow/run/artifact IDs are listed in Section 10.

### Statistics

Means use five training seeds and sample standard deviations. Mechanism effects are paired by seed. Retained bootstrap 95% intervals are reported descriptively. With only five seeds, no population-level significance or equivalence claim is made.

## 7. Results

### 7.1 Primary comparison

| System | Validation accuracy, mean ± sample SD | n |
|---|---:|---:|
| Full LAM-JEPA | `0.254915 ± 0.012997` | 5 |
| Matched supervised | `0.266441 ± 0.015460` | 5 |
| Paired LAM − matched | `-0.011525 ± 0.014099` | 5 |

LAM-JEPA does not satisfy the frozen superiority rule.

### 7.2 Chance-aware interpretation

Because the eligible task has four answer choices, uniform random-choice accuracy is `0.25`. The observed mean excess over this reference is descriptive only:

| System | Mean accuracy | Mean minus 0.25 |
|---|---:|---:|
| Full LAM-JEPA | `0.254915` | `+0.004915` |
| Matched supervised | `0.266441` | `+0.016441` |
| `no_planner` | `0.250169` | `+0.000169` |
| `no_target` | `0.261695` | `+0.011695` |
| Shuffled-label control | `0.263051` | `+0.013051` |

These values reinforce the narrow interpretation: this frozen experiment is primarily a failure/diagnostic case study. The shuffled-label control passes its broad `<0.35` failure detector but is numerically similar to the full model; passing the control does not demonstrate useful learned reasoning.

### 7.3 Mechanism ablations

| Configuration | Validation accuracy, mean ± sample SD |
|---|---:|
| Full | `0.254915 ± 0.012997` |
| `no_planner` | `0.250169 ± 0.012997` |
| `no_target` | `0.261695 ± 0.020395` |
| Shuffled-label | `0.263051 ± 0.014501` |

Paired effects:

- full minus `no_planner`: `+0.004746`, bootstrap 95% interval `[0.0, 0.014237]`;
- full minus `no_target`: `-0.006780`, bootstrap 95% interval `[-0.013559, 0.0]`.

Neither preregistered mechanism rule is met. The result supports only the statement that the tested one-step rollout and target path did not provide the required benefit **in this configuration**.

### 7.4 Bounded pretrained characterization

The retained bounded development comparison is adverse:

- LAM-JEPA: `0.15625`;
- pinned DeBERTa-v3-xsmall: `0.21875`;
- paired difference: `-0.0625`.

Because this is not a full five-seed matched confirmatory comparison, it remains characterization evidence.

## 8. Separately Versioned Quantized-Latent Repair

A later train-only investigation localized a trainability problem to the quantized latent path and introduced the narrow repair `arc-v5-stable-ema-residual-0.03125`.

This repair is not retroactively inserted into the v3 full-controls result. The repaired-v5 validation protocol is separately frozen and asks whether the reproduced trainability repair generalizes to validation without returning to prediction-support collapse. Its training configuration uses supervised cross-entropy only for that repair comparison.

The repaired validation verdict remains:

`VALID_NEGATIVE_OR_INCONCLUSIVE_VALIDATION`

The repair therefore establishes a bounded engineering/trainability recovery, not a quantization-generalization mechanism.

## 9. Reproducibility and Failure Preservation

### Independent full scientific reruns

The frozen v3 full-controls workflow was rerun without changing the scientific source/protocol.

**Attempt 2**
- workflow run `31203337502`, attempt 2;
- job `94178988063`;
- artifact `9149336081`;
- SHA-256 `c45710b5dae6a767ccb6bab7f6e3d8e9578752d8cf9b79fd82a65ae824dded1b`.

**Attempt 3**
- workflow run `31203337502`, attempt 3;
- job `94291056903`;
- artifact `9162165932`;
- SHA-256 `caa898f1ff046a337db9b5ddbffe1b332943a732868e2fd809abeda8ee89c30b`.

Aggregate means/SDs, paired mechanism deltas, bootstrap intervals, negative-control verdict and strict verifier output agree exactly. Low-order per-example floating-point probabilities drift across independent runners; byte-exact raw probability identity is not claimed.

### Seed-order reproducibility defect

A separate training-path defect seeded after model construction. Nominally identical one-step runs produced preserved conflicting losses (`10.853294...` vs `10.348779...`). The narrow software repair moved seeding before model initialization while retaining subsequent trainer seeding.

The repair did not change the ARC dataset, architecture, frozen scientific seed set, threshold, primary metric or locked-test policy. It is therefore classified as an execution reproducibility repair, not evidence rescuing the scientific hypothesis.

## 10. Claim-to-Evidence Provenance

| Claim | Immediate artifact | Raw / protocol anchor | Source anchor |
|---|---|---|---|
| full/matched ARC result | `RESULTS.md`, full-controls artifacts | ARC v3 protocol + retained five-seed result payload | scientific SHA `760aa7f...` |
| one-step rollout ablation fails contribution gate | full-controls artifact / verifier | v3 `no_planner` configuration and paired deltas | `src/lam_jepa/benchmarking/arc_challenge.py`, model source at scientific lineage |
| target path fails contribution gate | full-controls artifact / verifier | v3 `no_target` configuration and paired deltas | target-path implementation in model source |
| negative control does not trigger leakage ceiling | full-controls artifact / verifier | deterministic label-permutation control | frozen v3 protocol |
| independent aggregate reproduction | artifacts `9149336081`, `9162165932` | strict verifier reports and raw payloads | same scientific SHA |
| repaired v5 remains negative/inconclusive | repaired-v5 validation artifact family | `protocols/arc_challenge_v5_repaired_validation.json` | separately versioned repair lineage |

A final submission package should machine-check this matrix and fail closed on any missing link.

## 11. Failure Analysis

Three interacting failures matter.

1. **Absolute task performance is near chance.** This limits the interpretability of mechanism ablations: a component cannot demonstrate much benefit when the overall system has learned little task signal.
2. **One-step latent rollout does not pass its gate.** The frozen experiment gives no support for attributing performance to this transition path.
3. **EMA target path does not pass its gate.** The `no_target` mean is higher than the full-model mean in the retained validation.

The later quantized-latent repair shows a fourth distinction: fixing optimization/trainability can be real engineering progress without validating a scientific generalization hypothesis.

## 12. Limitations

1. The result is specific to one small frozen ARC configuration and the eligible exactly-four-choice validation set.
2. Five seeds support transparent seed-level reporting but not broad significance/equivalence claims.
3. Both scratch systems are near chance, so the study is more diagnostic than performance-competitive.
4. The pinned DeBERTa comparison is bounded characterization rather than a full matched strong-baseline study.
5. The experiment tests a one-step latent-action rollout, not beam/tree search or planning in general.
6. The historical architecture includes heads and educational functions not validated by this ARC objective.
7. The repaired-v5 line asks a separate trainability/generalization question and must not be merged into the v3 mechanism claim.
8. Independent runners show low-order floating-point drift in probability payloads.
9. The locked ARC test is intentionally unused after the development hypothesis failed.
10. Educational effectiveness, tutoring quality, grading reliability and student learning outcomes are not evaluated.
11. A final literature review should still check closest small-model ARC comparisons and negative-result/preregistration methodology before submission.

## 13. Broader Impact and Research Integrity

The practical value of this result is not a claim that a specific architecture failed forever. It is evidence that research infrastructure can prevent engineering activity from being mistaken for scientific validation.

Potential positive impact includes making adverse results reproducible, reducing repeated dead ends, and encouraging explicit separation between repairs and hypothesis tests. The main risk is overgeneralization: readers could incorrectly infer that JEPA, latent dynamics or planning broadly fail on reasoning. The manuscript therefore keeps the claim local to the frozen configuration.

No student-learning or educational-deployment benefit is claimed from this experiment.

## 14. Conclusion

Under the frozen ARC-Challenge development-validation protocol, the tested LAM-JEPA configuration does not outperform its gradient-active-parameter-matched supervised comparator. Its one-step latent-action-rollout and EMA target-path ablations do not meet the preregistered contribution criteria. A separately versioned quantized-latent trainability repair also fails its validation promotion rule. These adverse outcomes were retained through independent reruns, while the locked confirmatory test remained unused.

The appropriate next action for this hypothesis line is publication/archival of the bounded negative result, not post-hoc rescue. A future mechanism study must be separately preregistered.

## References — verified primary sources

1. Mahmoud Assran et al. **Self-Supervised Learning from Images with a Joint-Embedding Predictive Architecture.** arXiv:2301.08243, 2023.
2. Adrien Bardes et al. **Revisiting Feature Prediction for Learning Visual Representations from Video.** arXiv:2404.08471, 2024.
3. Julian Schrittwieser et al. **Mastering Atari, Go, Chess and Shogi by Planning with a Learned Model.** arXiv:1911.08265, 2019.
4. Mido Assran et al. **V-JEPA 2: Self-Supervised Video Models Enable Understanding, Prediction and Planning.** arXiv:2506.09985, 2025.
5. Seonghyeon Ye et al. **Latent Action Pretraining from Videos.** arXiv:2410.11758, 2024.
6. Jean-Bastien Grill et al. **Bootstrap Your Own Latent: A New Approach to Self-Supervised Learning.** arXiv:2006.07733, 2020.
7. Aaron van den Oord, Oriol Vinyals, Koray Kavukcuoglu. **Neural Discrete Representation Learning.** arXiv:1711.00937, 2017.
8. Shunyu Yao et al. **Tree of Thoughts: Deliberate Problem Solving with Large Language Models.** arXiv:2305.10601, 2023.
9. Hunter Lightman et al. **Let's Verify Step by Step.** arXiv:2305.20050, 2023.
10. Alethea Power et al. **Grokking: Generalization Beyond Overfitting on Small Algorithmic Datasets.** arXiv:2201.02177, 2022.

## Appendix A — Release gate

Before submission/release, require:

- final source commit/tag;
- owner-approved license decision;
- owner-approved authorship and `CITATION.cff`;
- exact dependency/environment lock where practical;
- all raw result artifacts and hashes;
- exact reproduction commands;
- machine-checked provenance matrix;
- figure/table regeneration commands;
- independent external reviewer/reproducer package;
- final claim ledger with explicit non-claims.

Use `[EXTERNAL VALIDATION REQUIRED]` for independent outside reproduction until it actually occurs.

## Appendix B — Claim table

| Claim | Status |
|---|---|
| Frozen documented ARC pipeline executes | Supported |
| Five-seed validation executed and independently rerun | Supported |
| Capacity-matched supervised comparison executed | Supported |
| LAM-JEPA beats matched supervised baseline | **Unsupported** |
| One-step latent-action rollout improves ARC | **Unsupported** |
| EMA target path improves ARC | **Unsupported** |
| Repaired quantization improves validation/generalization | **Unsupported** |
| Beam/tree planning was evaluated by the frozen ARC experiment | **False** |
| Educational effectiveness was evaluated | **False** |
| Broad JEPA failure is established | **False** |
| Locked ARC test was used to rescue the failed line | **False** |
| Publication/release is complete | **False** |
