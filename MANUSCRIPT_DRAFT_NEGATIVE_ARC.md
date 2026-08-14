# LAM-JEPA on ARC-Challenge: A Reproducible Falsification-First Evaluation

**Manuscript status:** evidence-backed working draft; not externally validated or submission-ready.  
**Scientific source:** `760aa7f9a73a177d5ff4ba7eb470f7e68ace63cb`.  
**Scientific claim boundary:** the current ARC superiority and planner/target contribution hypotheses are unsupported. The locked ARC confirmatory test remains unopened for this failed line.  
**Provenance companion:** `MANUSCRIPT_PROVENANCE.md`.

## Abstract

We evaluate the project-named LAM-JEPA system on ARC-Challenge using a frozen external-benchmark pipeline, a gradient-active-parameter-matched supervised comparator, mechanism ablations, a shuffled-label control, and a pinned pretrained comparator. Source inspection materially changes how the experiment should be described: the frozen ARC path is a small hashed-token, mean-pooled embedding model with vector quantization, learned sparse memory, a one-step latent-action rollout, and same-input exponential-moving-average (EMA) target alignment; it is not a Transformer encoder and it does not instantiate the canonical I-JEPA context-to-distinct-target prediction task. Across five frozen validation seeds, the full model achieved accuracy `0.2549152542 ± 0.0129968064`, while the matched supervised model achieved `0.2664406780 ± 0.0154600058`, for a paired LAM-minus-matched difference of `-0.0115254237 ± 0.0140994131`. Planner and target-path ablations likewise failed their preregistered contribution criteria: full minus `no_planner` was `+0.0047457627` with a 95% bootstrap interval `[0.0, 0.0142372881]`, while full minus `no_target` was `-0.0067796610` with interval `[-0.0135593220, 0.0]`. A bounded development comparison against pinned DeBERTa-v3-xsmall was adverse (`0.15625` vs `0.21875`), although that comparison is characterization evidence rather than a standalone inferiority claim. A later trainability repair passed its train-only gate but repaired validation remained negative/inconclusive. We preserve these outcomes, keep the confirmatory test locked, and report the experiment as a reproducible falsification case study rather than an architecture-superiority result.

## 1. Introduction

Representation-learning projects are vulnerable to a common failure mode: an interesting mechanism is treated as validated because the code runs, a favorable development slice exists, or a later repair improves trainability. This work asks a narrower question: under a frozen ARC-Challenge validation protocol, does the tested LAM-JEPA configuration outperform an appropriate capacity-matched supervised baseline, and do its planner and target-path mechanisms contribute measurably under the preregistered criteria?

The answer on the current evidence is no. The purpose of this manuscript is therefore not to report a superiority result. It is to document a reproducible negative/inconclusive evaluation with enough protocol detail, controls, raw evidence, source inspection, and claim boundaries to make the falsification useful.

The current contributions are:

1. a reproducible ARC-Challenge evaluation path with retained eligibility and exclusion evidence;
2. a gradient-active-parameter-matched supervised comparison;
3. five-seed planner/target ablations and a deterministic shuffled-label control;
4. a bounded pinned pretrained-comparator path;
5. a documented trainability repair whose later validation did not rescue the original scientific claim;
6. an explicit stop rule that forbids use of the locked ARC confirmatory test to rescue the failed hypothesis;
7. a source-level audit that constrains architecture and novelty wording to what was actually tested.

No claim of ARC superiority, planner benefit, target-path benefit, quantization benefit, Transformer reasoning capability, general JEPA failure, general benchmark superiority, or research completeness is made.

## 2. Related Work and Novelty Boundary

### 2.1 Joint-embedding predictive architectures

I-JEPA predicts representations of distinct target image blocks from a context block and updates a target encoder by exponential moving average [1]. This establishes representation-space target prediction and EMA targets as prior art. The frozen LAM ARC path differs in a scientifically important way: its online and target encoders receive the same serialized ARC input, and the ARC loss aligns the quantized online latent to that same-input EMA target. The present manuscript therefore uses conservative `target-encoder alignment` wording and does not claim to instantiate the canonical I-JEPA context-to-distinct-target task.

### 2.2 Quantized and latent actions

Vector-quantized discrete latent learning predates this project, including VQ-VAE [2]. Latent Action Pretraining (LAPA) uses a VQ-VAE-based objective to learn discrete latent actions from video before downstream action-model training [3]. More recent latent-action world-model work studies constrained continuous latent actions and planning in action-free video settings [4]. These directions make generic claims of novel discrete latent actions or latent-action world models inappropriate here.

Explicit latent planners in JEPA/world-model settings are also public prior art: FF-JEPA introduces an action-free latent planner alongside forward dynamics for long-horizon planning [5]. The current paper therefore does not claim that adding a latent planner to a JEPA-labelled system is a new general mechanism.

### 2.3 ARC and pretrained language models

ARC was introduced as a multiple-choice science-question benchmark designed to stress reasoning beyond earlier QA benchmarks [6]. The bounded pretrained comparison in this project uses a pinned DeBERTa-v3-xsmall checkpoint from the DeBERTaV3 family [7]. We do not compare the locked ARC test to public leaderboard numbers, and the development comparator is not used to claim broad inferiority.

### 2.4 Reproducibility practice

Reproducibility programmes and checklists are established in machine learning [8]. The contribution here is not the invention of reproducibility or preregistration; it is the use of frozen protocols, adverse-result retention, claim ledgers, artifact lineage, and explicit stop rules to make this particular negative result auditable.

### 2.5 Originality classification

`ORIGINALITY_AUDIT.md` classifies the present package conservatively as a **novel empirical observation / useful engineering and reproducibility contribution**, not a substantial mechanism or theoretical contribution. The exact component combination may be project-specific, but the frozen evidence does not show that the combination confers a scientifically useful advantage.

## 3. Problem Formulation and Hypotheses

Let `A_full` denote validation accuracy of the frozen full configuration; `A_match` the gradient-active-parameter-matched supervised comparator; `A_np` the `no_planner` ablation; and `A_nt` the `no_target` ablation.

The frozen scientific line tested three primary claims:

- **H1 — superiority:** the full configuration should outperform the matched supervised comparator under the frozen ARC validation protocol.
- **H2 — planner contribution:** removing the planner should produce a sufficiently adverse paired effect under the preregistered mechanism criterion.
- **H3 — target-path contribution:** replacing the EMA target path with the frozen `no_target` control should produce a sufficiently adverse paired effect under the preregistered mechanism criterion.

The deterministic shuffled-label control was required to remain below a frozen accuracy ceiling of `0.35`. It is a validity control, not evidence for H1–H3.

**Falsification rule:** if the frozen validation fails the superiority/mechanism criteria, the locked ARC confirmatory test is not opened to rescue the line. Architecture repairs or successor mechanisms become new versioned hypotheses.

## 4. Method

### 4.1 ARC serialization

Each ARC example is formatted as a question followed by indexed answer choices. The frozen `text_to_tokens` path lowercases the string, splits on whitespace, hashes each token deterministically with BLAKE2b, maps the hash modulo a vocabulary of 256 IDs, and pads/truncates the ARC sequence to 96 positions.

The ARC adapter supplies a zero-valued `numeric_x` vector for every example. The model pads this vector to its configured numeric width, so the numeric branch provides no example-varying ARC information.

### 4.2 Frozen encoder and latent path

The token encoder is **not a Transformer** in the frozen ARC scientific source. It consists of:

1. a learned token embedding (`vocab_size=256`, `embed_dim=32`);
2. learned positional vectors of length up to 512;
3. `self.encoder = nn.Identity()`;
4. layer normalization;
5. mean pooling over sequence positions.

A linear numeric branch maps the padded zero vector to 32 dimensions. Token and numeric representations are concatenated, fused by an MLP, normalized, and projected to a 32-dimensional latent `z`.

### 4.3 Quantizer and sparse memory

With the frozen default configuration, `z` passes through a 32-code vector quantizer using nearest-code assignment, EMA codebook statistics, commitment/codebook MSE, and a straight-through estimator, producing `z_q`.

A learned sparse memory with configured capacity 64 projects `z_q` to a query, selects top-k learned keys by scaled dot product, forms a softmax-weighted value retrieval, and applies a learned gated correction.

### 4.4 Latent-action rollout and answer head

The planner contains an 8-action policy, learned action embeddings, and separate MLPs for residual transition mean/log-variance. The ARC benchmark requires at least one planner step; the frozen retained protocol uses **one model step**. The final latent is transformed by a learned summary head, and a dedicated four-choice linear answer head produces the ARC logits.

### 4.5 EMA target path

The model maintains a target encoder/projector initialized from the online encoder/projector and updated by EMA with momentum `0.996` after training batches.

Crucially, in the frozen ARC forward pass, `target_z` is obtained by passing the **same `tokens` and `numeric_x`** through the target encoder/projector under `no_grad`. The `no_target` ablation does not delete the alignment term; it replaces the EMA target representation with `z.detach()`.

This is why the manuscript does not characterize the tested ARC mechanism as context-to-future or context-to-masked-target JEPA prediction.

### 4.6 ARC-specific objective

The reported ARC benchmark uses `_lam_arc_loss`, not the repository's larger generic `total_loss`. For an ARC minibatch,

`L = L_CE + 0.5 L_align + 0.25 L_quant + 0.25 L_traj`,

where:

- `L_CE` is four-choice cross-entropy;
- `L_align` is cosine alignment between `z_q` and the detached EMA target `target_z`;
- `L_quant` is the quantizer commitment/codebook loss;
- `L_traj` is mean MSE from planner rollout state(s) to `z_q.detach()`.

This is a hybrid supervised classification/alignment objective, not pure self-supervised JEPA training.

### 4.7 Matched supervised comparator

The supervised comparator uses the same `MultiViewEncoder`/projector family and a four-choice classifier, without the target/quantizer/memory/planner machinery. The frozen gradient-active parameter counts are:

| System | Gradient-active parameters |
|---|---:|
| LAM-JEPA | 86,372 |
| Matched supervised | 86,644 |

The ratio is `1.0031491687`.

## 5. Experimental Setup

### 5.1 Dataset and eligibility

The ARC-Challenge protocol preserves source order and uses a feature-only eligibility rule frozen before confirmatory access. Recorded eligible counts are:

- train: 1,117 / 1,119 rows;
- validation: 295 / 299 rows.

Excluded rows are retained as evidence. The locked ARC test was not downloaded or evaluated for the failed hypothesis line.

### 5.2 Frozen validation budget

The full-controls validation uses:

- seeds `1, 2, 3, 4, 5`;
- 20 epochs;
- batch size 32;
- learning rate `0.0003`;
- model steps 1;
- all 1,117 eligible training rows;
- all 295 eligible validation rows;
- CPU execution in the retained full-controls workflow.

The exact processor model is not asserted; retained evidence establishes GitHub-hosted Ubuntu, Python 3.11 and CPU execution, not a physical CPU SKU.

### 5.3 Pretrained comparator

The pinned comparator path uses `microsoft/deberta-v3-xsmall` at immutable revision `14809e4f1fe1895fcba8b258271a940c6ca45ec4`.

### 5.4 Statistical reporting

Primary five-seed summaries report means and sample standard deviations. Mechanism effects are paired by seed. The retained mechanism analysis uses bootstrap 95% confidence intervals for paired effects. The current package does not claim p-values or statistical significance beyond the frozen criteria.

## 6. Results

### 6.1 Capacity-matched result

| System | Validation accuracy |
|---|---:|
| LAM-JEPA | 0.2549152542 ± 0.0129968064 |
| Matched supervised | 0.2664406780 ± 0.0154600058 |
| Paired LAM − matched | -0.0115254237 ± 0.0140994131 |

The frozen validation does not support superiority over the capacity-matched supervised baseline.

### 6.2 Mechanism ablations

| Configuration | Validation accuracy |
|---|---:|
| Full LAM-JEPA | 0.2549152542 ± 0.0129968064 |
| `no_planner` | 0.2501694915 ± 0.0129968064 |
| `no_target` | 0.2616949153 ± 0.0203954020 |
| Shuffled-label control | 0.2630508475 ± 0.0145011862 |

Paired effects:

- full minus `no_planner`: `+0.0047457627`, 95% bootstrap CI `[0.0, 0.0142372881]`;
- full minus `no_target`: `-0.0067796610`, 95% bootstrap CI `[-0.0135593220, 0.0]`.

Neither required mechanism criterion is supported by the frozen evidence. The shuffled-label control remained below the frozen `0.35` ceiling, but its mean is not presented as favorable evidence for the architecture.

### 6.3 Bounded pretrained comparison

On the recorded bounded development comparison:

- LAM-JEPA: `0.15625`;
- DeBERTa-v3-xsmall: `0.21875`;
- paired difference: `-0.0625`.

This is characterization evidence and is not promoted into a broad inferiority claim.

### 6.4 Repaired validation

The independent recomputation verdict for the repaired ARC-v5 validation is:

`VALID_NEGATIVE_OR_INCONCLUSIVE_VALIDATION`.

The trainability repair passed its bounded train-only gate, but repaired validation did not establish generalization or quantization benefit. The repaired-minus-legacy paired mean was `+0.0040678024` with bootstrap 95% CI `[-0.0135593116, 0.0216949165]`; repaired-minus-no-quantizer was `+0.0033898324` with bootstrap 95% CI `[-0.0027118593, 0.0094915211]`.

## 7. Ablations

The manuscript-ready frozen ablations are `no_planner` and `no_target`.

- `no_planner` removes the latent-action rollout path for the comparison.
- `no_target` replaces the EMA target representation with the detached online `z`; it does **not** remove the alignment term entirely.

This distinction matters when interpreting the mechanism result. The target ablation tests the EMA target path under the frozen implementation, not the broader value of all possible target-prediction mechanisms.

The shuffled-label control remains visible even though its numerical outcome is uncomfortable for an intuitive narrative. Its purpose is diagnostic, not rhetorical.

No post-hoc ablation may be described as preregistered. Any new architecture repair, benchmark, target construction, or mechanism variant is a separate hypothesis with a new validation protocol.

## 8. Robustness and Failure Analysis

### 8.1 Independent reruns

The full frozen controls were independently rerun from scientific source SHA `760aa7f9a73a177d5ff4ba7eb470f7e68ace63cb` in workflow `31203337502`.

Retained successful artifacts include:

- attempt 2: job `94178988063`, artifact `9149336081`, digest `sha256:c45710b5dae6a767ccb6bab7f6e3d8e9578752d8cf9b79fd82a65ae824dded1b`;
- attempt 3: job `94291056903`, artifact `9162165932`, digest `sha256:caa898f1ff046a337db9b5ddbffe1b332943a732868e2fd809abeda8ee89c30b`.

Eight of ten retained files were byte-identical across those attempts. Raw result/input JSON contained low-order floating-point drift, with maximum observed numeric difference about `5.9186e-4`, but aggregate accuracies, paired mechanism effects, verifier reports, and the final scientific conclusion were exact across attempts.

### 8.2 Input/representation limitations

The source audit identifies a plausible contributor to the negative outcome that must remain a limitation rather than a rescue argument:

- ARC text is represented by deterministic whitespace-token hashes into only 256 token IDs;
- the encoder mean-pools embeddings without attention or contextual sequence processing;
- the numeric branch is constant across ARC examples.

These choices make the benchmark a test of this specific small architecture, not a broad test of modern language representation learning or JEPA as a family.

### 8.3 Target-construction limitation

The EMA target receives the same serialized input as the online encoder. Therefore the ARC alignment objective does not directly train prediction of a held-out future/masked target. Failure here cannot be interpreted as falsifying the broader JEPA hypothesis that latent prediction of distinct targets may be useful.

### 8.4 Trainability repair

A later train-only investigation localized a failure in the quantized latent path. The opt-in repair `arc-v5-stable-ema-residual-0.03125` passed its bounded trainability gate and was independently reproduced before repaired validation. It is treated as a new repaired configuration, not retroactive evidence for the original mechanism. Its validation remained negative/inconclusive.

### 8.5 Seed-order reproducibility defect

A pre-fix defect was found: model initialization occurred before the requested seed was applied. PR #61 / SHA `b72a97a99769b278eb8ec75bc5eab62dc9599f29` repaired the reproducibility plumbing without changing the frozen scientific protocol. Failed/pre-fix evidence remains retained.

## 9. Limitations

1. The conclusion is specific to the frozen ARC line and tested small configurations.
2. The frozen token encoder is a mean-pooled hashed-token embedding model, not a Transformer or modern pretrained language encoder.
3. The ARC target path uses same-input EMA alignment rather than distinct context/target prediction.
4. Five seeds quantify only a narrow protocol; the study does not establish that all LAM/JEPA variants fail.
5. The bounded DeBERTa comparison is not a full matched pretrained-comparator study.
6. The repaired quantization path changes trainability but does not validate the original mechanism claim.
7. Exact CPU model metadata is not retained in the paper-facing evidence.
8. Independent expert review is still absent. **[EXTERNAL VALIDATION REQUIRED]**
9. License/authorship/citation metadata remain owner-level release gates. **[EXTERNAL VALIDATION REQUIRED]**
10. The locked confirmatory test cannot ethically be used as a rescue set after the validation hypothesis failed.

## 10. Reproducibility and Provenance

Every quantitative manuscript statement is required to trace through:

`claim → table/figure → processed metric → raw retained artifact → frozen config/protocol → scientific source commit`.

`MANUSCRIPT_PROVENANCE.md` records the current edges and flags any missing raw-artifact pointer instead of treating a processed summary as complete provenance.

The defensible reproducibility statement is: independent reruns reproduce the aggregate scientific conclusion and verifier outputs, while low-level probabilities/checkpoint bytes are not claimed bitwise identical across all independent runners.

**[EXTERNAL VALIDATION REQUIRED]** for a reproduction performed by an independent outside reviewer rather than another project-controlled workflow.

## 11. Broader Impact and Ethics

The immediate scientific risk in this project is evidence inflation: presenting a named architecture, working implementation, or repaired training path as validated intelligence/reasoning progress when the controlled result is negative. The release therefore emphasizes adverse-result retention, narrow claim boundaries, and non-use of the locked confirmatory test after the validation failure.

No educational-effectiveness, AGI, human-equivalence, or production-safety claim follows from these ARC experiments.

## 12. Discussion

The most useful outcome is that the evidence pipeline prevented an executable research prototype from being mislabeled as a successful research result. Capacity matching removed one easy confound. Mechanism ablations prevented the full-model score from being attributed automatically to the planner or EMA target path. The trainability repair demonstrated why engineering recovery and scientific validation must remain separate: a repair can make optimization behave better without producing the expected validation advantage.

The source audit also changes how the project should be described. In the frozen ARC path, “LAM-JEPA” is a project identifier, not proof that a canonical JEPA prediction problem or Transformer reasoning model was evaluated. Making that distinction explicit strengthens the negative report because readers can understand exactly what failed and what remains untested.

This makes the project a stronger candidate for a falsification-first technical report, reproducibility study, or negative-results venue than for an architecture-superiority paper in its current form.

## 13. Conclusion

Under the frozen ARC-Challenge validation protocol, the tested LAM-JEPA configuration did not outperform the capacity-matched supervised baseline and did not validate its planner or EMA-target contribution criteria. A later trainability repair also failed to produce a positive repaired-validation verdict. The result is intentionally narrow: it falsifies the current frozen ARC claims, not JEPA, vector quantization, latent planning, or representation learning in general. The adverse results are retained as first-class artifacts, and the confirmatory test remains locked for this failed line.

## References

1. Mahmoud Assran, Quentin Duval, Ishan Misra, Piotr Bojanowski, Pascal Vincent, Michael Rabbat, Yann LeCun, Nicolas Ballas. **Self-Supervised Learning from Images with a Joint-Embedding Predictive Architecture.** arXiv:2301.08243, 2023.
2. Aaron van den Oord, Oriol Vinyals, Koray Kavukcuoglu. **Neural Discrete Representation Learning.** arXiv:1711.00937, 2017; revised 2018.
3. Seonghyeon Ye, Joel Jang, Byeongguk Jeon, Sejune Joo, Jianwei Yang, Baolin Peng, Ajay Mandlekar, Reuben Tan, Yu-Wei Chao, Bill Yuchen Lin, Lars Liden, Kimin Lee, Jianfeng Gao, Luke Zettlemoyer, Dieter Fox, Minjoon Seo. **Latent Action Pretraining from Videos.** arXiv:2410.11758, 2024.
4. Quentin Garrido, Tushar Nagarajan, Basile Terver, Nicolas Ballas, Yann LeCun, Michael Rabbat. **Learning Latent Action World Models In The Wild.** arXiv:2601.05230, 2026.
5. Sergi Masip, Jonathan Swinnen, Yutong Hu, Renaud Detry, Tinne Tuytelaars. **FF-JEPA: Long-Horizon Planning in World Models with Latent Planners.** arXiv:2606.09311, 2026.
6. Peter Clark, Isaac Cowhey, Oren Etzioni, Tushar Khot, Ashish Sabharwal, Carissa Schoenick, Oyvind Tafjord. **Think you have Solved Question Answering? Try ARC, the AI2 Reasoning Challenge.** arXiv:1803.05457, 2018.
7. Pengcheng He, Jianfeng Gao, Weizhu Chen. **DeBERTaV3: Improving DeBERTa using ELECTRA-Style Pre-Training with Gradient-Disentangled Embedding Sharing.** arXiv:2111.09543, 2021; ICLR 2023.
8. Joelle Pineau, Philippe Vincent-Lamarre, Koustuv Sinha, Vincent Lariviere, Alina Beygelzimer, Florence d'Alche-Buc, Emily Fox, Hugo Larochelle. **Improving Reproducibility in Machine Learning Research (A Report from the NeurIPS 2019 Reproducibility Program).** Journal of Machine Learning Research 22(164):1–20, 2021.

## Appendix A — Reproducibility package gate

Before public submission/release, the package must pin or expose:

- source commit;
- environment/dependency versions;
- exact seed list;
- frozen configs;
- data acquisition and eligibility commands;
- capacity-matched baseline command;
- ablation commands;
- pretrained-comparator command;
- evaluation/recomputation command;
- raw per-seed outputs;
- aggregate tables/bootstrap calculation;
- hardware/runtime metadata to the level actually retained;
- artifact hashes;
- figure/table generation commands;
- license, authorship, and citation metadata approved by the owner.

## Appendix B — Claim table

| Claim | Status |
|---|---|
| Frozen five-seed ARC validation executed | Supported |
| Capacity-matched baseline comparison executed | Supported |
| Independent reruns reproduce aggregate/verifier conclusion | Supported |
| Planner improves ARC | Unsupported |
| EMA target path improves ARC | Unsupported |
| LAM-JEPA beats matched supervised baseline | Unsupported |
| Repaired quantization improves generalization | Unsupported |
| Frozen ARC model is a Transformer | False |
| Frozen ARC alignment predicts a distinct masked/future target | False |
| Locked ARC confirmatory test used to rescue the line | False |
| LAM-JEPA is externally validated | False / pending |
| LAM-JEPA is submission-ready | False / pending packaging and external gates |

## Appendix C — Publication wording guard

Allowed headline:

> Under the frozen ARC-Challenge validation protocol, the tested LAM-JEPA configuration did not outperform the capacity-matched supervised baseline and its planner/EMA-target contribution criteria were not met. Independent reruns reproduce the aggregate negative conclusion and verifier verdict.

Forbidden without a new separately frozen study:

- “novel JEPA planner proven effective”;
- “Transformer reasoning architecture” for this frozen ARC model;
- “JEPA fails on ARC” as a family-level statement;
- “quantization improves generalization”;
- any claim based on opening the locked confirmatory test to rescue this line.
