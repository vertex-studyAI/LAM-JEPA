# ORIGINALITY AUDIT — LAM-JEPA ARC negative-result package

**Audit date:** 2026-08-14  
**Frozen scientific source:** `760aa7f9a73a177d5ff4ba7eb470f7e68ace63cb`  
**Current evidence/audit head at start of closure:** `88f759ef47263c416f2a667427286a3284d8221c`  
**Purpose:** determine what, if anything, is scientifically novel in the *tested* LAM-JEPA configuration without converting a negative result into a novelty claim.

## Bottom line

The current ARC package should **not** claim a novel JEPA mechanism, a novel vector-quantization mechanism, a novel EMA target network, a novel variance/covariance regularizer, or a novel general idea of latent planning. The strongest defensible contribution is a **novel empirical/engineering observation and falsification package**: a small hybrid JEPA-style system with quantization, sparse retrieval, and latent-action rollout was tested against a gradient-active-parameter-matched supervised baseline and frozen mechanism controls on ARC-Challenge; the superiority and planner/target contribution hypotheses were not supported, a trainability repair did not rescue validation, and the confirmatory test remained locked.

**Conservative classification:** `novel empirical observation / useful engineering and reproducibility contribution`, not `substantial mechanism novelty`.

## Source-verified boundary of the tested model

The frozen source matters more than the project name.

- `src/lam_jepa/model.py` at the frozen scientific commit defines a token encoder as token embeddings plus learned positional vectors, `LayerNorm`, and mean pooling; its `self.encoder` is `nn.Identity()`. The tested ARC model therefore must **not** be described as a Transformer encoder merely because the config contains `num_heads`/`num_layers` fields.
- The online representation is fused with a numeric projection through an MLP and projected to a 32-dimensional latent.
- The default path uses a 32-code EMA vector quantizer with nearest-code assignment, commitment/codebook MSE, and straight-through estimation.
- Sparse memory retrieves a top-k weighted combination from learned keys/values and applies a learned gated correction.
- The planner is an 8-action latent policy plus stochastic residual transition model rolled out for up to three steps.
- The target encoder/projector is initialized from the online path and updated by exponential moving average with momentum `0.996`.
- The training objective is **hybrid supervised + representation learning**, not pure self-supervised JEPA: cross-entropy is combined with cosine target alignment, variance/covariance/uniformity/geodesic regularizers, confidence/verifier/rubric losses, trajectory consistency, and quantization loss.

These implementation details are source-verifiable facts, not novelty claims.

## Closest directions

| Related direction | Similarity to frozen LAM-JEPA | Difference in this package | Is the difference scientifically meaningful? |
|---|---|---|---|
| I-JEPA — Assran et al., *Self-Supervised Learning from Images with a Joint-Embedding Predictive Architecture*, arXiv:2301.08243 (2023) | representation-space prediction; learned target encoder updated by EMA; explicit concern with collapse | LAM-JEPA is a small ARC multiple-choice system with supervised CE, quantization, sparse memory, auxiliary heads and latent-action rollout rather than masked image representation prediction | **Not enough for mechanism-novelty language by itself.** The modality/objective/implementation combination is different, but latent target prediction and EMA targets are established. |
| VQ-VAE — van den Oord, Vinyals, Kavukcuoglu, *Neural Discrete Representation Learning*, arXiv:1711.00937 (2017/2018) | learned discrete codebook and vector-quantized latent representation | LAM-JEPA couples quantization to the hybrid ARC objective and later studies a trainability repair | **Engineering difference only unless a distinct quantization mechanism/effect is demonstrated.** The negative ARC evidence does not establish one. |
| VICReg — Bardes, Ponce, LeCun, *VICReg: Variance-Invariance-Covariance Regularization for Self-Supervised Learning*, arXiv:2105.04906 (2021) | explicit variance and covariance regularization of learned representations | LAM-JEPA includes its own weighted variance/covariance terms among several losses | **No novelty claim.** These regularization categories are established prior art. |
| V-JEPA 2 — Assran et al., *V-JEPA 2: Self-Supervised Video Models Enable Understanding, Prediction and Planning*, arXiv:2506.09985 (2025) | JEPA representations connected to an action-conditioned world model and planning | LAM-JEPA's planner is tiny, internal, and evaluated through ARC ablations rather than robotics/world-model planning | **Weakens any broad “JEPA + planning is novel” claim.** Application and implementation differ, but planning with JEPA representations predates this closure package. |
| FF-JEPA — Masip, Swinnen, Hu, Detry, Tuytelaars, *FF-JEPA: Long-Horizon Planning in World Models with Latent Planners*, arXiv:2606.09311 (2026) | explicit latent planner associated with a JEPA/world-model setting | FF-JEPA addresses long-horizon physical planning with hierarchical forward models; LAM-JEPA uses a short discrete latent-action rollout inside an ARC classifier | **Directly prevents claiming the generic idea of a latent planner in a JEPA setting as novel in 2026.** |
| AI2 ARC — Clark et al., *Think you have Solved Question Answering? Try ARC, the AI2 Reasoning Challenge*, arXiv:1803.05457 (2018) | same benchmark family; multiple-choice grade-school science reasoning | this package contributes a frozen evaluation/controls package, not the benchmark | **Benchmark use is not novelty.** The defensible empirical contribution is the controlled negative result on the frozen configuration. |
| DeBERTaV3 — He, Gao, Chen, *DeBERTaV3: Improving DeBERTa using ELECTRA-Style Pre-Training with Gradient-Disentangled Embedding Sharing*, arXiv:2111.09543 (2021; ICLR 2023) | strong pretrained language-model family used as a bounded comparator | LAM-JEPA is trained under a small-model custom architecture/protocol and the DeBERTa-v3-xsmall comparison is development characterization only | **Comparator, not novelty.** Do not elevate the bounded comparison into a broad inferiority or SOTA claim. |
| Reproducibility programme — Pineau et al., *Improving Reproducibility in Machine Learning Research*, JMLR 22(164), 2021 | artifact retention, reproducibility discipline, explicit evaluation practices | LAM-JEPA operationalizes a claim ledger, frozen stop rule, artifact lineage, repeated replay and adverse-result retention within one project | **Useful engineering/research-practice contribution, but not invention of reproducibility or preregistration.** |

## Novelty boundary

### Established technique

- joint-embedding/representation-space prediction;
- EMA target encoders;
- vector quantization and straight-through discrete latent training;
- variance/covariance regularization;
- memory retrieval and learned gating as general mechanisms;
- latent/action-conditioned forward dynamics and planning as general mechanisms;
- cross-entropy classification and auxiliary calibration/verifier losses;
- ablation studies, matched baselines, frozen protocols, and reproducibility practice.

### Implementation novelty

The exact small-system composition — mean-pooled token/numeric encoder, EMA quantizer, learned sparse memory, 8-action stochastic latent rollout, several auxiliary heads, and the frozen ARC adapter — appears project-specific. That supports a claim of **implementation specificity**, not automatically scientific novelty.

### Combination novelty

A combination of these components in this exact ARC-scale package may be uncommon, but the audit does not establish that the combination itself is novel enough to be the paper's central claim. A combination claim would require a broader literature search and evidence that the combination creates a distinct behavior.

### Mechanism novelty

**Not established.** The frozen planner and target ablations do not meet their contribution criteria. The package therefore cannot use the negative ablation results as proof of a novel causal mechanism.

### Theoretical novelty

**None claimed.** No new theorem, bound, or mathematical formulation is required for the current negative-result paper.

### Empirical novelty

**Plausible and the strongest boundary.** The useful result is the tightly scoped falsification: under the frozen ARC-Challenge setup, the tested system does not establish superiority over the matched supervised comparator, and the tested planner/target mechanisms are unsupported; a trainability repair did not convert that into a positive validation result.

## Reviewer-risk assessment

| Risk | Severity | Why | Required paper response |
|---|---|---|---|
| “This is just a collection of known components with a new name.” | High if paper claims architecture novelty | Most categories are established and latent planning in JEPA/world models exists by 2026 | Frame the paper around the controlled negative result and evidence discipline; avoid generic mechanism-novelty language. |
| “Calling this a Transformer/JEPA overstates the frozen implementation.” | High | Frozen token encoder uses `nn.Identity()` after embedding/position addition; objective also includes supervised CE and many auxiliary losses | Describe the exact source graph and call it a hybrid JEPA-style objective. Do not imply a Transformer encoder was tested. |
| “ARC is classification, not evidence of general world-model planning.” | High | Planner is only evaluated through this bounded ARC setup | Keep world-model/general planning claims outside the manuscript. |
| “The negative result may be specific to small scale or configuration.” | Medium, valid limitation | Five seeds and a narrow configuration cannot rule out all variants | State the scope explicitly; do not generalize failure to JEPA as a family. |
| “Reproducibility is not scientific novelty.” | Medium | Correct; reproducibility practices are established | Present provenance/replay as evidence quality enabling the empirical negative claim, not as invention. |

## Final allowed novelty language

Safe: **“This work contributes a reproducible, falsification-first empirical evaluation of a specific hybrid JEPA-style ARC configuration, including a matched supervised baseline, mechanism ablations, a preserved trainability repair, and an explicit stop rule.”**

Unsafe without new evidence: **“We introduce a novel JEPA architecture / novel latent planner / novel quantized predictive mechanism / new state of the art.”**

## Source register

The literature entries above were checked against the corresponding primary paper pages (arXiv or JMLR) on 2026-08-14. The final bibliography must preserve exact titles/authors/identifiers and must not cite this audit as evidence for a scientific result. Repository-source statements are tied to frozen commit `760aa7f9a73a177d5ff4ba7eb470f7e68ace63cb`.
