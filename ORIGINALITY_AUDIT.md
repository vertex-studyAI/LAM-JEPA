# ORIGINALITY AUDIT — LAM-JEPA ARC negative-result package

**Audit date:** 2026-08-14  
**Frozen scientific source:** `760aa7f9a73a177d5ff4ba7eb470f7e68ace63cb`  
**Current evidence/audit head at start of closure:** `88f759ef47263c416f2a667427286a3284d8221c`  
**Purpose:** determine what, if anything, is scientifically novel in the *tested ARC configuration* without converting a negative result into a novelty claim.

## Bottom line

The current ARC package should **not** claim a novel JEPA mechanism, a novel vector-quantization mechanism, a novel EMA target network, or a novel general idea of latent planning. The strongest defensible contribution is a **novel empirical/engineering observation and falsification package**: a small project-named LAM-JEPA system with target-encoder alignment, quantization, sparse retrieval, and latent-action rollout was tested against a gradient-active-parameter-matched supervised baseline and frozen mechanism controls on ARC-Challenge; the superiority and planner/target contribution hypotheses were not supported, a trainability repair did not rescue validation, and the confirmatory test remained locked.

The source audit makes the naming boundary stricter than the project name suggests: in the frozen ARC path, the EMA target encoder receives the **same serialized ARC input** as the online encoder; there is no separate masked/future target view in the ARC forward pass. The ARC-specific objective aligns the quantized online representation to that same-input EMA target while also optimizing classification and trajectory/quantization terms. The manuscript should therefore avoid implying that the experiment is a canonical I-JEPA-style context-to-distinct-target prediction task.

**Conservative classification:** `novel empirical observation / useful engineering and reproducibility contribution`, not `substantial mechanism novelty`.

## Source-verified boundary of the tested ARC model

The frozen source matters more than the project name.

- ARC prompts concatenate question and choices, lower-case/split them, and map each whitespace token through a deterministic BLAKE2b hash into a 256-entry vocabulary, padded/truncated to 96 positions by the ARC adapter.
- The ARC adapter supplies `numeric_x` as zeros for every example. The model pads that vector to the configured numeric width, so the numeric branch provides no example-varying ARC information.
- `TokenEncoder` is token embedding + learned positional vectors + `LayerNorm` + mean pooling; `self.encoder = nn.Identity()`. The tested ARC model therefore must **not** be described as a Transformer encoder merely because the config contains `num_heads`/`num_layers` fields.
- The token and zero-valued numeric branches are fused through an MLP and projected to a 32-dimensional latent.
- The default path uses a 32-code EMA vector quantizer with nearest-code assignment, commitment/codebook MSE, and a straight-through estimator.
- Sparse memory retrieves a top-k weighted combination from learned keys/values and applies a learned gated correction; the configured memory capacity is 64.
- The planner is an 8-action latent policy plus stochastic residual transition model. The frozen ARC benchmark requires `model_steps >= 1`; the retained protocol uses one planner step.
- A dedicated four-choice ARC head is applied to a learned summary of the final latent state.
- The target encoder/projector is initialized from the online path and updated by EMA with momentum `0.996`.
- In the frozen `forward`, `target_z` is computed by passing the **same `tokens` and `numeric_x`** through the EMA target encoder/projector. With `use_target=False`, it becomes `z.detach()`.
- The ARC benchmark does **not** call the repository's generic `total_loss`. Its `_lam_arc_loss` is exactly: supervised four-choice cross-entropy + `0.5 × cosine_alignment(z_q, target_z)` + `0.25 × quantization_loss` + `0.25 × mean trajectory MSE to z_q.detach()`.
- The matched supervised baseline shares the same `MultiViewEncoder`/projector family but removes target, quantizer, memory, and planner machinery and trains a four-choice classifier with cross-entropy.

These are implementation facts tied to the frozen scientific commit, not novelty claims.

## Closest directions

| Related direction | Similarity to frozen LAM-JEPA | Difference in this package | Is the difference scientifically meaningful? |
|---|---|---|---|
| I-JEPA — Assran et al., *Self-Supervised Learning from Images with a Joint-Embedding Predictive Architecture*, arXiv:2301.08243 (2023) | representation-space targets and EMA target encoder are conceptually related | canonical I-JEPA predicts representations of distinct masked target blocks from a context block; the frozen LAM ARC path aligns a quantized online embedding to an EMA embedding of the **same serialized input** while jointly training an answer classifier and trajectory | **This is a material difference and weakens the use of “JEPA” as a mechanism claim.** The paper should call the tested path target-encoder representation alignment / project-named LAM-JEPA, not imply canonical context-to-target JEPA prediction. |
| VQ-VAE — van den Oord, Vinyals, Kavukcuoglu, *Neural Discrete Representation Learning*, arXiv:1711.00937 (2017/2018) | learned discrete codebook and vector-quantized latent representation | LAM-JEPA couples quantization to the ARC classification/alignment/trajectory objective and later studies a trainability repair | **Engineering difference only unless a distinct quantization mechanism/effect is demonstrated.** The negative ARC evidence does not establish one. |
| V-JEPA 2 — Assran et al., *V-JEPA 2: Self-Supervised Video Models Enable Understanding, Prediction and Planning*, arXiv:2506.09985 (2025) | JEPA representations connected to an action-conditioned world model and planning | LAM-JEPA's planner is tiny, internal, and evaluated only through bounded ARC ablations | **Weakens any broad “JEPA + planning is novel” claim.** |
| FF-JEPA — Masip, Swinnen, Hu, Detry, Tuytelaars, *FF-JEPA: Long-Horizon Planning in World Models with Latent Planners*, arXiv:2606.09311 (2026) | explicit latent planner associated with a JEPA/world-model setting | FF-JEPA addresses long-horizon physical planning with hierarchical forward models; LAM-JEPA uses a short discrete latent-action rollout inside an ARC classifier | **Directly prevents claiming the generic idea of a latent planner in a JEPA setting as novel in 2026.** |
| AI2 ARC — Clark et al., *Think you have Solved Question Answering? Try ARC, the AI2 Reasoning Challenge*, arXiv:1803.05457 (2018) | same benchmark family; multiple-choice grade-school science reasoning | this package contributes a frozen evaluation/controls package, not the benchmark | **Benchmark use is not novelty.** The defensible empirical contribution is the controlled negative result on the frozen configuration. |
| DeBERTaV3 — He, Gao, Chen, *DeBERTaV3: Improving DeBERTa using ELECTRA-Style Pre-Training with Gradient-Disentangled Embedding Sharing*, arXiv:2111.09543 (2021; ICLR 2023) | strong pretrained language-model family used as a bounded comparator | LAM-JEPA is trained under a small custom architecture/protocol and the DeBERTa-v3-xsmall comparison is development characterization only | **Comparator, not novelty.** Do not elevate the bounded comparison into a broad inferiority or SOTA claim. |
| Reproducibility programme — Pineau et al., *Improving Reproducibility in Machine Learning Research*, JMLR 22(164), 2021 | artifact retention, reproducibility discipline, explicit evaluation practices | LAM-JEPA operationalizes a claim ledger, frozen stop rule, artifact lineage, repeated replay and adverse-result retention within one project | **Useful engineering/research-practice contribution, but not invention of reproducibility or preregistration.** |

## Novelty boundary

### Established technique

- EMA/teacher target encoders and representation alignment;
- vector quantization and straight-through discrete latent training;
- memory retrieval and learned gating as general mechanisms;
- latent/action-conditioned forward dynamics and planning as general mechanisms;
- supervised multiple-choice classification;
- ablation studies, matched baselines, frozen protocols, and reproducibility practice.

### Implementation novelty

The exact small-system composition — deterministic hashed-token serialization, mean-pooled embedding encoder, same-input EMA target alignment, EMA quantizer, learned sparse memory, 8-action latent rollout, and four-choice ARC head — appears project-specific. That supports **implementation specificity**, not automatically scientific novelty.

### Combination novelty

The exact combination may be uncommon, but the audit does not establish that the combination is novel enough to be a central contribution. More importantly, the frozen results do not show that the combination produces a scientifically valuable advantage.

### Mechanism novelty

**Not established.** The frozen planner and target ablations do not meet their contribution criteria. Since the ARC target path is same-input EMA alignment rather than distinct-target prediction, the paper must also avoid attributing the result to a canonical JEPA predictive mechanism that was not actually instantiated in this benchmark path.

### Theoretical novelty

**None claimed.** No new theorem, bound, or mathematical formulation is required for the current negative-result paper.

### Empirical novelty

**Plausible and the strongest boundary.** The useful result is the tightly scoped falsification: under the frozen ARC-Challenge setup, this specific small target-alignment/quantized-memory/planner system does not establish superiority over the matched supervised comparator, and the tested planner/target mechanisms are unsupported; a trainability repair did not convert that into a positive validation result.

## Reviewer-risk assessment

| Risk | Severity | Why | Required paper response |
|---|---|---|---|
| “This is a collection of known components with a new name.” | High if paper claims architecture novelty | component categories and latent planning have prior art | frame the paper around the controlled negative result and evidence discipline; avoid generic mechanism-novelty language |
| “Calling the frozen ARC path a canonical JEPA is inaccurate.” | **Critical** | online and EMA target encoders receive the same serialized ARC input; there is no separate masked/future target view | state this explicitly and use conservative `target-encoder representation alignment` wording |
| “Calling this a Transformer is inaccurate.” | **Critical** | frozen `TokenEncoder.encoder` is `nn.Identity()` | describe embedding + positional vector + LayerNorm + mean pooling exactly |
| “The ARC input representation is extremely weak.” | High | whitespace BLAKE2b hashing into 256 IDs plus mean pooling discards rich linguistic structure; numeric input is constant zero | treat this as a central limitation and a plausible contributor to the negative result, not a post-hoc rescue |
| “ARC is classification, not evidence of general world-model planning.” | High | planner is only evaluated through this bounded ARC setup | keep world-model/general planning claims outside the manuscript |
| “The negative result may be specific to small scale or configuration.” | Medium, valid limitation | five seeds and a narrow configuration cannot rule out all variants | state scope explicitly; do not generalize failure to JEPA as a family |
| “Reproducibility is not scientific novelty.” | Medium | correct; reproducibility practices are established | present provenance/replay as evidence quality enabling the empirical negative claim, not as invention |

## Final allowed novelty language

Safe: **“This work contributes a reproducible, falsification-first empirical evaluation of a specific small ARC architecture combining same-input EMA target alignment, vector quantization, sparse memory, and a latent-action rollout, together with a matched supervised baseline, mechanism ablations, a preserved trainability repair, and an explicit stop rule.”**

Unsafe without new evidence: **“We introduce a novel JEPA architecture / novel latent planner / novel quantized predictive mechanism / Transformer reasoning model / new state of the art.”**

## Source register

The literature entries above were checked against corresponding primary paper pages (arXiv or JMLR) on 2026-08-14. Repository-source statements are tied to frozen commit `760aa7f9a73a177d5ff4ba7eb470f7e68ace63cb`, especially `src/lam_jepa/model.py`, `src/lam_jepa/data.py`, `src/lam_jepa/memory.py`, and `src/lam_jepa/benchmarking/arc_challenge.py`.
