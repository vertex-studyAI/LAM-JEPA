# ORIGINALITY AUDIT — LAM-JEPA ARC Negative Study

**Audit date:** 2026-08-14  
**Scope:** the frozen ARC-Challenge study and the implementation actually used by that study.  
**Policy:** established ingredients are not renamed as novel; venue/publication status is taken from current primary records where available. This is a conservative nearest-neighbor audit, not a claim of exhaustive systematic review.

## Bottom line

The current defensible contribution is **not** a new JEPA primitive, EMA target-network method, vector-quantization method, latent-action category, or first JEPA-for-language result. All of those directions have clear prior art.

The strongest evidence-supported contribution is:

> a frozen, checksum-addressed, multi-seed falsification study of a small JEPA/latent-action composite on the eligible ARC-Challenge development-validation task, with a gradient-active-parameter-matched supervised control, planner/target ablations, shuffled-label negative control, bounded pinned pretrained characterization, independent reruns, preserved reproducibility defects, and an explicit stop rule after the proposed superiority/mechanism claims failed.

**Conservative contribution class:** useful negative empirical observation + reproducibility/methodology package. **No substantial new architecture or theory contribution is established by the frozen ARC study.**

## Closest verified directions

| Related direction | Primary source checked | Similarity | Important difference | Meaning for novelty |
|---|---|---|---|---|
| JEPA / latent feature prediction | Assran et al., *Self-Supervised Learning from Images with a Joint-Embedding Predictive Architecture*, arXiv:2301.08243 (2023) | Predicts target representations in embedding space rather than reconstructing raw observations | I-JEPA is masked self-supervised visual representation learning; the frozen LAM ARC experiment is supervised multiple-choice classification with an auxiliary alignment objective and additional mechanisms | **Latent prediction is established.** LAM may test a JEPA-like ingredient in a different setting; it must not claim to invent JEPA |
| Video feature prediction / V-JEPA | Bardes et al., *Revisiting Feature Prediction for Learning Visual Representations from Video*, arXiv:2404.08471 (2024) | Non-generative feature-space prediction as the training objective | Large-scale video SSL and frozen-backbone transfer rather than small text multiple-choice reasoning | **Context, not novelty.** Reinforces that predictive feature learning is an established family |
| JEPA for language | Huang, LeCun & Balestriero, *LLM-JEPA: Large Language Models Meet Joint Embedding Predictive Architectures*, ICLR 2026 Poster, OpenReview `meGygz3CkM` | JEPA-style objectives applied directly to language-model pretraining/finetuning | Large pretrained LMs and multiple NLP/code datasets rather than a bespoke small ARC classifier | **Direct firstness constraint.** The LAM paper must not claim first JEPA for language/reasoning |
| Action-conditioned sequential JEPA | Ghaemi, Muller & Bakhtiari, *seq-JEPA: Autoregressive Predictive Learning of Invariant-Equivariant World Models*, NeurIPS 2025, OpenReview `GKt3VRaCU1` | Conditions predictive representation learning on action/transformation sequences | Visual transformation actions and invariant/equivariant world-model learning, not answer-choice QA | **Broad JEPA + action-conditioned dynamics novelty is unsafe** |
| EMA online/target networks | Grill et al., *Bootstrap Your Own Latent*, arXiv:2006.07733 (2020) | Online network predicts a slowly moving target-network representation | BYOL is self-supervised vision; LAM uses an EMA target path inside a different composite objective | **EMA targets are established.** Only an empirical contribution of LAM's target path could be claimed, and the frozen ablation does not support one |
| Vector-quantized latent codes | van den Oord, Vinyals & Kavukcuoglu, *Neural Discrete Representation Learning*, arXiv:1711.00937 (2017) | Nearest-code discrete latent representations with vector quantization | VQ-VAE is a generative representation model; LAM embeds a small EMA-updated quantizer inside a composite reasoning system | **Basic VQ is established.** Repaired-v5 does not establish a validation benefit from quantization |
| Latent-action pretraining | Ye et al., *Latent Action Pretraining from Videos*, arXiv:2410.11758 (2024) | Learns discrete latent actions and transfers predictive action abstractions | Video/VLA/robot-control setting rather than supervised ARC QA | **Latent actions are established.** Domain/composition differs, but category novelty is not defensible |
| Latent-action world models | Alles et al., *Latent Action World Models for Control with Unlabeled Trajectories*, arXiv:2512.10016 (2025) | Shared learned latent-action representation plus predictive dynamics | Offline control with labeled/unlabeled trajectories rather than ARC | **Context.** Latent action + learned dynamics is an established active family |
| Disentangled latent-action world models | Zhang et al., *DiLA: Disentangled Latent Action World Models*, arXiv:2605.15725 (2026) | Learns latent actions jointly with predictive world modeling | Video generation/planning with content/structure disentanglement | **Current-prior-art warning.** A 2026 manuscript cannot present latent-action world modeling as a new category |
| Continuous latent-action world models | Ayalew et al., *CLAW: Learning Continuous Latent Action World Models via Adversarial Latent Regularization*, arXiv:2606.04130 (2026) | Joint latent-action representation and world-model learning | Continuous action-free video, adversarial regularization, diffusion generation | **Current-prior-art warning.** Further weakens any broad latent-action mechanism novelty claim |
| Predictive embeddings aligned to latent actions | Luo et al., *Predictive Embedding as Latent Action: Towards VLA Pretraining in the Wild*, ICLR 2026 withdrawn submission, OpenReview `iGwN4eoN6k` | Explicitly connects predictive embeddings to latent-action structure | Large-scale VLA/video pretraining; withdrawn, so it is not used as validation evidence | **Novelty warning only.** Shows contemporaneous conceptual adjacency; do not cite it as established empirical authority |
| ARC benchmark | Clark et al., *Think you have Solved Question Answering? Try ARC, the AI2 Reasoning Challenge*, arXiv:1803.05457 (2018) | Same ARC benchmark family | LAM freezes an exactly-four-choice eligibility rule and leaves the confirmatory test unopened after validation failure | **Evaluation design, not a new benchmark.** The checksum/eligibility/stop discipline is study methodology |
| Strong pretrained comparator family | He, Gao & Chen, *DeBERTaV3: Improving DeBERTa using ELECTRA-Style Pre-Training with Gradient-Disentangled Embedding Sharing*, arXiv:2111.09543 (2021) | Source family of the pinned `microsoft/deberta-v3-xsmall` comparator | In LAM this comparator is only a bounded development characterization, not a matched confirmatory trial | **Baseline relevance only** |
| Controlled architecture evaluation | Melis, Dyer & Blunsom, *On the State of the Art of Evaluation in Neural Language Models*, arXiv:1707.05589 (2017) | Demonstrates that conclusions about newer architectures can reverse under controlled evaluation/strong tuning | Language modeling rather than ARC and not the same capacity-accounting design | **Methodological precedent.** Strong controls are not themselves novel, but they support the paper's falsification-first positioning |
| Inferential reproducibility | Hagmann, Meier & Riezler, *Towards Inferential Reproducibility of Machine Learning Research*, ICLR 2023, OpenReview `li4GQCQWkv` | Treats evaluation noise and repeated training as part of inference rather than nuisance to erase | More general statistical methodology than the LAM five-seed benchmark | **Methodological context, not novelty** |
| Exploratory vs confirmatory empirical ML | Herrmann et al., *Position: Why We Must Rethink Empirical Research in Machine Learning*, ICML 2024, OpenReview `DprrMz24tk` | Emphasizes limits of confirmatory framing and the need for stronger empirical discipline | Field-level metascience position rather than an ARC experiment | **Context for protocol discipline; not a claim that preregistration is new** |

## Contribution boundary

### Established technique

- embedding-space / feature prediction;
- EMA target networks;
- vector quantization and codebooks;
- latent actions and learned dynamics;
- supervised multiple-choice classification;
- ablation studies, shuffled-label controls, bootstrap intervals;
- frozen/preregistered evaluation discipline.

### Implementation novelty

The repository combines a compact token/numeric encoder, projector, quantizer, sparse retrieval, latent-action transition, EMA target path and ARC choice head with verification-oriented tooling. This may be useful engineering, but implementation combination by itself is **not** scientific novelty.

### Combination novelty

The exact small-model combination is unusual in this bounded search, but nearby primary work covers its central ingredients separately and, increasingly, in combinations. Safest classification: **unusual implementation combination; exact novelty not established**.

### Mechanism novelty

**Not supported by the current evidence.** The frozen `no_planner` and `no_target` attacks do not establish positive contribution from those mechanisms, and repaired quantization does not establish the preregistered validation benefit. An unusual failed mechanism is not converted into a scientific mechanism contribution by naming it.

### Theoretical novelty

**None established in the frozen ARC paper.** Broader mathematical or geometric ideas elsewhere in the repository must not be imported as experimentally validated theory unless they are explicitly formulated, tested and traced to evidence in this study.

### Empirical novelty

**Plausible but narrow:** the useful contribution is the specific controlled negative result and its audit trail—frozen protocol, dangerous baselines, mechanism ablations, adverse controls, independent reruns, preserved defects and stop rule. Negative-result methodology itself is established; the contribution is this particular evidence package.

## Claims prohibited by this audit

- “We introduce latent-space prediction.”
- “We introduce EMA target encoders.”
- “We introduce vector-quantized reasoning latents.”
- “We introduce latent actions/world models.”
- “We are the first JEPA for language/reasoning.”
- “Planning improves ARC reasoning.”
- “The EMA target improves ARC.”
- “Quantization improves ARC generalization.”
- “The experiment proves educational effectiveness.”

Several mechanism-benefit versions are not merely unverified; they are inconsistent with the frozen result.

## Recommended positioning

Retain the working title:

> *LAM-JEPA on ARC-Challenge: A Reproducible Falsification-First Evaluation*

Recommended contribution sentence:

> We report a frozen, reproducible negative evaluation of a small JEPA/latent-action composite on ARC-Challenge, showing that it does not beat a gradient-active-parameter-matched supervised control and that its planner and EMA-target paths do not satisfy predeclared mechanism-benefit criteria, while preserving the exact protocol, adverse controls, independent reruns and reproducibility failures needed to audit that conclusion.

## Search limitations and submission gate

This audit checks the closest obvious primary mechanism/evaluation families as of 2026-08-14; it is not an exhaustive systematic review. Before submission, rerun venue-specific searches for JEPA in language/reasoning, small-data JEPA, latent-action models, controlled negative studies on ARC/reasoning benchmarks, and matched-capacity evaluation. A closer prior work finding should **downgrade framing**, not trigger a cosmetic rename.
