# ORIGINALITY AUDIT — LAM-JEPA ARC negative-result package

**Audit date:** 2026-08-14  
**Frozen scientific source:** `760aa7f9a73a177d5ff4ba7eb470f7e68ace63cb`  
**Purpose:** determine what is scientifically defensible about the *tested ARC configuration* without converting a negative result into a novelty claim.

## Bottom line

The current ARC package should **not** claim a novel JEPA mechanism, vector-quantization mechanism, EMA target network, latent-action mechanism, or generic latent planner. The strongest defensible contribution is a **novel empirical observation / useful engineering and reproducibility package**: a specific small system combining same-input EMA target alignment, vector quantization, sparse memory, and a one-step latent-action rollout was evaluated against a gradient-active-parameter-matched supervised comparator and frozen mechanism controls on ARC-Challenge; the superiority and planner/target contribution hypotheses were not supported, a later trainability repair did not rescue validation, and the confirmatory test remained locked.

The source audit makes the naming boundary stricter than the project name suggests. In the frozen ARC path, the EMA target encoder receives the **same serialized ARC input** as the online encoder; there is no separate masked/future target view. The model is also not a Transformer encoder: `TokenEncoder.encoder` is `nn.Identity()` after embedding and learned positional addition, followed by LayerNorm and mean pooling.

**Conservative classification:** `novel empirical observation / useful engineering and reproducibility contribution`, not `substantial mechanism novelty`.

## Source-verified boundary of the tested ARC model

- ARC prompts are lower-cased, whitespace-tokenized, and deterministically hashed with BLAKE2b into a 256-entry vocabulary; the ARC adapter uses maximum length 96.
- `numeric_x` is zero for every ARC example and therefore carries no example-varying signal.
- The token path is embedding + learned positional vectors + LayerNorm + mean pooling, with no contextual Transformer block.
- The default latent path uses a 32-code EMA vector quantizer and learned sparse memory with configured capacity 64.
- The planner is an 8-action latent policy/residual transition model; the frozen ARC protocol uses one model step.
- The target encoder/projector is initialized from the online path and updated by EMA with momentum `0.996`.
- `target_z` uses the same tokens/numeric input as the online encoder; `no_target` substitutes `z.detach()` rather than removing alignment.
- The frozen ARC objective is `CE + 0.5*alignment + 0.25*quantization + 0.25*trajectory`.
- The planner trajectory loss pulls rollout state(s) toward the current `z_q.detach()`, rather than a held-out future state.

These are implementation facts, not novelty claims.

## Closest directions

| Related direction | Similarity | Difference in the frozen ARC package | Is the difference scientifically meaningful? |
|---|---|---|---|
| I-JEPA — Assran et al., arXiv:2301.08243 | representation-space target learning and EMA target encoder | canonical I-JEPA predicts distinct masked target-block representations from context; frozen LAM ARC aligns to an EMA representation of the **same serialized input** while training a classifier | **Material difference.** It weakens any claim that this ARC experiment is a canonical JEPA prediction test. |
| VQ-VAE — van den Oord et al., arXiv:1711.00937 | learned vector-quantized discrete latent representation | LAM combines quantization with ARC classification/alignment/trajectory and later a bounded repair | **Engineering difference only** absent evidence of a distinct useful quantization mechanism. |
| LAPA — Ye et al., arXiv:2410.11758 | learned discrete latent actions using vector-quantized objectives | LAPA learns actions from video; LAM's actions are internal latent choices inside the ARC architecture | **Blocks generic discrete-latent-action novelty.** |
| Learning Latent Action World Models In The Wild — Garrido et al., arXiv:2601.05230 | latent actions/world models and planning without direct action labels | different modality, action representation, and objective | **Blocks broad latent-action/world-model novelty language.** |
| FF-JEPA — Masip et al., arXiv:2606.09311 | explicit latent planner in a JEPA/world-model setting | FF-JEPA targets long-horizon physical planning; LAM uses a short discrete rollout in ARC classification | **Directly blocks generic “latent planner + JEPA is novel” language.** |
| AI2 ARC — Clark et al., arXiv:1803.05457 | same benchmark family | LAM contributes an evaluation package, not the benchmark | **Benchmark use is not novelty.** |
| DeBERTaV3 — He et al., arXiv:2111.09543 | pretrained model family used as bounded comparator | comparison is only development characterization | **Comparator, not novelty.** |
| ML reproducibility programme — Pineau et al., JMLR 2021 | reproducibility/checklist discipline | LAM operationalizes unusually explicit local provenance, reruns, adverse-result retention and a stop rule | **Useful evidence-quality contribution, not invention of reproducibility.** |

## Novelty boundary

### Established technique

- representation targets / EMA target encoders;
- vector quantization and straight-through discrete latent training;
- learned memory retrieval/gating as a general category;
- latent/action-conditioned dynamics and planning as a general category;
- supervised multiple-choice classification;
- ablations, capacity matching, frozen protocols, adverse-result retention and reproducibility practice.

### Implementation novelty

The exact small-system composition — deterministic hashed ARC serialization, mean-pooled embeddings, same-input EMA target alignment, EMA quantizer, learned sparse memory, 8-action rollout and four-choice head — appears project-specific. That supports **implementation specificity**, not automatically scientific novelty.

### Combination novelty

The combination may be uncommon, but this audit does not establish it as a central scientific contribution. More importantly, the frozen evidence does not show that the combination produces a useful advantage.

### Mechanism novelty

**Not established.** Planner and EMA-target contribution gates fail. The ARC target construction also does not instantiate distinct-target predictive learning, so the paper cannot attribute the result to a canonical JEPA mechanism that was not actually tested.

### Theoretical novelty

**None claimed.** No new theorem, bound, or mathematical formulation is established by the current package.

### Empirical novelty

**Plausible and strongest.** Under the frozen ARC-Challenge setup, this specific system does not establish superiority over the matched supervised comparator, its tested planner/EMA-target contribution criteria are unsupported, and a trainability repair does not convert the line into a positive validation result.

## Reviewer-risk assessment

| Risk | Severity | Why | Required response |
|---|---|---|---|
| “Known components with a new name.” | High if architecture novelty is claimed | main component categories have prior art | frame paper around controlled falsification/evidence, not architecture novelty |
| “Calling this a canonical JEPA is inaccurate.” | **Critical** | online/EMA target see the same serialized ARC input | state same-input target construction explicitly |
| “Calling this a Transformer is inaccurate.” | **Critical** | frozen token encoder has `nn.Identity()` contextual encoder | describe exact mean-pooled embedding path |
| “The ARC representation is extremely weak.” | High | 256-way hash collisions + no contextual sequence processing + constant numeric branch | central limitation; plausible explanation, not a rescue claim |
| “Planner ablation is not mechanism proof.” | High | rollout is regularized toward current `z_q`, and answer head consumes transformed final latent | report only failed contribution gate; no world-model mechanism proof |
| “Five seeds are narrow.” | Medium | true | keep conclusions configuration-specific |
| “Reproducibility is not novelty.” | Medium | true | present it as evidence quality enabling a trustworthy negative result |

## Final allowed novelty language

Safe:

> This work contributes a reproducible, falsification-first empirical evaluation of a specific small ARC architecture combining same-input EMA target alignment, vector quantization, sparse memory, and a latent-action rollout, together with a matched supervised comparator, mechanism ablations, a preserved trainability repair, and an explicit stop rule.

Unsafe without a separately frozen study:

- “novel JEPA architecture”;
- “novel latent planner”;
- “novel quantized predictive mechanism”;
- “Transformer reasoning architecture” for the frozen ARC model;
- “JEPA fails on ARC” as a family-level claim;
- “new state of the art.”

## Source register

Primary literature sources were verified on 2026-08-14. Repository-source claims above are tied to frozen scientific commit `760aa7f9a73a177d5ff4ba7eb470f7e68ace63cb`, especially `src/lam_jepa/model.py`, `src/lam_jepa/data.py`, `src/lam_jepa/memory.py`, and `src/lam_jepa/benchmarking/arc_challenge.py`.
