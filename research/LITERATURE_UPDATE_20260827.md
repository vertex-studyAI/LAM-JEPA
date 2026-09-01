# LAM-JEPA literature update — 27 August 2026

## Scope

This note updates the mechanism literature relevant to interpreting the frozen ARC result and designing a separately versioned successor study. It does **not** modify, rescue, or reinterpret the frozen ARC-v5 outcome. The locked confirmatory test remains locked.

## What the frozen result now looks like mechanistically

The current ARC paper already establishes three critical facts: the evaluated encoder is a small hashed-token mean-pooled model rather than a contextual Transformer; the EMA target sees the same serialized input rather than a distinct masked/future target; and the frozen full system failed to beat the gradient-active-parameter-matched supervised comparator or satisfy its planner/target-path contribution gates.

Recent literature makes two additional design lessons especially relevant for a successor study:

1. **EMA target updates are not, by themselves, a sufficient anti-collapse argument.** C-JEPA explicitly studies collapse in I-JEPA-like training and adds variance/covariance regularization (VICReg-style) to stabilize representations. VJ-VCR similarly uses variance-covariance regularization in a video JEPA setting. These works do not prove that LAM-JEPA's frozen failure was caused by collapse, but they make representation-variance diagnostics and anti-collapse controls scientifically mandatory in a successor.
2. **Vector quantization has its own collapse modes.** Recent work on VQ representation collapse distinguishes code/token collapse from continuous embedding collapse and links collapse to initialization, encoder capacity, and encoder/codebook dynamics. Newer work studies fixes such as diversity-preserving initialization/timing and non-stationary-aware codebook training. These are useful hypotheses for a new study, not licenses to retune the frozen ARC line.

## Relevant papers and direct implications

### JEPA / joint-embedding collapse

- **Mo & Tong, “Connecting Joint-Embedding Predictive Architecture with Contrastive Self-supervised Learning,” NeurIPS 2024.** OpenReview: https://openreview.net/forum?id=JvQnJWIj6m
  - Relevant claim: EMA alone is not treated as a complete collapse-prevention mechanism; the method adds variance/invariance/covariance regularization.
  - Successor implication: log per-dimension latent variance, covariance spectrum/effective rank, pairwise representation distances, and collapse thresholds during training; include a predeclared anti-collapse treatment rather than assuming EMA is sufficient.

- **Drozdov, Shwartz-Ziv & LeCun, “Video Representation Learning with Joint-Embedding Predictive Architectures,” arXiv:2412.10925.** https://arxiv.org/abs/2412.10925
  - Relevant claim: VJ-VCR explicitly uses variance and covariance regularization to avoid representation collapse in a predictive joint-embedding setting.
  - Successor implication: a clean ablation can compare predictive loss alone versus predictive loss + variance/covariance regularization under the same encoder, targets, parameter band, and training budget.

- **Gögl & Yau, “Var-JEPA,” arXiv:2603.20111.** https://arxiv.org/abs/2603.20111
  - Relevant idea: a variational formulation can make uncertainty and latent regularization explicit instead of relying only on architectural heuristics.
  - Successor implication: uncertainty-aware latent prediction is a possible later branch if the deterministic successor clears its primary gate; it should not be introduced into the first successor pilot because that would confound the central test.

### Vector-quantization collapse

- **Zhao et al., “Representation Collapsing Problems in Vector Quantization,” arXiv:2411.16550.** https://arxiv.org/abs/2411.16550
  - Relevant claim: VQ systems can exhibit token/codebook collapse and embedding collapse, with triggering conditions including initialization and limited encoder capacity.
  - Successor implication: code utilization is not enough by itself. Record code-frequency entropy, number of active codes, latent variance before quantization, quantization error, nearest-code margins, dead-code persistence, and per-seed utilization trajectories.

- **Zhao et al., “Early Quantization Shrinks Codebook: A Simple Fix for Diversity-Preserving Tokenization,” arXiv:2603.17052.** https://arxiv.org/abs/2603.17052
  - Relevant idea: early training dynamics can permanently reduce usable code diversity.
  - Successor implication: if quantization is retained, preregister a short quantizer warm-start/freeze policy or a no-quantizer control *before* outcome inspection. Do not adopt such a fix post hoc after a failed held-out result.

- **Lu et al., “NSVQ: Mitigating Codebook Collapse by Stabilizing Encoder Drift in Vector Quantization,” arXiv:2606.11363.** https://arxiv.org/abs/2606.11363
  - Relevant idea: encoder drift can destabilize sparsely updated codes; staged encoder/codebook training and code replacement are studied as remedies.
  - Successor implication: monitor encoder-distribution drift relative to codebook movement. Treat any staged-training treatment as a separately frozen factor, not an emergency repair.

## Successor-study design consequences

The cleanest next scientific question is **not** “can the old architecture be tuned until it wins?” It is:

> When a contextual encoder predicts genuinely withheld semantic/temporal targets in representation space, does a JEPA-style auxiliary objective improve generalization over a matched contextual supervised baseline under a frozen, equal-budget protocol?

The first successor should therefore remove three ambiguities at once while keeping the comparison narrow:

- use a contextual encoder appropriate for text rather than hashed-token mean pooling;
- use genuinely distinct context and target information rather than same-input EMA alignment;
- include explicit representation-collapse diagnostics, with quantization optional rather than assumed beneficial.

The primary comparison should still be against the same contextual encoder trained with ordinary supervised learning under matched optimization budget. A quantized variant should be a secondary ablation, not the headline treatment.

## Minimum diagnostics for every successor seed

Record at least: supervised validation accuracy; predictive/JEP loss; latent per-dimension variance; effective rank of the latent covariance; mean pairwise cosine similarity; target/online representation norm distributions; for VQ variants, active-code count, code entropy/perplexity, dead-code count, quantization error, assignment concentration, and code-switch rate; optimization curves; parameter count; token/context budget; wall-clock training time; exact data split hashes; source SHA; raw predictions; and failure reason.

A seed that trains but collapses is a scientific result and must remain in the aggregate. It must not be silently replaced.

## Integrity boundary

None of the cited work changes the status of the frozen ARC line. The current evidence remains a negative/inconclusive result for the tested configuration. These papers justify better diagnostics and a better-posed **new** experiment; they do not justify reopening the locked confirmatory test, architecture-shopping on the frozen validation set, deleting failed seeds, or describing a repaired successor as a validation of the original claim.
