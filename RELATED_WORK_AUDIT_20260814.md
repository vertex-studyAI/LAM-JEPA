# LAM-JEPA Related-Work / Originality Audit — 2026-08-14

**Purpose:** close the bibliographic/originality ambiguity around the current negative ARC manuscript without converting established components into novelty claims.

## Bottom line

The current LAM-JEPA architecture combines established research directions. **The defensible novelty of the current paper package is empirical/methodological, not a demonstrated new mechanism.** The frozen ARC evidence does not validate the planner or target path, so the manuscript must not use those components as proof of architectural novelty.

## Verified primary directions

| Source | Established direction | Relevance to LAM-JEPA | Claim boundary |
|---|---|---|---|
| Assran et al., *Self-Supervised Learning from Images with a Joint-Embedding Predictive Architecture*, arXiv:2301.08243 | JEPA-style prediction of target representations from context representations | Establishes joint-embedding predictive representation learning prior to this project | Do not claim latent target prediction / JEPA as novel here |
| Ye et al., *Latent Action Pretraining from Videos*, arXiv:2410.11758 | Discrete latent actions learned with vector quantization for action-model pretraining | Shows discrete latent-action learning is an established direction | Do not claim discrete latent actions or VQ latent-action pretraining as a new general principle |
| Assran et al., *V-JEPA 2: Self-Supervised Video Models Enable Understanding, Prediction and Planning*, arXiv:2506.09985 | JEPA video representations plus action-conditioned latent world-model planning | Strong prior overlap with “JEPA + latent dynamics + planning” | Educational domain and ARC evaluation are not enough to claim mechanism novelty |
| Garrido et al., *Learning Latent Action World Models In The Wild*, arXiv:2601.05230 | Latent-action world models, action representation choices, planning | Makes latent-action world modeling a mature active direction by 2026 | The burden is on exact empirical mechanism evidence, which current ARC ablations do not provide |
| Clark et al., *Think you have Solved Question Answering? Try ARC, the AI2 Reasoning Challenge*, arXiv:1803.05457 | ARC dataset and reasoning benchmark | Establishes benchmark origin and intended reasoning challenge | Current contribution is evaluation/reproducibility on ARC, not benchmark novelty |

## What the paper may safely claim

The current negative ARC paper can present the following as its contribution package:

1. a frozen ARC-Challenge evaluation pipeline with retained eligibility/exclusion evidence;
2. gradient-active-parameter matching against a supervised comparator;
3. five-seed planner and target-path ablations plus shuffled-label diagnostic control;
4. a pinned pretrained-comparator characterization path;
5. separation of a real seed-order software reproducibility defect from the scientific conclusion;
6. separation of a trainability repair from generalization evidence;
7. an explicit stop rule keeping the confirmatory test locked after validation failure;
8. independent reruns whose aggregate scientific conclusion and strict verifier output reproduce despite low-order floating-point drift in raw probabilities.

## What the paper must not claim

- JEPA itself is novel;
- latent action world models are novel;
- vector-quantized latent actions are novel;
- planning over latent world models is novel;
- the educational application makes the mechanism scientifically novel;
- planner contribution is demonstrated;
- target/EMA contribution is demonstrated;
- quantization benefit is demonstrated;
- the current model is superior on ARC;
- the current result establishes general failure of JEPA or latent-action reasoning systems.

## Recommended related-work framing

The manuscript should distinguish **architecture ancestry** from **the empirical question actually tested**:

> Joint-embedding predictive learning, latent-action representations and latent world-model planning are established directions. We do not claim novelty for these ingredients individually. Instead, we evaluate whether the tested small LAM-JEPA configuration provides measurable benefit on a frozen ARC-Challenge protocol when compared with a capacity-matched supervised model and when its planner and target paths are ablated. The result is negative/inconclusive, and we treat that falsification as the primary empirical finding.

## Conservative originality classification

- **Established technique:** high overlap.
- **Implementation novelty:** moderate, project-specific integration and evidence plumbing.
- **Combination novelty:** plausible but not scientifically sufficient by itself.
- **Mechanism novelty:** unsupported by current evidence.
- **Theoretical novelty:** none established in the frozen package.
- **Empirical novelty:** strongest current angle — reproducible adverse result with matched controls, repair provenance and locked-test stop rule.

## Publication implication

The current package is more defensible as a **falsification-first / reproducibility / negative-result technical report or workshop paper** than as a new-architecture superiority paper. Any future positive mechanism paper must be separately versioned and preregistered rather than editing the current failed hypothesis into a new one.
