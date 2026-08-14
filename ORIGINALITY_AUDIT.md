# ORIGINALITY AUDIT — LAM-JEPA

**Audit date:** 2026-08-14  
**Scientific object audited:** the current frozen ARC negative-result line, not the unvalidated full product vision.  
**Audit posture:** conservative. Similarity to prior work is treated as established unless a distinct tested mechanism and evidence show otherwise.

## Bottom line

The current evidence does **not** justify presenting LAM-JEPA as a new general JEPA mechanism, a new latent-action world-model paradigm, a new planning paradigm, a new discrete-representation method, a new EMA-target method, a new anti-collapse objective, a new verifier/search method, or a new grokking method.

Most named architectural ingredients have clear prior art. The strongest defensible research contribution today is instead:

> a falsification-first, reproducible evaluation of a specific small LAM-JEPA configuration on ARC-Challenge, including a gradient-active-parameter-matched supervised comparator, preregistered planner/target ablations, shuffled-label control, a bounded pinned-pretrained comparator, a separately tracked trainability repair, and an explicit stop rule that preserves the adverse result and leaves the confirmatory test locked.

That is primarily a **novel empirical/reproducibility package on a project-specific configuration**, not a demonstrated novel mechanism.

## Closest primary directions

| Related direction | Primary source(s) checked | Similarity to LAM-JEPA | Material difference | Is the difference scientifically meaningful on current evidence? |
|---|---|---|---|---|
| Joint-embedding predictive representation learning | Assran et al., *Self-Supervised Learning from Images with a Joint-Embedding Predictive Architecture*, arXiv:2301.08243 (I-JEPA); Bardes et al., *Revisiting Feature Prediction for Learning Visual Representations from Video*, arXiv:2404.08471 (V-JEPA) | Predict target representations in latent/feature space rather than reconstructing raw observations | LAM-JEPA applies predictive representation ideas to small educational/reasoning classifiers and adds action/planning machinery | **Not yet.** Application/combination differs, but current ARC experiment does not establish a new JEPA learning principle |
| Action-conditioned JEPA world models + planning | Assran et al., *V-JEPA 2: Self-Supervised Video Models Enable Understanding, Prediction and Planning*, arXiv:2506.09985 | Action-conditioned latent predictor/world model used for planning | LAM-JEPA uses discrete reasoning actions and educational/reasoning tasks rather than robot actions/image goals | **Potential application difference only.** Current ARC evidence does not show that latent-action planning adds value |
| Latent-action models learned from observation | Ye et al., *Latent Action Pretraining from Videos*, arXiv:2410.11758; Tharwat et al., *Latent Action Pretraining Through World Modeling*, arXiv:2509.18428; Garrido et al., *Learning Latent Action World Models In The Wild*, arXiv:2601.05230 | Discrete or constrained latent actions are used with learned world models to represent transitions and support downstream control/planning | LAM-JEPA's actions are reasoning/action abstractions inside a supervised small-model system, not unsupervised physical actions inferred from videos | **Distinct semantics, not demonstrated mechanism novelty.** The phrase “latent action model” itself is clearly not novel by 2026 |
| Latent learned dynamics + tree/search planning | Schrittwieser et al., *Mastering Atari, Go, Chess and Shogi by Planning with a Learned Model*, arXiv:1911.08265 (MuZero) | Learned latent dynamics, policy/value prediction and search-based planning | LAM-JEPA frames candidate transitions as educational reasoning actions and includes verification/confidence heads | **No current planning novelty.** The `no_planner` ablation does not satisfy the preregistered contribution gate |
| Explicit search over reasoning trajectories | Yao et al., *Tree of Thoughts*, arXiv:2305.10601 | Explore multiple reasoning trajectories and use evaluation to select paths | LAM-JEPA proposes search over learned latent states rather than textual thoughts | **Possible representation-level difference, but unsupported as an advantage.** Frozen planner effect is near zero/inconclusive |
| Process/step verification for reasoning | Lightman et al., *Let's Verify Step by Step*, arXiv:2305.20050 | Explicit verification of intermediate reasoning can improve reliability | LAM-JEPA adds verifier/confidence heads in a latent planning architecture | **Established design goal.** No current evidence isolates a novel verifier mechanism or educational benefit |
| EMA online/target encoders | Grill et al., *Bootstrap Your Own Latent*, arXiv:2006.07733; I-JEPA also uses a target path | Slow-moving/EMA target representations stabilize self-supervised prediction | LAM-JEPA combines a target path with its supervised/latent-action system | **Not novel.** Current target ablation is adverse (`full - no_target < 0`) under the frozen ARC line |
| Discrete latent quantization | van den Oord et al., *Neural Discrete Representation Learning*, arXiv:1711.00937 (VQ-VAE); LAPA also uses VQ-based latent actions | Vector quantization and discrete codebooks | LAM-JEPA uses quantization as a reasoning bottleneck and later tests a residual/EMA repair | **Not a new quantization mechanism on current evidence.** Trainability repair does not establish a generalization benefit |
| Variance/covariance anti-collapse regularization | Bardes et al., *VICReg*, arXiv:2105.04906 | Variance and covariance terms regularize representation geometry/collapse | LAM-JEPA includes related regularization within a larger composite objective | **Not novel as an ingredient.** No isolated evidence demonstrates a new regularization principle |
| Grokking on algorithmic tasks | Power et al., *Grokking: Generalization Beyond Overfitting on Small Algorithmic Datasets*, arXiv:2201.02177 | Delayed generalization on small algorithmic datasets; long training/regularization studied | LAM-JEPA deliberately includes algorithmic tasks and grokking-oriented schedules | **Not novel without a new mechanistic result.** Current ARC paper does not establish grokking behavior as its central result |

## Novelty boundary by component

### Established techniques

- joint-embedding / feature-space prediction;
- online/target or EMA target networks;
- vector quantization/discrete codebooks;
- latent dynamics/world models;
- action-conditioned prediction;
- tree/beam/search planning;
- value/verification-guided trajectory scoring;
- explicit reasoning verification;
- variance/covariance representation regularization;
- algorithmic-task grokking setups;
- confidence/calibration heads;
- memory retrieval as a general architecture pattern.

### Implementation novelty

Plausible but not sufficient for a scientific novelty claim:

- integrating these modules into one small educational/reasoning research codebase;
- gradient-active-parameter capacity matching infrastructure for the ARC configuration;
- fail-closed protocol/evidence tooling and retained artifact checks;
- clean separation of trainability repair from scientific validation.

### Combination novelty

**Plausible.** The exact combination of JEPA-inspired latent prediction, discrete reasoning actions, latent planning, quantization, verification, memory and educational heads may be uncommon as a single system.

However, “uncommon combination” is not by itself a mechanism contribution. The frozen ARC result currently shows no measurable planner benefit and an adverse/no-benefit target effect, so the combination is not empirically validated as a superior architecture.

### Mechanism novelty

**Not established.** A mechanism claim would require a component with a distinct causal/computational role and an ablation/prediction that survives strong controls. The two current mechanism targets do not pass:

- planner contribution: preregistered criterion unsupported;
- target-path contribution: preregistered criterion unsupported/adverse.

The repaired quantization path solves a bounded trainability problem but does not establish a new generalization mechanism.

### Theoretical novelty

**Not established.** The manuscript's broad latent geometry/Riemannian/topological language should not be presented as a theoretical contribution unless it is fully derived, implemented, tested and connected to a falsifiable prediction. Current ARC evidence does not do that.

### Empirical novelty

**Most plausible contribution.** The specific negative result, controls, matched capacity accounting, repair separation, adverse-control retention and locked-test stop rule form a useful empirical case study.

The novelty claim should be phrased as a project-specific empirical/reproducibility contribution, not as proof of a general law about JEPA, latent planning, educational reasoning or world models.

## Manuscript conflict requiring resolution

The repository currently contains two different paper narratives:

1. `paper.tex` presents a broad architecture-introduction story (“We introduce LAM-JEPA...”) with many components and intended educational capabilities.
2. `MANUSCRIPT_DRAFT_NEGATIVE_ARC.md` presents the evidence-backed frozen ARC falsification result.

These cannot both be treated as the current scientific paper. The second narrative is the defensible one under present evidence. The architecture-heavy manuscript may remain as a historical/design document, but unsupported claims about suitability, reasoning, explanations, adaptive tutoring, grokking or general planning should not leak into the negative-result submission.

## Conservative contribution classification

| Candidate classification | Verdict | Reason |
|---|---|---|
| likely incremental architecture | **YES** | components and high-level composition strongly overlap established directions |
| useful engineering contribution | **YES** | reproducibility, matched-capacity, artifact/protocol and stop-rule tooling are valuable |
| novel empirical observation | **PLAUSIBLE / bounded** | project-specific negative mechanism/superiority result with unusually explicit evidence preservation |
| novel combination | **PLAUSIBLE** | exact educational latent-action/JEPA stack may be unusual, but combination benefit is not supported |
| plausible mechanism contribution | **NO on current evidence** | planner/target contribution gates fail; quantization repair is trainability-only |
| potentially substantial research contribution | **NOT YET** | would require a sharper scientific question and evidence beyond one narrow negative ARC line |

## Recommended paper claim

A defensible title/contribution direction is close to the existing negative draft:

> **LAM-JEPA on ARC-Challenge: A Reproducible Falsification-First Evaluation of Latent Planning and Target-Path Claims**

The paper should emphasize:

- the frozen external-benchmark protocol;
- active-parameter capacity matching;
- mechanism ablations;
- negative control;
- the trainability-repair/scientific-validation distinction;
- adverse-result retention;
- locked confirmatory test after failed development evidence.

It should **not** claim that it introduces latent actions, JEPA planning, EMA targets, vector quantization, verifier-guided reasoning, or grokking as new concepts.

## Literature still worth checking before submission

The following targeted checks remain useful and should be completed without broadening into an unbounded literature survey:

1. negative-result / falsification-first reporting norms in modern ML;
2. preregistration / reusable holdout / adaptive overfitting methodology relevant to keeping ARC test locked;
3. parameter-matching / compute-matching best practices for architecture comparisons;
4. prior small-model ARC-Challenge studies close to the exact comparator regime;
5. whether any prior educational-reasoning model already uses the same specific latent-action + predictive-embedding + planner combination.

If a closer mechanism is found, narrow the novelty language further. Do not search for wording that preserves a desired novelty claim.
