# LAM-JEPA — THREE-REVIEWER ATTACK

**Date:** 2026-08-14  
**Target manuscript:** `MANUSCRIPT_DRAFT_NEGATIVE_ARC.md` as the evidence-backed paper line.  
**Rule:** criticisms are not requests to make the result positive. They are tests of whether the current negative result is scientifically interpretable and publishable.

## Reviewer 1 — Scientific / novelty skeptic

### Strongest criticism

The broad LAM-JEPA architecture is mostly a composition of established ideas: joint-embedding feature prediction, EMA targets, vector quantization, latent/action-conditioned dynamics, search/reasoning trajectories, verifier-guided reasoning, variance/covariance regularization, memory and grokking-style algorithmic training. By 2026, even the phrase and research area “latent action world model” has substantial prior art. The current ARC study does not demonstrate a new mechanism; it demonstrates that two project-specific mechanism hypotheses fail under one narrow configuration.

The repository also contains a conflicting architecture-introduction manuscript (`paper.tex`) whose abstract and method imply broad educational reasoning/planning/grokking capabilities that the frozen ARC line does not validate. A reviewer will view this as claim instability unless one paper is explicitly canonicalized.

### Severity

**MAJOR / acceptance-threatening if broad novelty language remains.**

### Evidence required to answer

- exact prior-art mapping in `ORIGINALITY_AUDIT.md`;
- source-derived statement of what is actually new in the evaluated configuration;
- explicit separation between architecture vision and evaluated scientific claim;
- explanation of why the falsification protocol itself is a useful contribution beyond “our model failed.”

### Cheapest decisive work

**No new experiment.** Rewrite/canonicalize the paper around the falsification-first ARC study and remove unsupported “we introduce a new latent-action JEPA planning mechanism” language.

### Acceptance impact

If narrowed properly, the work may fit a negative-results/reproducibility/methodology-oriented venue. If the paper still claims broad architecture novelty, rejection risk is high.

---

## Reviewer 2 — Experimental / statistical skeptic

### Strongest criticism

The full model (`~0.255`) and matched supervised baseline (`~0.266`) are both near the four-choice random-chance reference of `0.25`. The shuffled-label control (`~0.263`) is numerically similar to the full model, even though it passes the preregistered leakage ceiling `<0.35`. This makes the current experiment highly informative as a **failure/trainability diagnosis**, but weak as an architecture-performance study.

Five seeds are enough to preserve the frozen experiment, but not enough for strong distributional/statistical claims. Bootstrap intervals over `n=5` paired seeds should be treated as descriptive and fragile. The paper should avoid implying that a CI endpoint at zero proves a population-level absence of mechanism effect.

The preregistered v3 protocol also names calibration and answer-choice-order robustness. The current negative draft foregrounds accuracy and mechanism deltas; if calibration/robustness outputs were retained, they should be reported or explicitly scoped out with provenance. If they were not completed, the paper must not imply every preregistered secondary analysis is complete.

Finally, the capacity-matched baseline is a fair architectural control but itself weak in absolute performance. The pinned DeBERTa comparator is only a bounded development characterization (`0.15625` vs `0.21875`) rather than a complete five-seed full-validation strong-baseline study.

### Severity

**MAJOR for interpretation; MODERATE for the bounded negative conclusion.** The negative conclusion remains valid for the frozen comparison, but the paper could be dismissed as “two near-chance scratch models” without careful framing.

### Evidence required to answer

- chance-aware reporting for every primary system;
- per-seed table, not mean/SD alone;
- exact sample counts and paired construction;
- retained calibration/choice-order outputs or an explicit statement that they are not part of the manuscript-ready evidence;
- clear distinction between preregistered analyses and later exploratory characterization;
- no significance language from `n=5` bootstrap intervals;
- exact active-parameter accounting and compute/runtime evidence for the matched baseline.

### Cheapest decisive work

1. **First choice: no new training.** Recompute/report chance-relative accuracy, per-seed paired values, calibration metrics and choice-order robustness from retained raw artifacts if those fields already exist.
2. If a reviewer truly requires a stronger comparator, freeze a **new exploratory characterization protocol** for the already pinned DeBERTa revision on the same eligible validation rows. Do not pretend a new run was preregistered under v3, and do not access the locked test.

### Acceptance impact

A transparent near-chance failure paper can be scientifically useful. A paper that uses near-chance numbers to make mechanism-general claims is not credible.

---

## Reviewer 3 — Mechanism / confounding skeptic

### Strongest criticism

The frozen ARC experiment does not evaluate the broad “planner” described in `paper.tex`.

The implemented ARC path calls the LAM backbone with `model_steps = 1`. The actual `LatentActionModel.rollout` performs a one-step latent transition using an action policy; deterministic evaluation uses the highest-probability action. The ARC evaluator does not perform the beam/tree/hybrid search algorithm shown in the conceptual manuscript. Therefore `no_planner` in this experiment is more precisely a **one-step latent-action-rollout ablation**, not evidence about search-based planning in general.

This distinction is critical because the broad paper text describes candidate-set expansion, top-K retention, value/verifier scoring and multi-step search, while the frozen ARC classifier simply takes the backbone's final `latent_summary` after the configured rollout and applies a four-choice head.

A second confound is that many advertised product/education heads (value, verifier, confidence, rubric, decoder functionality) are not themselves the supervised ARC target being evaluated. Their existence in the architecture cannot be used as evidence that the system verifies reasoning, grades rubrics or improves tutoring.

A third issue is failure-regime identifiability: when the full model is near chance, an ablation failing to reduce accuracy does **not** establish that a mechanism is generally useless. It establishes only that the mechanism did not produce the preregistered benefit in this low-performing frozen configuration.

The repaired v5 line further changes the training question: the repair validation protocol explicitly uses a supervised-cross-entropy-only objective for its repair comparison and is a trainability/generalization test, not a clean test of the original full composite mechanism.

### Severity

**CRITICAL for mechanism wording; LOW for the project-specific negative outcome if wording is narrowed.**

### Evidence required to answer

- source-derived architecture graph for the exact ARC path;
- exact definition of `no_planner` and `no_target` in the executed code/config;
- distinction between one-step latent rollout and beam/tree search;
- list of gradient-active vs inactive modules under the ARC objective;
- explicit statement that untrained/unused product heads are outside the ARC evidence;
- separation of v3 full-controls evidence from v5 trainability-repair evidence.

### Cheapest decisive work

**No new experiment is necessary to fix the paper.** Extract the exact module/active-parameter graph from source and rewrite “planner” as “one-step latent-action rollout” wherever the ARC evidence is discussed, unless a specific source path proves actual search was executed.

A future multi-step/search mechanism experiment may be scientifically interesting, but it must be a new preregistered hypothesis and should not be run merely to rescue this manuscript.

### Acceptance impact

If the manuscript claims to falsify “planning” broadly, this criticism is fatal. If it claims that a **specific one-step latent-action rollout and EMA target path failed their preregistered contribution gates in this ARC configuration**, the result is defensible.

---

# Cross-review synthesis

## What survives all three reviewers

The strongest robust paper claim is:

> Under a frozen exactly-four-choice ARC-Challenge validation protocol, this specific small LAM-JEPA ARC configuration did not outperform a gradient-active-parameter-matched supervised comparator; its one-step latent-action-rollout and target-path ablations did not meet preregistered contribution criteria; a separately versioned quantization/trainability repair also failed its validation promotion rule; these adverse outcomes were independently reproduced and the locked test remained unused.

## What does not survive

Do not claim:

- a new general latent-action world-model paradigm;
- a new JEPA mechanism;
- search-based planning benefit or failure in general;
- educational/tutoring effectiveness;
- verifier/rubric/calibration-head benefit merely because those modules exist;
- grokking benefit;
- vector-quantization benefit;
- broad ARC inferiority/superiority from the bounded pretrained comparator;
- broad “JEPA fails on reasoning” conclusions.

## Highest-value revisions, in order

1. canonicalize the negative ARC manuscript as the scientific paper line;
2. replace conceptual architecture prose with exact source-derived ARC-path description;
3. rename the evaluated planner mechanism precisely as one-step latent-action rollout where appropriate;
4. add a chance-aware/per-seed statistical table from retained evidence;
5. reconcile every preregistered secondary metric/robustness item with retained artifacts;
6. add verified related work from `ORIGINALITY_AUDIT.md`;
7. complete claim→artifact→raw→config→commit provenance;
8. only then decide whether any *new* strong-baseline characterization is worth the cost.

## Recommendation

**Continue paper conversion. Do not run a rescue experiment.** The project remains Tier S because the negative result is unusually well preserved and closeable, not because the underlying architecture has been validated.
