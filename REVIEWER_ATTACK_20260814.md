# LAM-JEPA REVIEWER ATTACK — 2026-08-14

**Scope:** current negative ARC-Challenge manuscript only.  
**Rule:** reviewers attack the strongest defensible claim. They may narrow or reject the paper; they may not justify post-hoc rescue of the locked ARC hypothesis.

## Reviewer 1 — Scientific / Novelty Skeptic

### Strongest criticism

The architecture is a composite of established ingredients—JEPA-style representation prediction, moving-average targets, vector quantization, learned retrieval, and latent actions/world-model transitions—and the frozen ARC experiment does not show that its planner, target path, or repaired quantizer is beneficial. If the paper is framed as an architecture contribution, the novelty claim is weak and partly contradicted by its own ablations.

### Severity

**HIGH** if submitted as a positive architecture paper.  
**MODERATE** if submitted as a falsification/reproducibility study with conservative novelty language.

### Evidence required to answer

- verified closest-work map;
- exact separation of established technique vs implementation combination vs mechanism claim;
- manuscript title/abstract/introduction that lead with the negative controlled result rather than architectural ambition;
- explicit statement that latent actions are unannotated latent indices, not validated semantic reasoning operations.

### Cheapest decisive action

No new ARC experiment. Keep `ORIGINALITY_AUDIT.md` and the source-locked Related Work section synchronized; run a final primary-source literature refresh immediately before submission.

### Acceptance threat

**YES.** Overclaiming novelty is potentially fatal. The appropriate response is to narrow the contribution, not manufacture a new distinction.

## Reviewer 2 — Experimental Skeptic

### Strongest criticism

The scientific conclusion rests on a five-seed development-validation study with the confirmatory ARC test intentionally unused. The strong pretrained comparison is bounded rather than a full matched confirmatory trial, and the simple hashing-based token representation may make the experiment a test of this particular small model rather than JEPA for language broadly. A reviewer could also question whether five seeds provide enough precision to distinguish small effects.

### Severity

**HIGH** for any broad performance claim.  
**MODERATE** for the narrow negative claim actually made.

### Evidence required to answer

- exact frozen protocol, data hashes, eligibility/exclusion evidence, seed list, and stop rule;
- gradient-active capacity accounting for the matched supervised control;
- paired per-seed results, sample standard deviations, bootstrap intervals, and raw outputs;
- explicit distinction between the matched baseline and bounded DeBERTa characterization;
- locked-test non-access assertion;
- independent aggregate rerun evidence.

### Cheapest decisive action

Finish the provenance/table/figure package from already-retained artifacts. Do **not** use the locked test or add favorable seeds to rescue the result. If a future venue requires a modern pretrained-encoder comparison as the central question, define that as a separately versioned preregistered study rather than silently extending this one.

### Acceptance threat

**YES, but scope-manageable.** The paper must say “this tested configuration failed” rather than “JEPA fails for reasoning/language.”

## Reviewer 3 — Mechanism / Confounding Skeptic

### Strongest criticism

The full model contains quantization, learned retrieval, a stochastic latent transition, an EMA target path, and auxiliary modules. The observed full-model score therefore cannot establish which component helps, and extra machinery/capacity/optimization interactions may hurt. The two required mechanism ablations do not support planner or target benefit; `no_target` is numerically better in mean accuracy. The paper also risks describing the generic model's confidence/verifier/rubric heads even though they are not part of the ARC objective.

### Severity

**HIGH** for mechanism claims.  
**LOW** for the negative statement that the tested planner/target gates failed.

### Evidence required to answer

- source-locked ARC-specific objective, not the repository's broader generic loss;
- gradient-active parameter report explaining why nominal inactive auxiliary parameters are not used for matching;
- exact semantics of `no_target` and `no_planner`;
- no causal claim for memory or quantization without a frozen supporting experiment;
- clear separation between the v3 scientific run and later v5 trainability repair.

### Cheapest decisive action

No new post-hoc mechanism experiment is necessary to support the present negative paper. Strengthen the source-locked Method and Claim Table. Treat any existing old `no_quant`/`no_memory` results outside the frozen ARC v3 evidence as exploratory unless their protocol provenance proves otherwise.

### Acceptance threat

**YES** if the manuscript tries to attribute success or failure causally to untested components. **NO** if it reports only the predeclared mechanism failures and leaves other causes unresolved.

## Cross-review decision

### Claims that survive all three attacks

1. The frozen five-seed ARC development-validation pipeline executed under the recorded protocol.
2. LAM-JEPA did not outperform the gradient-active-parameter-matched supervised comparator in mean validation accuracy.
3. The `no_planner` and `no_target` comparisons did not meet their predeclared mechanism-benefit criteria.
4. Independent workflow attempts reproduced the aggregate scientific conclusion and verifier outputs.
5. Low-order raw floating-point drift exists and is preserved rather than hidden.
6. A later quantized-latent repair restored a bounded trainability gate but did not establish validation/generalization benefit.

### Claims rejected by the review attack

- novel JEPA primitive;
- novel EMA-target mechanism;
- novel vector-quantization method;
- first JEPA for language/reasoning;
- interpretable latent reasoning actions;
- planner benefit;
- target benefit;
- causal memory benefit;
- quantization generalization benefit;
- educational effectiveness;
- broad JEPA/language failure;
- research completeness.

## Paper gate after review

**Status: PAPER PACKAGE ADVANCED, NOT SUBMISSION-READY.**

The scientific story is coherent as a narrow negative/falsification report. Remaining blockers are publication packaging and provenance rather than scientific rescue:

- final figure/table provenance bundle;
- legacy positive-paper quarantine/supersession warning;
- owner-approved license, authorship/citation metadata, and immutable release choice;
- clean-install/reproduction QA on the release candidate;
- final venue-specific literature refresh and independent paper review.
