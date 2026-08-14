# REVIEWER ATTACK — LAM-JEPA ARC negative-result package

**Date:** 2026-08-14  
**Scope:** attack the current manuscript without changing the frozen scientific outcome.  
**Rule:** criticism may narrow claims, expose limitations, or motivate a separately versioned future study; it may not authorize rescue tuning or opening the locked confirmatory test.

## Reviewer 1 — Scientific / novelty skeptic

### Strongest criticism

Architecture-level novelty is weak and the project name risks overstating what was tested. Representation targets and EMA target encoders are established by I-JEPA; vector-quantized latent learning is established by VQ-VAE; discrete/latent actions and latent-action world models are established directions; explicit latent planners in JEPA/world-model settings are public prior art by 2026. More importantly, the frozen ARC path does not implement canonical context-to-distinct-target JEPA prediction: online and target encoders receive the same serialized ARC input.

### Severity

**CRITICAL** if the paper claims a novel JEPA mechanism or family-level JEPA conclusion.  
**LOW–MEDIUM** if framed as a source-audited negative result for this exact configuration.

### Evidence required

- exact source graph at scientific SHA `760aa7f9a73a177d5ff4ba7eb470f7e68ace63cb`;
- explicit distinction between same-input EMA alignment and canonical context/target prediction;
- verified prior-work map;
- removal of Transformer/novel-planner/novel-quantizer language.

### Cheapest decisive response

**No new experiment.** Correct the manuscript and novelty boundary.

### Acceptance threat

**High** for an architecture-novelty paper; **manageable** for negative-results/reproducibility/technical-report framing.

### Current resolution

`ORIGINALITY_AUDIT.md`, `RELATED_WORK_TODO.md`, and the manuscript now encode this boundary.

---

## Reviewer 2 — Experimental skeptic

### Strongest criticism

The frozen ARC experiment is reproducible, but its input representation is unusually weak: deterministic whitespace-token hashing into only 256 IDs, followed by learned embeddings and mean pooling without contextual sequence processing. The numeric branch is constant zero across ARC examples. The matched supervised comparator shares the same encoder family, so it is useful for isolating added LAM machinery, but it does not establish competitiveness with strong NLP representations. The bounded DeBERTa comparison is development characterization, not a matched confirmatory study.

The shuffled-label control is also adverse to an intuitive story: its mean validation accuracy (`~0.2631`) is numerically above the full model mean (`~0.2549`), although still below the frozen `0.35` ceiling. The paper must expose this result rather than reduce it to a pass/fail badge.

### Severity

**HIGH** if claiming ARC competitiveness, reasoning ability, or broad architecture inference.  
**MEDIUM** for the narrow negative claim that the full system failed to beat its matched supervised control and mechanism gates.

### Evidence required

- exact input serialization and encoder source;
- matched gradient-active parameter counts;
- per-seed/aggregate results and paired effects;
- shuffled-label result in the main evidence story;
- bounded characterization label on DeBERTa comparison;
- no p-value/significance claim beyond retained frozen criteria.

### Cheapest decisive experiment

For the **current negative claim**, no additional experiment is required: a stronger baseline cannot convert the observed matched-control failure into superiority. If a future paper wants broader ARC relevance, a separately frozen simple-text/pretrained characterization could be useful, but it would be **new post-freeze characterization**, not rescue of H1–H3, and must not touch the locked confirmatory test.

### Acceptance threat

**Medium-high.** Credibility depends on treating the weak input encoder and adverse shuffled-label result as central limitations/evidence.

### Current resolution

The manuscript now describes the input path exactly and retains the shuffled-label outcome.

---

## Reviewer 3 — Mechanism / confounding skeptic

### Strongest criticism

The planner and target-path ablations do not cleanly test broad predictive mechanisms.

1. `no_target` replaces the EMA target with `z.detach()` but leaves the cosine alignment term in the ARC loss. It tests the EMA-target path, not “target prediction versus no target objective.”
2. The EMA target sees the same ARC input as the online encoder. There is no held-out future or masked target in this benchmark path.
3. The planner trajectory loss pulls rollout states toward the current quantized latent `z_q.detach()`. On this setup that is latent consistency, not direct predictive-future supervision.
4. The answer head consumes the final post-rollout latent, so an apparent planner effect could reflect representational transformation/capacity rather than meaningful planning.
5. The input representation itself is heavily bottlenecked by hashing/mean pooling.

Therefore even a positive planner/target ablation would not alone prove a world-model planning mechanism; the current negative ablations certainly do not.

### Severity

**CRITICAL** for mechanism-proof language.  
**Does not invalidate the frozen negative result** if the paper states precisely what each ablation changes.

### Evidence required

- source-level `target_z` construction;
- source-level `_lam_arc_loss`;
- exact semantics of `no_target`/`no_planner`;
- acknowledgement that trajectory consistency is not future-state supervision;
- prohibition on broad world-model/planning claims.

### Cheapest decisive experiment

**None within this frozen line.** A scientifically meaningful successor must be a new preregistered study with genuinely distinct context/target information (future, masked, or otherwise withheld), a contextual encoder appropriate to the task, and cleaner mechanism ablations. That is a new question, not a repair of this paper.

### Acceptance threat

**High** if the manuscript says planner/target mechanisms were mechanistically proven or broadly falsified; **low** if reported as failed contribution gates for exactly these frozen components.

### Current resolution

The manuscript explicitly states that `no_target` is an EMA-target substitution rather than deletion of alignment, identifies same-input target construction, and limits conclusions to the tested ARC implementation.

---

# Reviewer synthesis

## Acceptance-threatening claims removed / forbidden

- novel JEPA mechanism;
- novel latent planner;
- novel generic latent-action mechanism;
- Transformer encoder in the frozen ARC configuration;
- canonical context-to-future/masked-target JEPA prediction in this ARC path;
- ARC competitiveness/SOTA;
- general JEPA failure;
- planner as proven world-model reasoning;
- quantization benefit;
- external validation or submission readiness before those gates exist.

## Strongest current paper claim

> Under the frozen ARC-Challenge protocol, this specific small architecture combining hashed-token mean-pooled encoding, same-input EMA target alignment, vector quantization, sparse memory, and a one-step latent-action rollout did not outperform its gradient-active-parameter-matched supervised comparator, and the frozen planner/EMA-target contribution criteria were not met. Independent reruns reproduce the aggregate negative conclusion and verifier verdict.

## Decision

**KEEP AS A NEGATIVE-RESULT / REPRODUCIBILITY PAPER CANDIDATE.**

Do not spend scientific compute rescuing the frozen hypothesis. Remaining closure work is raw-pointer/figure provenance, owner-level license/authorship decisions, and independent external review/reproduction.
