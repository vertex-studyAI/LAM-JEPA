# REVIEWER ATTACK — LAM-JEPA ARC negative-result package

**Date:** 2026-08-14  
**Scope:** attack the current manuscript without changing the frozen scientific outcome.  
**Rule:** reviewer criticism may narrow claims, expose limitations, or motivate a separately versioned future study; it may not authorize rescue tuning or opening the locked confirmatory test.

## Reviewer 1 — Scientific / novelty skeptic

### Strongest criticism

The architecture-level novelty is weak and the project name risks overstating what was tested. Representation-space targets and EMA target encoders are established by I-JEPA; vector-quantized latent learning is established by VQ-VAE; latent planning in JEPA/world-model settings is public prior art by 2026. More importantly, the frozen ARC path does not implement canonical context-to-distinct-target JEPA prediction: both online and EMA target encoders receive the same serialized ARC input. Calling the negative result a test of “JEPA” broadly would therefore be scientifically misleading.

### Severity

**CRITICAL if the paper claims a novel JEPA mechanism or family-level JEPA conclusion.**  
**LOW-MEDIUM if the paper is framed as a source-audited negative result for this exact project configuration.**

### Evidence required to answer

- exact source graph at scientific SHA `760aa7f9a73a177d5ff4ba7eb470f7e68ace63cb`;
- explicit distinction between same-input EMA alignment and canonical context/target prediction;
- verified prior-work map;
- removal of Transformer/novel-planner/novel-quantizer language.

### Cheapest decisive response

**No new experiment.** Correct the manuscript and novelty boundary. The current scientific value is the controlled falsification/evidence package, not mechanism novelty.

### Acceptance threat

**High** for an architecture-novelty paper; **manageable** for a negative-results/reproducibility/technical-report framing.

### Resolution in this branch

`ORIGINALITY_AUDIT.md`, `RELATED_WORK_TODO.md`, and `MANUSCRIPT_DRAFT_NEGATIVE_ARC.md` now make this boundary explicit.

---

## Reviewer 2 — Experimental skeptic

### Strongest criticism

The frozen ARC experiment is reproducible but the representation pipeline is unusually weak: deterministic whitespace-token hashing into only 256 IDs, followed by learned embeddings and mean pooling with no contextual sequence model. The matched supervised comparator shares that family, so parameter matching is fair for isolating added LAM machinery, but it does not establish competitiveness with strong language baselines. The bounded DeBERTa comparison is development characterization rather than a matched final experiment. Five seeds are adequate to expose the observed non-superiority under the frozen protocol but do not support broad statements about the architecture family.

The shuffled-label control is also uncomfortable: its mean validation accuracy (`~0.2631`) is numerically above the full model mean (`~0.2549`), although it remains below the frozen `0.35` ceiling. The paper must show this result rather than use the control only as a pass/fail badge.

### Severity

**HIGH** if the manuscript claims ARC competitiveness, reasoning ability, or general architectural inference.  
**MEDIUM** for the narrow negative claim “the full system did not beat its matched supervised control and mechanism gates failed.”

### Evidence required to answer

- exact input serialization and encoder source;
- matched active-parameter counts;
- per-seed results and paired effects;
- negative-control row retained in the main results table;
- explicit statement that the pretrained comparison is bounded development evidence;
- no claim of statistical significance beyond frozen criteria.

### Cheapest decisive experiment

For the **current negative claim**, no additional experiment is required: stronger baselines cannot convert the observed matched-control failure into superiority. If the authors want a broader ARC relevance claim, the cheapest useful *new characterization* would be a separately labeled, frozen simple-text baseline (for example a linear/bag-of-words or other strong simple classifier using the same train/validation split) and a properly scoped pretrained baseline. That would be **new post-freeze characterization**, not a rescue of H1–H3, and must not touch the locked confirmatory test.

### Acceptance threat

**Medium-high.** The paper is credible only if it treats the weak input encoder as a central limitation rather than hiding it.

### Resolution in this branch

The manuscript now describes the input representation exactly, retains the shuffled-label outcome, and narrows all conclusions to the tested configuration.

---

## Reviewer 3 — Mechanism / confounding skeptic

### Strongest criticism

The planner and target-path ablations do not cleanly test broad predictive mechanisms.

1. `no_target` replaces the EMA target with `z.detach()` but leaves the cosine alignment term in the ARC loss. It therefore tests the EMA target path, not “target prediction vs no target loss.”
2. The EMA target sees the same ARC input as the online encoder. There is no held-out future/masked target in this benchmark path.
3. The planner trajectory loss pulls each rollout state toward the current quantized latent `z_q.detach()`. On this ARC setup, that is a latent-consistency objective rather than evidence of learning a predictive future state.
4. The answer head consumes the final latent after the planner, so a planner effect can reflect representational transformation/capacity rather than meaningful planning.
5. The numeric branch is constant across ARC examples, and the text encoder lacks contextual sequence processing.

Thus, even a positive planner/target ablation would not by itself prove a causal “world-model planning” mechanism; the current negative ablations certainly do not.

### Severity

**CRITICAL for mechanism-proof language.**  
**Does not invalidate the frozen negative result** if the paper states precisely what each ablation changes.

### Evidence required to answer

- source-level description of `target_z` construction;
- source-level description of `_lam_arc_loss`;
- explicit semantics of `no_target` and `no_planner`;
- acknowledgement that trajectory consistency is not future-state supervision;
- prohibition on general world-model/planning claims.

### Cheapest decisive experiment

**None within this frozen line.** A scientifically meaningful successor would need a new preregistration that defines genuinely distinct context/target information (future, masked, or otherwise withheld), a contextual encoder appropriate to the task, and ablations that remove or replace the hypothesized mechanism cleanly. That would be a new versioned question, not a repair of the current paper.

### Acceptance threat

**High** if the manuscript says the planner/target mechanisms were mechanistically proven or broadly falsified. **Low** if they are reported as failed contribution gates for exactly these components.

### Resolution in this branch

The manuscript now states that `no_target` is an EMA-target replacement rather than removal of alignment, identifies same-input target construction, and limits the conclusion to the frozen ARC implementation.

---

# Reviewer synthesis

## Acceptance-threatening claims removed/forbidden

- novel JEPA mechanism;
- novel latent planner;
- Transformer encoder in the frozen ARC configuration;
- canonical context-to-future/masked-target JEPA prediction in the ARC path;
- ARC competitiveness/SOTA;
- general JEPA failure;
- planner as proven world-model reasoning;
- quantization benefit;
- external validation or submission readiness before those gates exist.

## Current strongest paper claim after attack

> Under the frozen ARC-Challenge protocol, this specific small architecture combining hashed-token mean-pooled encoding, same-input EMA target alignment, vector quantization, sparse memory, and a one-step latent-action rollout did not outperform its gradient-active-parameter-matched supervised comparator, and the frozen planner/EMA-target contribution criteria were not met. Independent reruns reproduce the aggregate negative conclusion and verifier verdict.

## Decision

**KEEP AS A NEGATIVE-RESULT / REPRODUCIBILITY PAPER CANDIDATE.**  
Do not spend scientific compute rescuing the frozen hypothesis. Remaining closure work is provenance/package QA, owner-level license/authorship decisions, and independent external review/reproduction.
