# ICDM 2026 Teen Research Track — Submission Execution Packet

Status date: 2026-08-28

## Official venue facts re-verified

Official Teen Research Symposium call currently states:
- submission deadline: **2026-08-30 Anywhere on Earth (AoE)**
- paper length: **up to 5 pages total, including figures, tables and references**
- IEEE Computer Society Proceedings manuscript format
- single-blind review
- first author must be a high school student enrolled at submission
- high-school student must be the primary contributor
- first-author affiliation must clearly include **High School Student**
- if accepted, a student author is expected to present in person and must be accompanied by a guardian

Venue call: https://icdm2026.neu.edu.cn/CallforTeen_en/

## Canonical source and scientific boundary

Use `MANUSCRIPT_DRAFT_NEGATIVE_ARC.md` as the only scientific source for the current submission line.

Do **not** import claims from the stale positive architecture narrative in `paper.tex`.

Frozen conclusion:

> Under the frozen ARC-Challenge validation protocol, the current LAM-JEPA configuration does not establish superiority over the capacity-matched supervised comparator, and the planner/target-path ablations do not establish the preregistered mechanism contributions. The locked confirmatory ARC test remains unopened for this failed line.

This is the central result, not a weakness to hide.

## Five-page compression plan

The submission should prioritize technical soundness, clarity and reproducibility over architecture breadth.

### Page 1
- title/authors/affiliation
- compact abstract
- introduction
- contribution bullets
- concise related-work/originality boundary

### Page 2
- ARC serialization and model path actually tested
- objective
- matched supervised comparator
- frozen hypotheses and stop rule

### Page 3
- dataset/eligibility
- frozen training/evaluation protocol
- primary results table
- paired mechanism effects

### Page 4
- pretrained-comparator characterization
- repaired-validation result
- reproducibility/rerun evidence
- failure analysis

### Page 5
- limitations
- conclusion
- references

## Keep

- exact five-seed means/SDs
- paired mechanism effects and bootstrap intervals
- matched active-parameter counts
- ARC eligibility counts
- exact frozen seeds/budget
- explicit same-input EMA-target description
- shuffled-label diagnostic even though it is uncomfortable
- locked-test stop rule
- repaired validation remaining negative/inconclusive
- independent rerun statement

## Cut or aggressively compress

- generic educational-AI motivation
- untested production ambitions
- broad architecture modules outside the frozen ARC path
- geometric/topological machinery from stale `paper.tex`
- long descriptions of memory/planning components beyond what is needed to interpret ablations
- unsupported novelty language
- any claim that the tested path is a canonical I-JEPA objective

## Author-controlled blockers

These cannot be safely inferred by automation:

- [ ] final truthful author list and order
- [ ] first-author primary-contributor confirmation
- [ ] first-author affiliation including `High School Student`
- [ ] approved email/contact metadata
- [ ] contribution statement if used
- [ ] repository license/release decision if the code is linked publicly

## Final scientific audit

Before upload, every sentence must satisfy one of:
1. directly supported by frozen code/protocol/result evidence;
2. established background with a verified citation;
3. explicitly marked limitation/interpretation.

Forbidden upgrades:
- no ARC superiority claim
- no planner benefit claim
- no target-path benefit claim
- no quantization benefit claim
- no general JEPA failure claim
- no confirmatory-test claim
- no `research complete` claim

## Numerical table to preserve

| Configuration | ARC-Challenge validation accuracy |
|---|---:|
| Full LAM-JEPA | 0.2549152542 ± 0.0129968064 |
| Matched supervised | 0.2664406780 ± 0.0154600058 |
| no_planner | 0.2501694915 ± 0.0129968064 |
| no_target | 0.2616949153 ± 0.0203954020 |
| shuffled-label control | 0.2630508475 ± 0.0145011862 |

Paired effects:
- full − matched: `-0.0115254237 ± 0.0140994131`
- full − no_planner: `+0.0047457627`, bootstrap 95% CI `[0.0, 0.0142372881]`
- full − no_target: `-0.0067796610`, bootstrap 95% CI `[-0.0135593220, 0.0]`

Bounded pretrained characterization:
- LAM-JEPA: `0.15625`
- pinned DeBERTa-v3-xsmall: `0.21875`
- difference: `-0.0625`

Repaired validation:
- verdict: `VALID_NEGATIVE_OR_INCONCLUSIVE_VALIDATION`
- repaired − legacy mean: `+0.0040678024`, bootstrap 95% CI `[-0.0135593116, 0.0216949165]`
- repaired − no-quantizer mean: `+0.0033898324`, bootstrap 95% CI `[-0.0027118593, 0.0094915211]`

## Final submission sequence

1. Approve author metadata.
2. Produce IEEE two-column PDF from canonical negative manuscript only.
3. Confirm total page count <= 5 including references.
4. Cross-check every number against retained paper assets/claim ledger.
5. Search final source/PDF for stale positive phrases: `superior`, `outperform`, `novel latent action`, `Transformer`, `future latent`, `grokking`, `research complete`.
6. Confirm `High School Student` appears in first-author affiliation.
7. Upload under Teen Research Track.
8. Re-open uploaded PDF from the submission portal and visually inspect it.
9. Confirm author metadata in portal matches PDF.
10. Preserve submission receipt/paper ID as evidence.

## Deadline policy

Do not start new rescue experiments merely to improve the story before the deadline. The scientifically strongest submission is the bounded negative result with clean evidence and honest falsification-first framing.
