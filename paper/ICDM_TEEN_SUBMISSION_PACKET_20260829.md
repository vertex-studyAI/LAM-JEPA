# ICDM 2026 Teen Research Track — Final Submission Packet

Status: **SCIENCE/BUILD GO; FINAL UPLOAD BLOCKED ONLY ON OWNER-CONTROLLED METADATA + FINAL REBUILD**

This packet is a copy/paste + verification surface for the final submission. It must not be used to bypass the scientific or authorship gates in `ICDM_TEEN_SUBMISSION_GATE_20260828.md`.

## Official venue facts (reverified 2026-08-29)

- Venue: IEEE ICDM 2026 Teen Research Track / High School Student Research Symposium
- Deadline: **2026-08-30 AoE**
- Format: IEEE Computer Society Proceedings
- Length: **up to 5 pages total, including figures, tables, references**
- Review: single-blind (authors visible)
- Eligibility: first author must be a currently enrolled high-school student
- Contribution: high-school student must be the primary contributor
- First-author affiliation must visibly include: **High School Student**
- Submission: ICDM 2026 submission system, Teen Research Track
- If accepted: at least one author must present in person; high-school presenters must be accompanied by a guardian
- Primary source: https://icdm2026.neu.edu.cn/CallforTeen_en/

## Canonical submission source

- PR: #111
- Branch: `paper/submission-source-20260827`
- Venue source: `paper/icdm_teen_2026.tex`
- Bibliography: `paper/references.bib`
- Claim/provenance source: `MANUSCRIPT_PROVENANCE.md`, `REPRODUCE.md`
- Source verifier: `scripts/ci/verify_icdm_teen_submission_source.py`

## Frozen scientific boundary

Supported claim only:

> Under the frozen ARC-Challenge validation protocol, the tested LAM-JEPA configuration did not outperform its gradient-active-parameter-matched supervised comparator and did not establish the preregistered planner or EMA-target contribution criteria. The confirmatory ARC test remains locked/unopened.

Do **not** convert this into claims of general JEPA failure, Transformer behavior, ARC superiority, planner benefit, target-path benefit, quantization benefit, or external reproducibility.

Frozen headline evidence:

- LAM-JEPA: `0.2549 ± 0.0130`
- Matched supervised: `0.2664 ± 0.0155`
- Paired LAM−matched: `−0.0115 ± 0.0141`
- Seeds: `{1,2,3,4,5}`
- Eligible train rows: `1117`
- Eligible validation rows: `295`
- Epochs: `20`
- Batch size: `32`
- Learning rate: `3e-4`
- Scientific source SHA: `760aa7f9a73a177d5ff4ba7eb470f7e68ace63cb`
- Confirmatory ARC test: locked/unopened

## Submission-form copy surface

### Title

LAM-JEPA on ARC-Challenge: A Reproducible Falsification-First Evaluation

### Abstract

We evaluate the project-named LAM-JEPA system on ARC-Challenge under a frozen protocol with a gradient-active-parameter-matched supervised comparator, mechanism ablations, a shuffled-label validity control, and a pinned pretrained comparator. Source inspection constrains the architecture claim: the frozen ARC path is a small hashed-token, mean-pooled embedding model with vector quantization, sparse memory, a one-step latent-action rollout, and same-input exponential-moving-average target alignment; it is not a Transformer and does not instantiate canonical I-JEPA context-to-distinct-target prediction. Across five frozen validation seeds, LAM-JEPA achieved 0.2549±0.0130 accuracy versus 0.2664±0.0155 for the matched supervised model, with paired difference −0.0115±0.0141. Planner and target-path ablations failed their preregistered contribution criteria. A later bounded trainability repair did not produce a positive validation verdict. We retain these adverse outcomes, keep the confirmatory ARC test locked, and report a reproducible falsification case study rather than an architecture-superiority result.

### Recommended keywords (if the portal requests them)

Use up to five, in this order unless the portal imposes a different taxonomy:

1. ARC-Challenge
2. reproducibility
3. representation learning
4. falsification
5. negative results

These are descriptive keywords only and must not be interpreted as additional scientific claims.

## OWNER-ONLY fields — do not invent

Fill only with truthful, approved information:

- First author full name: `OWNER REQUIRED`
- Final author order: `OWNER REQUIRED`
- School / organization: `OWNER REQUIRED`
- City / country: `OWNER REQUIRED`
- Author email: `OWNER REQUIRED`
- Contact author/person: `OWNER REQUIRED`
- Contact email: `OWNER REQUIRED`
- Any mentor/faculty coauthors and affiliations: `OWNER REQUIRED`

Before the final build, confirm explicitly that:

- the first author is currently enrolled in high school;
- the first author is the primary contributor;
- `High School Student` remains visible in the first-author affiliation;
- all listed authors approved authorship/order;
- no author or affiliation is fabricated or inferred.

## Final rebuild gate after owner metadata is inserted

Do not upload the current placeholder PDF. After truthful metadata replaces the three placeholders, rerun the exact existing paper-build pipeline and require all of the following on the metadata-bearing head:

- Research claim boundary: SUCCESS
- ARC Download Transport CI: SUCCESS
- ARC Protocol V2 QA: SUCCESS
- Reproducibility CI: SUCCESS
- Paper Build: SUCCESS
- actual `paper/icdm_teen_2026.tex` compiles
- final PDF is ≤5 pages inclusive
- no unresolved citations/references
- all fonts embedded
- no clipped/overlapping/broken text on visual inspection
- final source SHA recorded
- final PDF SHA-256 recorded
- final bibliography SHA-256 recorded
- confirmatory-test state remains locked/unopened
- scientific verdict remains negative/inconclusive

## Pre-upload claim audit

Verify each statement below before submission:

- [ ] No ARC superiority claim
- [ ] No planner-benefit claim
- [ ] No EMA-target-benefit claim
- [ ] No quantization-benefit claim
- [ ] No Transformer claim
- [ ] No general-JEPA-failure claim
- [ ] No claim that the confirmatory ARC test was run
- [ ] No claim of independent external reproduction unless new direct evidence exists
- [ ] Negative/mixed outcomes are retained
- [ ] Trainability repair is clearly separated from the frozen original hypothesis
- [ ] All numerical results trace to retained artifacts/provenance

## Portal sequence — user-only

1. Open the official ICDM 2026 submission system and choose **Teen Research Track**.
2. Enter the final approved author list/order and affiliations exactly as in the certified PDF.
3. Ensure the first-author affiliation visibly includes **High School Student**.
4. Paste the title above exactly unless a final source edit intentionally changes it.
5. Paste the abstract above exactly unless a final source edit intentionally changes it.
6. Enter the recommended keywords if the portal requests keywords.
7. Upload only the **final metadata-bearing, exact-head certified PDF**.
8. Before confirming, compare title, author order, affiliation, abstract, and PDF against the certified source.
9. Submit before **2026-08-30 AoE**.
10. Preserve the assigned paper/submission ID, any password or revision credential, confirmation page/email, timestamp, and exact submitted PDF.

## Post-submit evidence record

Immediately record in the repository or private submission log:

- submission ID
- submitted-at timestamp
- final PR/source-head SHA
- final PDF SHA-256
- final bibliography SHA-256
- exact submitted filename
- receipt/confirmation location
- whether revisions remain possible before deadline

Do not commit private portal passwords or credentials to the public repository.

## Current certified pre-metadata baseline

The previous placeholder build on PR source head `216981125d17262b6fc65874b51009993a6654c5` had all five workflows green and produced a 2-page PDF with embedded fonts, no unresolved references, and no obvious visual layout defect. That artifact is a **pre-metadata baseline only**, not the final upload artifact. The final metadata-bearing rebuild must be certified again.
