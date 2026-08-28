# ICDM 2026 Teen Research Track — Finalization Gate (2026-08-28)

## Target

Paper: **LAM-JEPA on ARC-Challenge: A Reproducible Falsification-First Evaluation**

Venue: **IEEE ICDM 2026 Teen Research Track / High School Student Research Symposium**

Primary-source requirements re-verified 2026-08-28 from the official ICDM 2026 Teen Research Symposium call:

- submission deadline: **August 30, 2026, AoE**;
- maximum length: **5 pages total, including figures, tables, and references**;
- format: **IEEE Computer Society Proceedings** manuscript format;
- review: **single-blind**;
- first author must be a high-school student and primary contributor;
- first-author affiliation must explicitly include **High School Student**;
- submission must use the **Teen Research Track**.

Official call: https://icdm2026.neu.edu.cn/CallforTeen_en/

## Scientific decision

**GO CONDITIONALLY — scientifically defensible as a narrow negative/falsification paper; not yet upload-ready.**

The frozen conclusion must remain negative/inconclusive. This package does not authorize any rescue experiment, access to the locked ARC test split, post-hoc threshold/seed changes, or wording that implies architecture/mechanism superiority.

Canonical scientific manuscript source remains `MANUSCRIPT_DRAFT_NEGATIVE_ARC.md`. The editable submission source on this branch is `paper/main.tex`. The pre-falsification root `paper.tex` is not evidence-authorized.

## Submission thesis / contribution boundary

The defensible thesis is:

> Under a frozen ARC-Challenge validation protocol, the tested project-named LAM-JEPA configuration did not outperform a gradient-active-parameter-matched supervised comparator and did not establish the preregistered planner or EMA-target contribution criteria; preserving the adverse result, frozen stop rule, source audit, and reproducibility trail provides a useful falsification/reproducibility case study.

Allowed contribution claims:

1. frozen ARC-Challenge evaluation with retained eligibility/exclusion evidence;
2. parameter-matched supervised comparison;
3. five-seed planner/target ablations plus deterministic shuffled-label validity control;
4. bounded pinned DeBERTa-v3-xsmall characterization comparison;
5. documented trainability repair whose later validation remained negative/inconclusive;
6. explicit stop rule keeping the failed line's confirmatory test locked;
7. source-level audit narrowing architecture/novelty language to the mechanism actually tested;
8. project-controlled independent reruns that reproduce aggregate scientific conclusions and verifier outputs, with low-level floating-point drift explicitly disclosed.

Forbidden claims include ARC superiority, planner benefit, target-path benefit, quantization benefit, Transformer reasoning capability, canonical I-JEPA implementation, broad JEPA failure, broad benchmark inferiority/superiority, bitwise cross-run reproducibility, external independent reproduction, or research completeness.

## Evidence already present

- Scientific source SHA recorded in the manuscript: `760aa7f9a73a177d5ff4ba7eb470f7e68ace63cb`.
- Frozen validation seeds: `1,2,3,4,5`.
- Frozen training budget: 20 epochs, batch size 32, learning rate `3e-4`, one planner step.
- Eligible ARC rows: train `1117/1119`; validation `295/299`.
- Locked ARC test: not downloaded/evaluated for this failed line.
- Full LAM-JEPA validation accuracy: `0.2549152542 ± 0.0129968064`.
- Matched supervised validation accuracy: `0.2664406780 ± 0.0154600058`.
- Paired LAM-minus-matched: `-0.0115254237 ± 0.0140994131`.
- Full-minus-no_planner: `+0.0047457627`, retained bootstrap 95% CI `[0.0, 0.0142372881]`.
- Full-minus-no_target: `-0.0067796610`, retained bootstrap 95% CI `[-0.0135593220, 0.0]`.
- Repaired validation verdict remains `VALID_NEGATIVE_OR_INCONCLUSIVE_VALIDATION`.
- Project-controlled independent rerun artifacts/digests are identified in the canonical manuscript and provenance files.

These values must be regenerated/checked against retained artifacts before final PDF release; this file is not a new source of scientific truth.

## Remaining hard gates

### 1. Venue formatting — BLOCKED

`paper/main.tex` is currently venue-neutral (`article`) rather than IEEE proceedings format. Convert the evidence-authorized text to an IEEE Computer Society proceedings-compatible source. Do not import claims from stale `paper.tex`.

### 2. Five-page proof — BLOCKED

Build the exact final source and retain the PDF. Verify **<=5 pages total including references, figures, and tables**. Do not claim this gate from source inspection alone.

### 3. Author metadata — OWNER CONTROLLED

Before PDF release, replace placeholders with the truthful final author list/order, approved affiliation(s), and contact information. The first author's affiliation must explicitly include `High School Student`, and the high-school first author must truly be the primary contributor.

### 4. Claim-to-evidence audit — REQUIRED

For every quantitative statement in the final five-page paper, verify:

`claim -> table/figure/text -> processed metric -> raw retained artifact -> frozen config/protocol -> scientific source SHA`.

Use `CLAIM_LEDGER.md`, `MANUSCRIPT_PROVENANCE.md`, `INDEPENDENT_PAPER_ASSET_VERIFICATION_20260814.md`, and `REPRODUCE.md`; resolve any mismatch by weakening/removing the claim, never by changing evidence.

### 5. Numerical/figure consistency — REQUIRED

Regenerate or re-check all retained paper assets. The final PDF values must match the evidence-authorized artifacts exactly to the stated precision. No hand-edited result number may supersede generated evidence.

### 6. Related work / citations — REQUIRED

Keep the originality boundary conservative. Verify each citation in the final bibliography against the cited work and remove any unsupported novelty statement. Generic novelty claims for VQ, latent actions, latent planners, EMA targets, or JEPA are not allowed.

### 7. External review — NOT A VENUE FACT

Independent outside review/reproduction is desirable but is not represented here as an official ICDM Teen Track requirement. If absent, say absent. Do not use it to falsely block a scientifically honest submission, and do not claim it occurred.

### 8. Repository/license metadata — OWNER CONTROLLED

If code is linked/released with the submission, owner-approved license/release metadata and third-party compatibility must be resolved. This is separate from the scientific paper gate.

### 9. Upload consistency — USER ACTION

At submission time:

- choose `Teen Research Track`;
- upload the exact audited final PDF;
- enter authors/affiliations/title/abstract matching the PDF;
- verify first-author high-school designation;
- retain the submission receipt and paper ID.

## Hard GO/NO-GO rule

**GO** only if all of the following are true before the official deadline:

- IEEE proceedings format used;
- final PDF builds and is <=5 pages total;
- author/affiliation metadata is truthful and complete;
- every central scientific/numerical claim passes provenance audit;
- no stale positive claim enters the PDF;
- locked ARC test remains untouched;
- final PDF/source are retained as the exact submitted artifacts.

Otherwise **NO-GO for submission in the current state**. Do not weaken scientific gates to meet the date.
