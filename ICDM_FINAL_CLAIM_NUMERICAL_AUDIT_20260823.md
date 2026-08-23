# ICDM 2026 Teen Track — Final Claim and Numerical Audit

**Audit date:** 23 August 2026
**Paper branch:** `paper/icdm-teen-ieee-draft-20260823`
**Canonical manuscript:** `MANUSCRIPT_DRAFT_NEGATIVE_ARC.md`
**IEEE submission draft:** `paper/icdm2026_teen_negative_arc.tex`
**Claim source:** `CLAIM_LEDGER.md`
**Numerical source:** `INDEPENDENT_PAPER_ASSET_VERIFICATION_20260814.md`

## Verdict

**GREEN — INTERNAL CLAIM / NUMERICAL AUDIT PASSED**

This audit closes the final internal claim-boundary and numerical-cross-check gates for the IEEE draft. It does not close owner-controlled authorship/licensing metadata, genuine outside review/reproduction, or actual submission.

## Claim-boundary audit

| Claim family | Ledger state | IEEE draft state | Verdict |
|---|---|---|---|
| Frozen five-seed ARC validation executed; test locked | VERIFIED | Stated with frozen protocol and locked-test boundary | PASS |
| LAM-JEPA mean validation accuracy ≈ 0.2549 | VERIFIED | Reported as `0.2549 ± 0.0130` | PASS |
| Matched supervised mean exceeds LAM-JEPA mean | VERIFIED | Reported as `0.2664 ± 0.0155` vs `0.2549 ± 0.0130` | PASS |
| ARC superiority supported | FALSIFIED / UNSUPPORTED | Explicitly denied | PASS |
| Planner benefit supported | FALSIFIED / UNSUPPORTED | Explicitly denied | PASS |
| EMA target-path benefit supported | FALSIFIED / UNSUPPORTED | Explicitly denied | PASS |
| Shuffled-label control below 0.35 ceiling | VERIFIED | Reported as control, not positive architecture evidence | PASS |
| Aggregate rerun conclusion reproduced | VERIFIED | Bounded to separate CI reruns; no claim of genuine outside reproduction | PASS |
| Raw outputs byte-identical | FALSE | Draft explicitly reports low-order drift instead | PASS |
| Trainability repair establishes generalization/quantization benefit | UNSUPPORTED | Explicitly denied | PASS |
| Confirmatory ARC test used to rescue failed line | FALSE | Explicit stop rule preserves lock | PASS |
| Current system is a Transformer / canonical distinct-target I-JEPA | Unsupported by source audit | Explicitly denied | PASS |
| General JEPA failure follows from this experiment | Unsupported | Explicitly denied | PASS |

## Numerical audit

The IEEE draft's rounded values are consistent with independently reconstructed retained artifacts:

| Quantity | Independently verified value | IEEE value | Verdict |
|---|---:|---:|---|
| LAM-JEPA mean accuracy | `0.2549152493` | `0.2549` | PASS |
| LAM-JEPA sample SD | `0.0129968006` | `0.0130` | PASS |
| Matched supervised mean | `0.2664406806` | `0.2664` | PASS |
| Matched supervised sample SD | `0.0154600003` | `0.0155` | PASS |
| Paired LAM − matched mean | `-0.0115254313` | `-0.0115` | PASS |
| Paired SD | `0.0140994057` | `0.0141` | PASS |
| No-planner mean | `0.2501694888` | `0.2502` | PASS |
| No-target mean | `0.2616949081` | `0.2617` | PASS |
| Shuffled-label mean | `0.2630508393` | `0.2631` | PASS |
| Full − no-planner mean | `+0.0047457606` | `+0.00475` | PASS |
| Full − no-planner 95% bootstrap CI | `[0.0, 0.0142372817]` | `[0, 0.01424]` | PASS |
| Full − no-target mean | `-0.0067796588` | `-0.00678` | PASS |
| Full − no-target 95% bootstrap CI | `[-0.0135593176, 0.0]` | `[-0.01356, 0]` | PASS |
| Bounded LAM comparator mean | `0.15625` | `0.15625` | PASS |
| Bounded DeBERTa comparator mean | `0.21875` | `0.21875` | PASS |
| Bounded paired difference | `-0.0625` | `-0.0625` | PASS |

The repaired-validation rounded values in the IEEE draft also preserve the retained negative/inconclusive verdict and are not promoted into a positive result.

## Protocol audit

The draft preserves the verified frozen protocol:

- seeds 1–5;
- 20 epochs;
- batch size 32;
- learning rate `0.0003`;
- one model step;
- 1,117 eligible training rows;
- 295 eligible validation rows;
- no confirmatory ARC test access.

## Formatting / release audit

The paper CI previously produced a **3-page** IEEE draft and enforces the venue's **≤5-page** gate. Visual inspection recorded no clipping or broken glyphs. A final bibliography entry landing on page 3 is aesthetic layout polish, not a page-limit or scientific-integrity failure.

## Remaining release blockers

Only evidence that cannot be created by internal audit remains:

1. truthful author list/order and first-author eligibility confirmation;
2. approved affiliation/contact metadata;
3. license / third-party compatibility and optional `CITATION.cff` decision;
4. genuine outside review/reproduction if retained as a project release gate;
5. real submission-system access and final submission receipt / paper ID.

No experiment, seed, threshold, test-set access, or scientific conclusion was altered by this audit.
