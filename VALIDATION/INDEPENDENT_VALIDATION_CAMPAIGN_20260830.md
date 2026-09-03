# LAM-JEPA Independent Validation Campaign

**Started:** 2026-08-30  
**Status reconciled:** 2026-09-03  
**Scientific state:** frozen negative/inconclusive result; **one genuinely external frozen-protocol rerun/review is retained in the repository evidence chain; broader independent reproduction target remains incomplete.**

## Objective
Obtain independent, auditable evaluation of the retained LAM-JEPA result without overstating the evidence. The campaign succeeds only when people outside the author team independently inspect or rerun the frozen artifact and their outcomes are retained whether positive, negative, partial, or failed.

## Frozen headline to validate
Under the frozen ARC-Challenge validation protocol, LAM-JEPA did not outperform the capacity-matched supervised baseline and its planner/target mechanism criteria were not met. Independent internal reruns reproduce the aggregate negative conclusion; low-level raw outputs are not byte-identical across runners.

An external reviewer subsequently reran the frozen protocol, reproduced the retained headline numbers, and identified a material failure-mode correction: retained full/control outputs collapse to constant classifiers, with the measured information-loss chain localized to the vector-quantizer path. The bounded authorized interpretation is a reproducible failure-mechanism report, not architecture superiority or a general JEPA conclusion. See `paper/EXTERNAL_REVIEW_VQ_COLLAPSE_CORRECTION_20260831.md`.

## Validation targets
1. **Three independent reproductions** of the core frozen experiment. **Progress: 1/3 retained external reruns.**
2. **Five technical audits** of methodology, baselines, statistics, and claim boundaries. **Progress: at least 1 substantive external review retained; do not double-count the same person as multiple independent validators.**
3. **One environment-diverse reproduction** outside the original development machine/runner. **Evidence exists for one external rerun, but environment-diversity completion should be claimed only when the retained report records enough environment detail.**
4. **One external critique** focused specifically on whether the negative result is scientifically informative enough for a workshop/negative-results venue. **Still open unless a retained report explicitly addresses venue fit.**

## Rules
- Never ask a validator to endorse LAM-JEPA.
- Never hide a failed reproduction.
- Never relabel internal reruns as external validation.
- Never broaden the claim beyond the frozen protocol.
- Record validator identity only with permission.
- Record environment, commit SHA, command, deviations, result, and verdict.
- A single external reviewer can satisfy several *evidence fields* but must not be counted as several independent people.

## Validator roles
| Role | Ask | Target | Current bounded progress |
|---|---|---:|---|
| Reproducer | Run frozen core experiment and return outputs/verdict | 3 | 1 retained external rerun |
| Methods reviewer | Identify strongest methodological weakness | 2 | 1 external review identified VQ-collapse mechanism and claim corrections |
| Baseline reviewer | Audit fairness/completeness of comparison | 1 | external review checked capacity-match ratio; dedicated baseline-review completion not separately claimed |
| Statistics reviewer | Audit uncertainty and decision criteria | 1 | external review flagged five-seed/bootstrap interpretation; dedicated statistics-review completion not separately claimed |
| Venue-fit reviewer | Assess publishability as negative/reproducibility work | 1 | open |

## Minimum evidence returned by a reproducer
- validator or anonymous validator ID
- date
- repository commit SHA
- operating system / accelerator / Python version
- exact command(s)
- deviations from documented environment
- core metric table
- verifier verdict
- whether aggregate conclusion matched
- unexpected behavior
- signed-off status: REPRODUCED / PARTIAL / NOT REPRODUCED / BLOCKED

## Current retained external evidence

The repository records one external rerun/review in `paper/EXTERNAL_REVIEW_VQ_COLLAPSE_CORRECTION_20260831.md`.

The retained summary states that the reviewer:

- independently reproduced the frozen headline metrics, ARC hashes/eligible counts, per-seed counts, parameter-count comparison, source-method correspondence, and references;
- found constant-classifier collapse across retained conditions/seeds;
- measured the path `distinct pre-quantizer z -> one VQ code -> constant z_q -> constant predictions`;
- observed that removing quantization restored input dependence in a bounded causal check but **did not** establish above-chance task performance;
- required narrower planner/EMA-target language and stronger failure-analysis disclosure.

The summary records that the full external notes arrived as `REVIEW_FOR_ISSUE_102.md` and were summarized into GitHub issue #102. This campaign file does not invent missing environment or identity fields that are not visible in the retained repository summary.

## Outreach sequence
### Wave A — 10 highly relevant researchers
Ask for a 15-minute methodology critique or a bounded reproduction.

### Wave B — 20 PhD students / research engineers
Ask specifically for independent reproduction using the public artifact.

### Wave C — open reproduction challenge
After two successful external reproductions or after all obvious packaging blockers are fixed, publish a concise call inviting independent reruns and explicitly welcoming failures.

## Success gate
`EXTERNALLY_VALIDATED` may be used only after at least one genuinely independent person outside the author team has completed an auditable rerun or equivalent evaluation. That minimum evidence boundary is now supported by the retained external-review correction summary, subject to the exact wording of the claim ledger.

Stronger wording such as **multiple independent reproductions**, **broad external validation**, or a completed 3-reproducer campaign is **not** supported yet. `INDEPENDENTLY_REPRODUCED` should be tied to the one retained report rather than implying three independent reruns.

## Next actions
- [x] Freeze an exact public scientific revision for reproducibility work.
- [x] Produce validator report template.
- [x] Retain one external rerun/review summary with reproduced metrics and material methodological critique.
- [ ] Add `VALIDATION/REPORT_INDEX.md` and a discrepancy ledger so external evidence cannot be lost in manuscript prose.
- [ ] Recover/archive the complete first external report metadata where permission and source availability allow; do not invent missing environment fields.
- [ ] Obtain a **second** independent frozen-protocol rerun/review.
- [ ] Obtain a **third** independent frozen-protocol rerun/review.
- [ ] Obtain a dedicated venue-fit critique for negative/reproducibility publication value.
- [ ] Log replies and failed/blocked attempts without selective reporting.
- [ ] Update claim ledger and paper only from retained evidence.
