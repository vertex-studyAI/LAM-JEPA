# AISTATS 2027 abstract evidence precision cross-check — 27 September 2026

## Scope

This note checks only numerical presentation in the AISTATS abstract. It does not authorize submission, alter the frozen scientific protocol, reopen the locked ARC test, or change any retained raw result.

## Observed representation difference

Two retained repository surfaces encode the same mechanism effects with slightly different last-digit floating-point values:

| Quantity | Conference-ready gate | ARC negative-result table | Stable abstract value |
|---|---:|---:|---:|
| full - no_planner | +0.0047457627 | +0.0047457606 | +0.0047 |
| planner CI upper endpoint | 0.0142372881 | 0.0142372817 | 0.0142 |
| full - no_target | -0.0067796610 | -0.0067796588 | -0.0068 |
| target CI lower endpoint | -0.0135593220 | -0.0135593176 | -0.0136 |

These differences are below 1e-8 and do not change a sign, confidence-interval boundary, preregistered threshold comparison, or scientific verdict.

The system-level means are likewise reported in the abstract at four decimal places:
- full: 0.2549 +/- 0.0130;
- matched supervised: 0.2664 +/- 0.0155;
- paired mean difference: -0.0115.

## Decision

The abstract and abstract-registration packet now use four-decimal presentation. This is intentionally less precise than the retained evidence tables. The detailed evidence surfaces remain unchanged and continue to retain their original numeric representations.

The scientific interpretation remains exactly the same:
- the superiority gate is not met;
- the planner contribution gate is not met;
- the EMA-target contribution gate is not met;
- the result remains bounded negative/inconclusive;
- the historical confirmatory ARC test remains locked.

## Remaining gates

This precision cleanup does not close:
- final author-set approval and OpenReview profile checks;
- AISTATS submission-quota/reviewer-eligibility checks;
- official AISTATS style-file provenance;
- exact final PDF build and visual inspection;
- final portal metadata entry and timestamped receipt;
- owner-approved release/license metadata.

No model/data execution or outcome-bearing rerun was performed.
