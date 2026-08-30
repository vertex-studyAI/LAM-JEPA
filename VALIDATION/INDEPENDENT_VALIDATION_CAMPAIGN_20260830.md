# LAM-JEPA Independent Validation Campaign

**Started:** 2026-08-30
**Scientific state:** reproducible negative result; external validation not yet established.

## Objective
Obtain independent, auditable evaluation of the retained LAM-JEPA result without overstating the evidence. The campaign succeeds only when people outside the author team independently inspect or rerun the frozen artifact and their outcomes are retained whether positive, negative, partial, or failed.

## Frozen headline to validate
Under the frozen ARC-Challenge validation protocol, LAM-JEPA did not outperform the capacity-matched supervised baseline and its planner/target mechanism criteria were not met. Independent internal reruns reproduce the aggregate negative conclusion; low-level raw outputs are not byte-identical across runners.

## Validation targets
1. **Three independent reproductions** of the core frozen experiment.
2. **Five technical audits** of methodology, baselines, statistics, and claim boundaries.
3. **One environment-diverse reproduction** outside the original development machine/runner.
4. **One external critique** focused specifically on whether the negative result is scientifically informative enough for a workshop/negative-results venue.

## Rules
- Never ask a validator to endorse LAM-JEPA.
- Never hide a failed reproduction.
- Never relabel internal reruns as external validation.
- Never broaden the claim beyond the frozen protocol.
- Record validator identity only with permission.
- Record environment, commit SHA, command, deviations, result, and verdict.

## Validator roles
| Role | Ask | Target |
|---|---|---:|
| Reproducer | Run frozen core experiment and return outputs/verdict | 3 |
| Methods reviewer | Identify strongest methodological weakness | 2 |
| Baseline reviewer | Audit fairness/completeness of comparison | 1 |
| Statistics reviewer | Audit uncertainty and decision criteria | 1 |
| Venue-fit reviewer | Assess publishability as negative/reproducibility work | 1 |

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

## Outreach sequence
### Wave A — 10 highly relevant researchers
Ask for a 15-minute methodology critique or a bounded reproduction.

### Wave B — 20 PhD students / research engineers
Ask specifically for independent reproduction using the public artifact.

### Wave C — open reproduction challenge
After two successful external reproductions or after all obvious packaging blockers are fixed, publish a concise call inviting independent reruns and explicitly welcoming failures.

## Success gate
`EXTERNALLY_VALIDATED` may be used only after at least one genuinely independent person outside the author team has completed an auditable rerun or equivalent evaluation. Stronger wording such as `INDEPENDENTLY_REPRODUCED` requires a completed reproduction with retained report and evidence.

## Next actions
- [ ] Freeze exact public commit for validator wave 1
- [ ] Verify one-command reproduction path from clean environment
- [ ] Produce validator report template
- [ ] Recruit first 10 prospects
- [ ] Send first 5 bounded asks
- [ ] Log replies without selective reporting
- [ ] Archive completed reports
- [ ] Update claim ledger only after evidence exists
