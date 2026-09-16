# LAM-JEPA publication gate — 17 September 2026

## Scope

This gate records the paper-facing state after integrating the seed-level mechanism-ablation decomposition into the manuscript. It does **not** modify the frozen ARC protocol, data, seeds, thresholds, model, held-out policy, or scientific verdict.

Publication branch: `automation/publication-seed-ablation-20260916`

Gate head before this note: `a5cb2a3825dcd562023570a084e15bae0302cc28`

Frozen scientific source: `760aa7f9a73a177d5ff4ba7eb470f7e68ace63cb`

Confirmatory ARC test: **locked / not evaluated for this failed hypothesis line**.

## Evidence reconciled

The retained frozen protocol uses five seeds and 295 eligible validation rows. The independent audit records these seed-level accuracies:

- full: `0.2406779677, 0.2644067705, 0.2644067705, 0.2406779677, 0.2644067705`;
- `no_planner`: `0.2406779677, 0.2644067705, 0.2406779677, 0.2406779677, 0.2644067705`;
- `no_target`: `0.2406779677, 0.2644067705, 0.2813559175, 0.2406779677, 0.2813559175`;
- shuffled-label control: `0.2644067705, 0.2406779677, 0.2813559175, 0.2644067705, 0.2644067705`.

Because every seed uses the fixed 295-row cohort, those values map to exact correct-count vectors:

- full: `[71, 78, 78, 71, 78]`;
- `no_planner`: `[71, 78, 71, 71, 78]`;
- `no_target`: `[71, 78, 83, 71, 83]`;
- shuffled-label: `[78, 71, 83, 78, 78]`.

Therefore:

- full minus `no_planner`: `[0, 0, +7, 0, 0]`, mean accuracy effect `+7 / (5 × 295) = +0.0047457627`;
- full minus `no_target`: `[0, 0, -5, 0, -5]`, mean accuracy effect `-10 / (5 × 295) = -0.0067796610`.

These reproduce the retained mechanism summaries and do not clear the frozen `+0.01` contribution threshold. Their retained bootstrap intervals include zero at a boundary.

## Manuscript integration completed

`paper/main.tex` now contains:

1. the existing aggregate mechanism table and negative H2/H3 interpretation;
2. a seed-level correct-count table for full, `no_planner`, `no_target`, and shuffled-label conditions;
3. explicit wording that the planner aggregate is driven by one seed and the target-path aggregate by two seeds;
4. the bounded interpretation required by the external collapse review: because the reviewed retained runs are constant classifiers, these sparse shifts are not presented as stable input-conditional mechanism effects.

`CLAIM_LEDGER.md` now includes the seed-level decomposition as claim C25 with a bounded public-wording rule.

## Claims preserved

Allowed:

- the frozen ARC line is negative/inconclusive for superiority and mechanism attribution;
- seed-level mechanism deltas are sparse under the fixed 295-row validation cohort;
- project-controlled reruns reproduce the aggregate adverse conclusion and verifier outputs;
- one bounded external frozen-protocol rerun/review reproduced the retained headline result and identified single-code VQ collapse with constant predictions in the reviewed runs;
- removing quantization in that bounded diagnostic restored input dependence but did not establish above-chance ARC performance.

Still forbidden:

- ARC superiority;
- validated planner benefit;
- validated EMA-target benefit;
- quantization benefit;
- general JEPA/vector-quantization/planner failure claims;
- broad independent replication, multi-site validation, or peer-reviewed validation;
- opening the locked ARC confirmatory test to rescue this line.

## Current automated gate state at creation

For branch head `a5cb2a3825dcd562023570a084e15bae0302cc28`:

- Research claim boundary: **success**;
- ARC Successor Freshness Audit: **success**;
- ARC Download Transport CI: **success**;
- Paper Build: **pending at time of this record**;
- Reproducibility CI: **pending at time of this record**;
- ARC Protocol V2 QA: **pending at time of this record**.

Pending checks are not treated as passing.

## Release blockers

Scientific claim reconciliation is no longer the immediate blocker for this PR. Remaining gates are:

1. Paper Build must complete successfully on the exact publication head.
2. Remaining CI/reproducibility checks must complete without changing the frozen evidence interpretation.
3. Authorship/order remains owner-controlled.
4. Repository/publication license remains owner-controlled.
5. Final bibliography/venue-format review remains required.
6. Any stronger independent-replication wording requires additional genuinely independent reports.

## Merge rule

Keep PR #178 draft until the exact-head paper build and required checks are green and owner-controlled release metadata is resolved. Do not merge merely because the scientific result is internally consistent.
