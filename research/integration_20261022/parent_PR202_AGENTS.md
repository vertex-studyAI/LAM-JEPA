# Research-session entry contract

The single canonical project state is `research/RESEARCH_STATE.json`, maintained
on [PR #203](https://github.com/vertex-studyAI/LAM-JEPA/pull/203), branch
`codex/paired-statistical-report-20261010`. This planner branch does not create another state.

## Read before work

Read the local canonical state when present. If it is absent on this review
branch, first read the exact pinned canonical snapshot:
[research/RESEARCH_STATE.json at d40a4c4a3a9490a252265b1c54229506dd6a7b3a](https://github.com/vertex-studyAI/LAM-JEPA/blob/d40a4c4a3a9490a252265b1c54229506dd6a7b3a/research/RESEARCH_STATE.json).

Then resolve the current head of canonical PR #203 and read that same path at
the resolved immutable commit, so later append-only history is not lost by
using an older pin. The pin is a concrete fallback and provenance anchor; it
must never be used to overwrite newer canonical history. Follow the state's
applicable historical truth/protocol references.

For planner work, also read:

- `FINAL_STATUS_2026-09-30.md`
- `protocols/BEAM_PLANNER_V2_DEVELOPMENT_20261010.md`
- `protocols/BEAM_PLANNER_V2_NUMERICS_20261010.md`
- `research/verification/maintenance_20261010/receipt.json`

## Update before finishing

Append the session's actual decisions, verification identities, failures,
unresolved gates and next action to that same canonical state on its current
branch. Preserve every existing field and historical record. Link any local
planner development record and exact source revision there; do not create a
second canonical file on this branch or silently replace a newer state with
the pinned snapshot.

## Closed-study boundary

The original ARC study remains closed as a reproducible negative/failure-mechanism
study. Its locked confirmatory test stays unopened and cannot be used to rescue
the failed line. The opt-in planner-v2 implementation and its numerical checks
are engineering maintenance only. Preserve frozen code, outputs, claims and
negative/invalid/inconclusive history. A material change to the hypothesis,
mechanism, evaluation or scientific claim requires a separately versioned
successor hypothesis, falsifier, protocol and authorized budget. Engineering
test passes do not establish efficacy or complete scientific checkpoints.
