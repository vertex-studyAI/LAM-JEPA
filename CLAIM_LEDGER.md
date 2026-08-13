# LAM-JEPA Claim Ledger

**Evidence cutoff:** 13 August 2026  
**Rule:** every public scientific statement must map to retained evidence. A GREEN negative result is allowed; unsupported positive wording is not.

| ID | Claim | Evidence | Status | Public wording |
|---|---|---|---|---|
| C01 | Frozen ARC full-controls validation executed with 5 seeds, 20 epochs, all eligible train/validation rows, test locked | `experiments/repro_wave_2026_08_13/experiment_metadata.json`; `REPRODUCE.md`; frozen protocol/verifier | VERIFIED | Allowed |
| C02 | Full LAM-JEPA mean validation accuracy is ~0.2549 on the frozen five-seed ARC validation | retained full-controls artifacts/metadata | VERIFIED | Allowed with protocol scope |
| C03 | Capacity-matched supervised baseline has higher mean validation accuracy than LAM-JEPA | retained matched-baseline evidence; `RESEARCH_STATUS.md` | VERIFIED | Allowed; do not generalize beyond tested protocol |
| C04 | LAM-JEPA superiority on ARC is supported | same evidence as C03 | FALSIFIED / UNSUPPORTED | Forbidden |
| C05 | Planner provides a validated ARC contribution | full − `no_planner` = +0.0047457606, 95% bootstrap CI [0.0, 0.0142372817], frozen criterion unmet | FALSIFIED / UNSUPPORTED | Forbidden |
| C06 | Target path provides a validated ARC contribution | full − `no_target` = −0.0067796588, 95% bootstrap CI [−0.0135593176, 0.0], frozen criterion unmet | FALSIFIED / UNSUPPORTED | Forbidden |
| C07 | Deterministic shuffled-label control stayed below frozen 0.35 ceiling | mean 0.2630508393; metadata/verifier | VERIFIED | Allowed as control result |
| C08 | Independent reruns reproduce the aggregate scientific result and verifier verdict | workflow `31203337502`; retained attempts 2/3 plus independent attempt-4 audit in `experiments/repro_wave_2026_08_13/independent_audit.json` | VERIFIED | Allowed |
| C09 | Independent reruns are byte-identical at raw prediction/checkpoint level | artifact comparison; deterministic replay metadata | FALSE | Forbidden |
| C10 | Low-order cross-run floating-point drift exists while aggregate scientific conclusion is unchanged | retained artifact comparisons; latest attempt-3→4 maximum observed numeric drift `0.0007445961236953735`; aggregate summary exact | VERIFIED | Allowed |
| C11 | Pre-fix training replay had a seed-order reproducibility defect | retained pre-fix evidence; PR #61 lineage | VERIFIED | Allowed |
| C12 | Seed-order repair changed the scientific ARC protocol | PR #61/metadata | FALSE | Forbidden |
| C13 | Repaired-v5 trainability gate passed | repaired-v5 retained evidence | VERIFIED | Allowed only as trainability result |
| C14 | Repaired-v5 validation established generalization or quantization benefit | repaired-v5 verdict `VALID_NEGATIVE_OR_INCONCLUSIVE_VALIDATION` | UNSUPPORTED | Forbidden |
| C15 | Locked ARC confirmatory test was used to rescue the failed line | protocol/evidence | FALSE | Forbidden |
| C16 | Locked ARC confirmatory test remains unopened for this failed hypothesis line | protocol/metadata | VERIFIED | Allowed |
| C17 | Frozen raw full-controls output contains a stale claim-boundary sentence | metadata `reporting_metadata_defect` | VERIFIED | Allowed as documented reporting defect |
| C18 | The stale sentence invalidates the scientific metrics | independent command/protocol/verifier evidence | FALSE | Forbidden |
| C19 | LAM-JEPA is research-complete/submission-ready | open packaging/bibliography/legal gates | FALSE | Forbidden |
| C20 | LAM-JEPA is GREEN for a reproducible negative scientific result | C01–C18 plus preserved evidence | VERIFIED, claim-specific | Allowed with explicit qualifier |

## Canonical headline

> Under the frozen ARC-Challenge validation protocol, LAM-JEPA did not outperform the capacity-matched supervised baseline and its planner/target mechanism criteria were not met. Independent reruns through retained attempt 4 reproduce the aggregate negative conclusion and verifier verdict, while low-level raw floating-point outputs are not byte-identical across runners.

## Never collapse these states

`REPRODUCED_NEGATIVE_RESULT` does not imply `SUPERIOR`, `EXTERNALLY_VALIDATED`, `SUBMISSION_READY`, `PRODUCTION_VERIFIED`, or `RESEARCH_COMPLETE`.
