# LAM-JEPA Claim Ledger

**Evidence cutoff:** 5 September 2026  
**Rule:** every public scientific statement must map to retained evidence. A GREEN negative result is allowed; unsupported positive wording is not.

| ID | Claim | Evidence | Status | Public wording |
|---|---|---|---|---|
| C01 | Frozen ARC full-controls validation executed with 5 seeds, 20 epochs, all eligible train/validation rows, test locked | `experiments/repro_wave_2026_08_12/experiment_metadata.json`; `REPRODUCE.md`; frozen protocol/verifier | VERIFIED | Allowed |
| C02 | Full LAM-JEPA mean validation accuracy is ~0.2549 on the frozen five-seed ARC validation | retained full-controls artifacts/metadata | VERIFIED | Allowed with protocol scope |
| C03 | Capacity-matched supervised baseline has higher mean validation accuracy than LAM-JEPA | retained matched-baseline evidence; `RESEARCH_STATUS.md` | VERIFIED | Allowed; do not generalize beyond tested protocol |
| C04 | LAM-JEPA superiority on ARC is supported | same evidence as C03 | FALSIFIED / UNSUPPORTED | Forbidden |
| C05 | Planner provides a validated ARC contribution | full − `no_planner` = +0.0047457606, 95% bootstrap CI [0.0, 0.0142372817], frozen criterion unmet | FALSIFIED / UNSUPPORTED | Forbidden |
| C06 | Target path provides a validated ARC contribution | full − `no_target` = −0.0067796588, 95% bootstrap CI [−0.0135593176, 0.0], frozen criterion unmet | FALSIFIED / UNSUPPORTED | Forbidden |
| C07 | Deterministic shuffled-label control stayed below frozen 0.35 ceiling | mean 0.2630508393; metadata/verifier | VERIFIED | Allowed as control result |
| C08 | Independent project-controlled reruns reproduce the aggregate scientific result and verifier verdict | workflow `31203337502`, attempts 2/3; artifact digests in metadata | VERIFIED | Allowed |
| C09 | Independent project-controlled reruns are byte-identical at raw prediction/checkpoint level | artifact comparison; deterministic replay metadata | FALSE | Forbidden |
| C10 | Low-order cross-run floating-point drift exists while aggregate scientific conclusion is unchanged | artifact comparison, max observed numeric drift ~5.9186e-4 | VERIFIED | Allowed |
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
| C21 | One genuinely external frozen-protocol rerun/review reproduced the retained headline metrics and bounded collapse diagnosis | `paper/EXTERNAL_REVIEW_VQ_COLLAPSE_CORRECTION_20260831.md`; `paper/main.tex` | VERIFIED, bounded external reproduction | Allowed only with one-reviewer/frozen-protocol scope |
| C22 | In the externally reviewed retained runs, the tested quantized path collapsed distinct pre-quantizer latents to a single VQ code per run and produced constant downstream predictions | `paper/EXTERNAL_REVIEW_VQ_COLLAPSE_CORRECTION_20260831.md`; externally reviewed retained artifacts summarized there | VERIFIED, bounded mechanism diagnosis | Allowed; do not generalize to vector quantization or JEPA broadly |
| C23 | Removing quantization in the bounded external diagnostic restored input-dependent predictions and established above-chance ARC performance | `paper/EXTERNAL_REVIEW_VQ_COLLAPSE_CORRECTION_20260831.md` | FALSE | Forbidden; the diagnostic localizes collapse only |
| C24 | The external review constitutes peer-reviewed publication or broad multi-site independent replication | `paper/EXTERNAL_REVIEW_VQ_COLLAPSE_CORRECTION_20260831.md`; current validation campaign state | FALSE | Forbidden |

## Canonical headline

> Under the frozen ARC-Challenge validation protocol, LAM-JEPA did not outperform the capacity-matched supervised baseline and its planner/target mechanism criteria were not met. Independent project-controlled reruns reproduce the aggregate negative conclusion and verifier verdict. One genuinely external frozen-protocol rerun/review also reproduced the retained headline metrics and found that the reviewed quantized path collapsed distinct pre-quantizer latents to a single VQ code per run, yielding constant downstream predictions. This supports a bounded reproducible failure-mechanism report, not architecture superiority or a general JEPA/vector-quantization conclusion.

## Never collapse these states

`REPRODUCED_NEGATIVE_RESULT` does not imply `SUPERIOR`, `SUBMISSION_READY`, `PRODUCTION_VERIFIED`, or `RESEARCH_COMPLETE`.

`ONE_EXTERNAL_FROZEN_PROTOCOL_REPRODUCTION` does not imply `BROAD_INDEPENDENT_REPLICATION`, `MULTI_SITE_VALIDATION`, or `PEER_REVIEWED_PUBLICATION`.
