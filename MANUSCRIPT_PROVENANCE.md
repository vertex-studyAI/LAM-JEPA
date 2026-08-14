# LAM-JEPA Manuscript Provenance Matrix

**As of:** 2026-08-14  
**Scope:** frozen ARC negative/inconclusive paper line only. This file does not authorize test access, hyperparameter rescue, publication, or a superiority claim.

## Provenance rule

Every quantitative manuscript statement must resolve as:

`claim -> manuscript table/figure -> processed metric -> raw artifact -> frozen protocol/config -> scientific code revision`

A missing edge is a paper-package blocker. Processed summaries are not substitutes for raw-artifact pointers.

## A. Full-controls claims — complete chain

| Claim / display | Processed metric | Raw retained evidence | Protocol / execution | Scientific source | Status |
|---|---|---|---|---|---|
| Frozen five-seed full ARC validation executed; locked test absent | `experiments/repro_wave_2026_08_12/RESULTS.md`; `EVIDENCE_AUDIT_20260813.md` | workflow `31203337502` attempts 2/3; artifacts `9149336081`, `9162165932`; digests `sha256:c45710b5dae6a767ccb6bab7f6e3d8e9578752d8cf9b79fd82a65ae824dded1b` and `sha256:caa898f1ff046a337db9b5ddbffe1b332943a732868e2fd809abeda8ee89c30b` | `.github/workflows/arc-protocol-v3-full-controls-validation.yml`; `scripts/benchmark/run_arc_protocol_v3_controls.py`; seeds 1–5; 20 epochs; batch 32; LR 0.0003; model steps 1; train 1117; validation 295 | `760aa7f9a73a177d5ff4ba7eb470f7e68ace63cb` | **COMPLETE** |
| Full accuracy `0.2549152493 ± 0.0129968006` | full-controls result/verifier | same artifacts | same | `760aa7...` | **COMPLETE** |
| `no_planner` accuracy + paired effect | paired seed aggregate/bootstrap | same artifacts | frozen `no_planner` arm in same runner | `760aa7...` | **COMPLETE** |
| `no_target` accuracy + paired effect | paired seed aggregate/bootstrap | same artifacts | frozen `no_target` arm in same runner | `760aa7...` | **COMPLETE** |
| shuffled-label control `0.2630508393 ± 0.0145011803` below 0.35 ceiling | strict verifier + aggregate | same artifacts | deterministic shuffled-label arm | `760aa7...` | **COMPLETE** |
| aggregate/verifier conclusion independently reproduced with low-order raw drift | `EVIDENCE_AUDIT_20260813.md` | attempt-2/3 artifacts and comparison | independent GitHub-hosted runs of frozen workflow | `760aa7...` | **COMPLETE** |

## B. Capacity-matched supervised comparison — complete chain

The exact full-budget matched-comparison raw artifact was recovered from the frozen V3 workflow rather than inferred from summary prose.

| Claim | Processed evidence | Raw artifact | Protocol / source | Status |
|---|---|---|---|---|
| active parameters LAM `86,372`, supervised `86,644`, ratio `1.0031491687` | `RESULTS.md`; `RESEARCH_STATUS.md`; `EVIDENCE_AUDIT_20260813.md` | workflow run `31203337225`, full job `92948597957`, artifact `9003785715` (`arc-matched-v3-validation-full`), digest `sha256:132688567583372ec7562e0b1f2223f6cb964655df25f410edded392688dda8b` | `.github/workflows/arc-matched-v3-validation.yml`; seeds 1–5; 20 epochs; batch 32; LR `0.0003`; train 1117; validation 295; CPU; scientific source `760aa7...` | **COMPLETE** |
| LAM `0.2549152542 ± 0.0129968064`; matched `0.2664406780 ± 0.0154600058`; paired `-0.0115254237 ± 0.0140994131` | same processed ledgers plus `experiments/reproducibility-wave-20260812.json` | same run/job/artifact/digest | same frozen full-budget matched path | **COMPLETE** |

The full job logs verify the requested 5-seed/20-epoch budget, all 1,117 eligible training rows and 295 eligible validation rows, gradient-active parameter accounting, and validation-only claim boundary.

## C. Bounded pretrained DeBERTa characterization — complete provenance, bounded science

The adverse `0.15625` vs `0.21875` comparison is a **tiny two-seed development smoke**, not a final matched or confirmatory baseline. Completing its provenance does not promote its scientific status.

| Claim | Processed evidence | Raw artifact | Protocol / source | Status |
|---|---|---|---|---|
| frozen `microsoft/deberta-v3-xsmall@14809e4f1fe1895fcba8b258271a940c6ca45ec4`; 70,830,337 trainable parameters; LAM `0.15625`; DeBERTa `0.21875`; paired delta `-0.0625` | issue #10 adverse-evidence record; `RESULTS.md`; `RESEARCH_STATUS.md`; `EVIDENCE_AUDIT_20260813.md` | dedicated `ARC protocol DeBERTa baseline` workflow run `31193106007`; artifact `8999680432` (`arc-protocol-deberta-smoke`); digest `sha256:f56a9e5e76a008dd655f56ace84cd308122990354e12b77a1fff998936f4e8a8` | PR #20 head `e4046d1a9725fe62f32c575c128dc0503e2118a1`; two seeds; 8 train rows; 16 validation rows; one gradient step/seed; locked test absent; canonical implementation later retained via PR #23 | **COMPLETE PROVENANCE / DEVELOPMENT-ONLY CLAIM** |

The verifier verdict was `FROZEN_DEBERTA_BASELINE_EXECUTION_VERIFIED_ONLY`. Issue #10 records that the exact frozen model revision and the adverse means/delta were later reproduced exactly, while still marking the run as non-confirmatory and capacity/compute unmatched.

A different later workflow artifact (`9003740436`) is only another DeBERTa execution smoke and is **not** used as provenance for the `0.21875` manuscript number.

## D. Repaired-v5 line

| Claim | Processed evidence | Frozen lineage | Status |
|---|---|---|---|
| trainability repair passed bounded train-only gate; repaired validation remained `VALID_NEGATIVE_OR_INCONCLUSIVE_VALIDATION` | `RESULTS.md`; `RELEASE_PROVENANCE.md`; repaired-v5 result package | repair `df249086e9171febaa77333a4c62888f35265c40`; protocol freeze `168f6beb434610752da4cb2cb6161f15ee026663`; validation `18bd608a05bc308056e6279b347ff3ddb2b751be`; verifier-only tolerance fix `05c039fcc02c09c0aa1c1487596dcdd741ee6d51` | **STRONG; surface raw artifact IDs in final release manifest if venue requires** |

## E. Source-level method claims — verified

All source claims below refer to scientific SHA `760aa7f9a73a177d5ff4ba7eb470f7e68ace63cb`.

| Method claim | Source | Status |
|---|---|---|
| deterministic BLAKE2b whitespace-token hashing modulo 256 | `src/lam_jepa/data.py` | **VERIFIED** |
| ARC maximum length 96; zero-valued `numeric_x` for every example | `src/lam_jepa/benchmarking/arc_challenge.py` | **VERIFIED** |
| token encoder has embedding + learned positions + LayerNorm + mean pool with `self.encoder = nn.Identity()` | `src/lam_jepa/model.py` | **VERIFIED** |
| 32-code EMA quantizer | `src/lam_jepa/model.py` | **VERIFIED** |
| learned key/value sparse memory; configured capacity 64 | `src/lam_jepa/memory.py`; model config | **VERIFIED** |
| 8-action latent transition; frozen ARC model steps = 1 | `src/lam_jepa/model.py`; frozen ARC protocol | **VERIFIED** |
| target encoder receives the same ARC tokens/numeric input as the online encoder; EMA momentum 0.996 | `src/lam_jepa/model.py` | **VERIFIED** |
| ARC objective is `CE + 0.5*alignment + 0.25*quantization + 0.25*trajectory` | `src/lam_jepa/benchmarking/arc_challenge.py` | **VERIFIED** |
| `no_target` substitutes `z.detach()` while alignment remains present | `src/lam_jepa/model.py`; unchanged ARC loss | **VERIFIED** |

### Consequences for manuscript wording

- The frozen ARC model is **not a Transformer encoder**.
- The ARC experiment does **not** instantiate canonical context-to-distinct-target JEPA prediction; its target is a same-input EMA representation.
- The generic repository `total_loss` is **not** the reported ARC benchmark objective.
- The planner trajectory objective is consistency to the current quantized latent, not direct supervision on a held-out future state.

## F. Figure/table provenance

Existing paper displays are generated views of retained evidence, not independent sources:

- `paper/ARC_NEGATIVE_RESULT_TABLES.md` — canonical frozen result tables;
- `paper/figures/arc_mechanism_effects.svg` — mechanism-effect visualization from retained paired effects;
- any final comparison figure must show the adverse matched baseline and shuffled-label control without selective omission.

Before public release, each figure/table must expose its generation command/source data in the release manifest.

## G. Metadata defects and corrections

1. The frozen full-controls raw payload contains one stale prose sentence implying the final five-seed/20-epoch invocation had not occurred. Executed arguments and the independent verifier establish that the final budget did execute. The raw artifact remains preserved unchanged; this is a documented reporting-metadata defect.
2. `experiments/reproducibility-wave-20260812.json` historically carried a top-level `scientific_source_commit` pointing to the pre-fix seed-order source `2f59b429...` even though the canonical full-controls metrics are tied to `760aa7...`. This branch corrects that metadata transparently: `760aa7...` is now the full-controls source, while `2f59b429...` is preserved explicitly as the pre-fix seed-order lineage. No raw artifact, result, protocol, seed, threshold, or scientific conclusion changed.

## H. Current paper provenance verdict

### GREEN — internal provenance

- frozen full/no-planner/no-target/shuffled-control quantitative evidence;
- independent full rerun artifact IDs/digests;
- capacity-matched full-budget raw artifact/digest;
- bounded adverse DeBERTa smoke raw artifact/digest and explicit development-only boundary;
- exact ARC source/model/objective boundary;
- locked-test non-access;
- seed-order bug/fix lineage;
- negative scientific conclusion.

### NOT GREEN — release/external gates

- final figure/table generation manifest;
- owner-approved license/authorship/citation metadata;
- independent outside reproduction/review.

**Package state:** `INTERNAL_PROVENANCE_GREEN / EXTERNAL_VALIDATION_PENDING / RELEASE_METADATA_PENDING / NOT_SUBMISSION_READY`.
