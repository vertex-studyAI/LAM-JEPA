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

## B. Capacity-matched supervised comparison — partial chain

| Claim | Processed evidence | Raw artifact | Protocol / source | Status |
|---|---|---|---|---|
| active parameters LAM `86,372`, supervised `86,644` | `RESULTS.md`; `RESEARCH_STATUS.md`; `EVIDENCE_AUDIT_20260813.md` | repository records a retained matched-baseline lineage, but the paper-facing ledgers inspected in this closure do **not name an exact artifact ID/digest** | frozen matched-baseline path, same validation data/paired seed budget | **PARTIAL — RAW ARTIFACT POINTER REQUIRED** |
| LAM `0.2549152542 ± 0.0129968064`; matched `0.2664406780 ± 0.0154600058`; paired `-0.0115254237 ± 0.0140994131` | same processed ledgers plus `experiments/reproducibility-wave-20260812.json` | exact raw artifact pointer/digest not yet surfaced here | frozen matched-baseline lineage | **PARTIAL — RAW ARTIFACT POINTER REQUIRED** |

The result remains reportable as retained evidence, but the paper package is not provenance-complete until the exact raw matched-baseline artifact is located or the absence is explicitly documented.

## C. Bounded pretrained characterization — partial chain

| Claim | Processed evidence | Raw artifact | Protocol / source | Status |
|---|---|---|---|---|
| pinned `microsoft/deberta-v3-xsmall@14809e4f1fe1895fcba8b258271a940c6ca45ec4`; LAM `0.15625`, DeBERTa `0.21875`, delta `-0.0625` | `RESULTS.md`; `RESEARCH_STATUS.md`; machine-readable reproducibility summary | exact comparator artifact ID/digest is not named in the paper-facing ledgers inspected in this closure | retained pretrained comparator path; pinned model revision | **PARTIAL — RAW ARTIFACT POINTER REQUIRED** |

This comparison is development characterization even if its provenance chain is completed.

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

Existing paper displays must be treated as generated views of retained evidence, not independent sources:

- `paper/ARC_NEGATIVE_RESULT_TABLES.md` — canonical frozen result tables;
- `paper/figures/arc_mechanism_effects.svg` — mechanism-effect visualization from retained paired effects;
- any final comparison figure must show the adverse matched baseline and the shuffled-label control without selective omission.

Before public release, each figure/table must expose its generation command/source data in the release manifest.

## G. Metadata defects and corrections

1. The frozen full-controls raw payload contains one stale prose sentence implying the final five-seed/20-epoch invocation had not occurred. Executed arguments and the independent verifier establish that the final budget did execute. The raw artifact remains preserved unchanged; this is a documented reporting-metadata defect.
2. `experiments/reproducibility-wave-20260812.json` has historically carried a top-level `scientific_source_commit` pointing to the pre-fix seed-order source `2f59b429...` even though the canonical full-controls metrics are tied to `760aa7...`. This must be corrected transparently by preserving the old value as pre-fix lineage and recording a metadata-correction note. No scientific value may change.

## H. Current paper provenance verdict

### GREEN

- frozen full/no-planner/no-target/shuffled-control quantitative evidence;
- independent full rerun artifact IDs/digests;
- exact ARC source/model/objective boundary;
- locked-test non-access;
- seed-order bug/fix lineage;
- negative scientific conclusion.

### NOT GREEN

- exact raw matched-supervised artifact pointer/digest in the paper-facing chain;
- exact raw pretrained-comparator artifact pointer/digest in the paper-facing chain;
- final figure/table generation manifest;
- owner-approved license/authorship/citation metadata;
- independent outside reproduction/review.

**Package state:** `EVIDENCE_STRONG / PROVENANCE_PARTIAL / EXTERNAL_VALIDATION_PENDING / NOT_SUBMISSION_READY`.
