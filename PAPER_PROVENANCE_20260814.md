# PAPER PROVENANCE MAP — 2026-08-14

**Purpose:** enforce `claim → table/figure → processed artifact → raw artifact → experiment config/workflow → code commit` for the ARC negative-result manuscript.  
**Rule:** a missing link is a blocker/flag, not an invitation to infer provenance.

## A. Fully traced full-controls claims

| Claim | Manuscript location | Paper display | Processed evidence | Raw retained evidence | Protocol / command | Scientific source | Provenance state |
|---|---|---|---|---|---|---|---|
| Five-seed frozen full ARC validation executed; test locked | §5.1–5.2, §8.1 | `paper/ARC_NEGATIVE_RESULT_TABLES.md` Table 1 | `experiments/repro_wave_2026_08_12/RESULTS.md`; `EVIDENCE_AUDIT_20260813.md` | Actions run `31203337502` attempts 2/3; artifacts `9149336081` / `9162165932`; digests `c45710…` / `caa898…` | `.github/workflows/arc-protocol-v3-full-controls-validation.yml`; `scripts/benchmark/run_arc_protocol_v3_controls.py`; frozen protocol/verifier | `760aa7f9a73a177d5ff4ba7eb470f7e68ace63cb` | **COMPLETE** |
| Full accuracy `0.2549152493 ± 0.0129968006` | §6.1/6.2 | Table 1 | same full-controls results + verifier | same artifacts | same | `760aa7…` | **COMPLETE** |
| `no_planner` accuracy and paired effect | §6.2, §7 | Tables 1–2 | same full-controls results + bootstrap summary | same artifacts | same runner with frozen ablation | `760aa7…` | **COMPLETE** |
| `no_target` accuracy and paired effect | §6.2, §7 | Tables 1–2 | same full-controls results + bootstrap summary | same artifacts | same runner with frozen ablation | `760aa7…` | **COMPLETE** |
| shuffled-label control `0.2630508393 ± 0.0145011803` below frozen `0.35` ceiling | §6.2, §7 | Table 1 | same full-controls results + strict verifier | same artifacts | same frozen control path | `760aa7…` | **COMPLETE** |
| independent aggregate/verifier reproduction with low-order raw drift | §8.1, §10 | reproducibility text/table | `EVIDENCE_AUDIT_20260813.md`; reproducibility-wave results | attempt-2/3 artifacts and cross-artifact comparison | same frozen workflow | `760aa7…` | **COMPLETE** |

## B. Capacity-matched comparison

| Claim | Manuscript location | Paper display | Processed evidence | Raw retained evidence | Protocol / command | Source | Provenance state |
|---|---|---|---|---|---|---|---|
| active params `86,372` vs `86,644` | §4.7 | `paper/ARC_NEGATIVE_RESULT_TABLES.md` Table 3 + `RESEARCH_STATUS.md` | retained matched-baseline summary | repository says “retained matched-baseline artifact lineage” but the current paper-facing ledgers do **not name a concrete raw artifact ID/digest** | frozen matched-baseline path in repository | frozen matched-baseline lineage | **PARTIAL — RAW ARTIFACT POINTER REQUIRED** |
| LAM `0.2549152542 ± 0.0129968064`; matched `0.2664406780 ± 0.0154600058`; paired `-0.0115254237 ± 0.0140994131` | Abstract, §6.1 | Table 3 | `RESULTS.md`; `RESEARCH_STATUS.md`; `experiments/reproducibility-wave-20260812.json` | raw artifact exists per retained lineage but is not explicitly identified in current paper-facing provenance | frozen matched-baseline path | frozen matched-baseline lineage | **PARTIAL — RAW ARTIFACT POINTER REQUIRED** |

**Action:** locate and record the exact matched-baseline workflow/run/artifact/digest before calling the paper provenance-complete. Do not drop the baseline result; flag the missing pointer.

## C. Bounded pretrained characterization

| Claim | Manuscript location | Paper display | Processed evidence | Raw retained evidence | Protocol / command | Source | Provenance state |
|---|---|---|---|---|---|---|---|
| pinned revision `microsoft/deberta-v3-xsmall@14809e4…`; LAM `0.15625`, DeBERTa `0.21875` | §5.3, §6.3 | Table 4 | `RESULTS.md`; `RESEARCH_STATUS.md`; machine-readable reproducibility-wave summary | exact raw comparator artifact ID/digest is not named in the paper-facing ledgers inspected in this closure | pretrained comparator path retained in repository | pinned model revision + repository comparator code | **PARTIAL — RAW ARTIFACT POINTER REQUIRED** |

This comparison remains development characterization even after provenance is completed.

## D. Repaired-v5 line

| Claim | Manuscript location | Processed evidence | Raw/config lineage | Source lineage | Provenance state |
|---|---|---|---|---|---|
| trainability repair passed bounded train-only gate; repaired validation remained `VALID_NEGATIVE_OR_INCONCLUSIVE_VALIDATION` | §6.4, §8.4 | `RESULTS.md`; `RELEASE_PROVENANCE.md`; v5 result package | protocol `168f6beb434610752da4cb2cb6161f15ee026663`; validation execution `18bd608a05bc308056e6279b347ff3ddb2b751be`; verifier tolerance fix `05c039fcc02c09c0aa1c1487596dcdd741ee6d51` | repair merge `df249086e9171febaa77333a4c62888f35265c40` | **STRONG / artifact IDs should be surfaced in final release bundle if required by venue** |

## E. Source-level method claims

| Method claim | Source file at scientific SHA | State |
|---|---|---|
| hashed whitespace tokenization with BLAKE2b modulo 256 | `src/lam_jepa/data.py` | **VERIFIED** |
| ARC max length 96 and zero `numeric_x` | `src/lam_jepa/benchmarking/arc_challenge.py` | **VERIFIED** |
| `TokenEncoder.encoder = nn.Identity()`, LayerNorm + mean pool | `src/lam_jepa/model.py` | **VERIFIED** |
| 32-code EMA vector quantizer | `src/lam_jepa/model.py` | **VERIFIED** |
| sparse learned key/value memory, configured capacity 64 | `src/lam_jepa/memory.py`; model config | **VERIFIED** |
| 8-action latent transition; one frozen ARC model step | `src/lam_jepa/model.py`; ARC protocol/result ledger | **VERIFIED** |
| same-input EMA `target_z` | `src/lam_jepa/model.py` | **VERIFIED** |
| ARC-specific `CE + 0.5 align + 0.25 quant + 0.25 trajectory` | `src/lam_jepa/benchmarking/arc_challenge.py` | **VERIFIED** |
| `no_target` replaces EMA target with `z.detach()` rather than deleting alignment | `src/lam_jepa/model.py`; ARC loss unchanged | **VERIFIED** |

All source-level method claims above refer to `760aa7f9a73a177d5ff4ba7eb470f7e68ace63cb`.

## F. Metadata defects discovered during provenance closure

1. The frozen raw full-controls artifact contains a stale sentence implying the final five-seed/20-epoch run was not executed; executable arguments and independent verifier prove that it was. The raw artifact remains unchanged and the defect is documented.
2. `experiments/reproducibility-wave-20260812.json` currently has top-level `scientific_source_commit = 2f59b429…` even though its canonical full-validation metrics and current paper lineage are tied to the later frozen full-controls scientific SHA `760aa7…`. The old `2f59b…` value corresponds to the preserved pre-fix seed-order lineage and is also recorded under `pre_fix_reproduction`. This is a **summary-metadata ambiguity** and should be corrected transparently in a new commit while preserving the old value in a correction note.

## G. Paper provenance verdict

### GREEN

- full/no-planner/no-target/shuffled-control scientific values;
- independent full rerun artifact IDs/digests;
- source-level architecture and ARC objective;
- locked-test boundary;
- seed-order bug/fix lineage;
- negative scientific conclusion.

### NOT YET GREEN

- capacity-matched comparison raw artifact pointer/digest in the paper-facing provenance chain;
- pretrained comparator raw artifact pointer/digest in the paper-facing provenance chain;
- owner-approved license/authorship/citation metadata;
- independent external reproduction/review.

**Paper package status:** `EVIDENCE_STRONG / PROVENANCE_PARTIAL / EXTERNAL_VALIDATION_PENDING / NOT_SUBMISSION_READY`.
