# LAM-JEPA Manuscript Provenance Matrix

**As of:** 2026-08-14  
**Scope:** frozen ARC negative/inconclusive paper line only. This file does not authorize test access, hyperparameter rescue, publication, or a superiority claim.

## Provenance rule

Every quantitative manuscript statement must resolve as:

`claim -> table/figure -> processed metric -> raw artifact -> frozen protocol/config -> scientific code revision`

A missing edge is a paper-package blocker. Processed summaries are not substitutes for raw-artifact pointers.

## A. Full controls and mechanism claims — COMPLETE

| Claim / display | Raw retained evidence | Frozen execution/source | Status |
|---|---|---|---|
| five-seed full / `no_planner` / `no_target` / shuffled-label ARC validation | workflow `31203337502`; successful rerun artifact `9162165932`, digest `sha256:caa898f1ff046a337db9b5ddbffe1b332943a732868e2fd809abeda8ee89c30b`; independent successful artifact `9149336081`, digest `sha256:c45710b5dae6a767ccb6bab7f6e3d8e9578752d8cf9b79fd82a65ae824dded1b` | protocol v3; seeds 1–5; 20 epochs; batch 32; LR `3e-4`; model steps 1; train 1117; validation 295; scientific SHA `760aa7f9a73a177d5ff4ba7eb470f7e68ace63cb` | **COMPLETE** |
| full accuracy and planner/target paired effects | `arc-protocol-v3-full-controls-validation.json` inside artifact `9162165932`; strict verifier report in same artifact | same | **COMPLETE** |
| shuffled-label control below frozen 0.35 ceiling | same artifact; deterministic label permutation seed `20260807` | same | **COMPLETE** |
| low-order floating drift does not change aggregate/verifier conclusion | artifacts `9149336081` and `9162165932`; `EVIDENCE_AUDIT_20260813.md` | same | **COMPLETE** |

The raw full-controls artifact reports full mean `0.2549152493476868`, `no_planner` mean `0.25016948878765105`, `no_target` mean `0.26169490814208984`, shuffled-label mean `0.263050839304924`, and paired mechanism effects consistent with the manuscript. Small final-decimal differences in prose are rounding/representation only.

## B. Gradient-active-parameter-matched supervised comparison — COMPLETE

GitHub Actions workflow **ARC Protocol V3 Matched Baseline**, run `31203337225`, executed at scientific SHA `760aa7f9a73a177d5ff4ba7eb470f7e68ace63cb` and completed successfully.

Raw full-validation artifact:

- artifact ID: **`9003785715`**;
- name: `arc-matched-v3-full-validation`;
- artifact digest: **`sha256:13268856e9be2d9da91addc7935a9cd7bdc4bc7b0a59e527905c6de6fa0f87cc`**;
- raw result file: `matched-v3-full-validation.json`;
- verifier: `matched-v3-full-validation-verification.json`;
- verifier verdict: `PROTOCOL_V3_MATCHED_BASELINE_EXECUTION_VERIFIED_ONLY`.

The raw result itself records:

- 1,117 eligible training rows and 295 eligible validation rows;
- seeds `[1,2,3,4,5]`;
- 20 epochs, batch size 32, learning rate `0.0003`;
- 700 optimization steps per model per seed;
- CPU execution;
- LAM total trainable parameters `200,020`;
- LAM gradient-active parameters `86,372`;
- matched-supervised trainable/active parameters `86,644`;
- matched depth 1, hidden size 752;
- relative active-parameter gap `0.0031491687`;
- LAM accuracy `0.2549152493476868 ± 0.01299680055624953`;
- matched-supervised accuracy `0.2664406806230545 ± 0.015460000271125466`;
- paired LAM-minus-matched `-0.011525431275367736 ± 0.014099405697391458`;
- `test_split_accessed=false`.

**Status: COMPLETE RAW PROVENANCE.**

## C. Pinned pretrained characterization — COMPLETE, BOUNDED CLAIM ONLY

GitHub Actions workflow **ARC pretrained baseline**, run `31203337145`, completed successfully at scientific SHA `760aa7f9a73a177d5ff4ba7eb470f7e68ace63cb`.

Raw DeBERTa artifact:

- artifact ID: **`9003740436`**;
- name: `arc-pretrained-v2-deberta-smoke`;
- artifact digest: **`sha256:ff63544689d995c162b2eea3850fd06032115485b8007c6ccc5b01f8689c9b8d`**;
- raw result file: `arc-pretrained-v2-deberta.json`;
- verifier: `arc-pretrained-v2-deberta-verification.json`;
- verifier verdict: `PROTOCOL_V2_PRETRAINED_BASELINE_EXECUTION_VERIFIED_ONLY`.

The raw artifact pins `microsoft/deberta-v3-xsmall@14809e4f1fe1895fcba8b258271a940c6ca45ec4`, records `transformers==4.57.6`, 70,830,337 trainable parameters, CPU execution, two seeds, 8 training examples, 16 validation examples, one epoch and one train step per seed. Its summary is exactly:

- LAM `0.15625 ± 0.0441941738`;
- pretrained `0.21875 ± 0.0441941738`;
- paired LAM-minus-pretrained `-0.0625 ± 0.0883883476`.

This is **development characterization only**. The artifact itself states it is not the final five-seed budget, not compute-matched, not an independent reproduction, not a test result, and not evidence that either model is superior.

**Status: COMPLETE RAW PROVENANCE / CLAIM REMAINS BOUNDED.**

## D. Repaired-v5 line — STRONG, SEPARATE FROM ORIGINAL RESULT

The trainability repair and later negative/inconclusive validation remain separately versioned:

- repair: `df249086e9171febaa77333a4c62888f35265c40`;
- protocol freeze: `168f6beb434610752da4cb2cb6161f15ee026663`;
- validation: `18bd608a05bc308056e6279b347ff3ddb2b751be`;
- verifier-only tolerance fix: `05c039fcc02c09c0aa1c1487596dcdd741ee6d51`.

The repair may be described as passing its bounded trainability gate. It did not rescue the original generalization or mechanism claims.

## E. Source-level method claims — COMPLETE

`METHOD_SOURCE_AUDIT_20260814.md` pins the manuscript Method section to executable source. Verified consequences include:

- deterministic BLAKE2b whitespace-token hashing modulo 256;
- ARC max length 96 and zero-valued numeric ARC input;
- token embedding + learned positions + LayerNorm + mean pooling with `self.encoder = nn.Identity()`;
- 32-code quantizer and learned sparse memory;
- 8-action latent transition with one ARC model step;
- same-input EMA target path with momentum `0.996`;
- ARC objective `CE + 0.5*alignment + 0.25*quantization + 0.25*trajectory`;
- `no_target` substitutes detached online `z` while retaining the alignment term;
- AdamW, LR `3e-4`, gradient clip `1.0`, then EMA target update;
- matched baseline uses supervised cross-entropy and gradient-active parameter matching.

The frozen ARC implementation is **not** accurately described as a Transformer encoder or canonical context-to-distinct-target I-JEPA.

## F. Figure/table generation — GENERATOR ESTABLISHED

Canonical deterministic generator:

`scripts/analysis/generate_arc_negative_paper_assets.py`

It consumes the three raw JSON result files above and emits:

- `arc_validation_accuracy.csv`;
- `ARC_NEGATIVE_RESULT_TABLES.generated.md`;
- `arc_validation_accuracy.generated.svg`;
- `manifest.json`.

The SVG is deliberately plain and labels its error bars as **sample standard deviation**, not confidence intervals. The generator asserts protocol-v3 seed/budget/eligibility fields before producing paper assets.

Reproduction pattern after downloading/extracting the three retained artifacts:

```bash
python scripts/analysis/generate_arc_negative_paper_assets.py \
  --full-controls /path/to/arc-protocol-v3-full-controls-validation.json \
  --matched /path/to/matched-v3-full-validation.json \
  --pretrained /path/to/arc-pretrained-v2-deberta.json \
  --out-dir paper/generated
```

Existing paper displays remain generated views, never independent sources. Any final comparison figure must show the adverse matched baseline and shuffled-label control without selective omission.

## G. Metadata defects preserved, not hidden

1. The full-controls raw payload contains a stale prose sentence implying the final five-seed/20-epoch invocation had not occurred. Executed arguments, five seed records and independent verifier evidence establish that it did. The raw artifact remains unchanged and the prose defect is documented.
2. `experiments/reproducibility-wave-20260812.json` historically carries a top-level `scientific_source_commit` pointing to pre-fix seed-order source `2f59b429...` although the canonical full-controls metrics are tied to `760aa7...`. Preserve the old value as lineage and record any correction as metadata only; no scientific metric may change.

## H. Current paper provenance verdict

### GREEN internally

- full/no-planner/no-target/shuffled raw evidence and independent rerun digests;
- exact matched-supervised raw artifact pointer/digest and parameter accounting;
- exact pretrained raw artifact pointer/digest with bounded claim language;
- source-exact ARC model/objective description;
- deterministic table/figure generator;
- locked-test non-access;
- negative scientific conclusion.

### NOT GREEN / outside or owner-controlled

- owner-approved license;
- final authorship and citation metadata;
- independent outside reproduction/reviewer attack;
- any actual submission/publication decision.

**Package state:** `INTERNAL_PROVENANCE_GREEN / SCIENTIFIC_RESULT_NEGATIVE / EXTERNAL_VALIDATION_PENDING / OWNER_METADATA_PENDING / NOT_PUBLISHED`.
