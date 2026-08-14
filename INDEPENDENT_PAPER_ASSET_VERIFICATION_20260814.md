# Independent Paper-Asset Verification — 2026-08-14

**Purpose:** independently verify the current LAM-JEPA negative-paper numerical assets from the retained GitHub Actions artifacts without new training, new seeds, or ARC test access.

**Verification surface:** three immutable retained workflow artifacts downloaded independently from GitHub Actions and parsed outside the originating workflow.

## Retained artifact identities

| Evidence | Artifact ID | Downloaded ZIP SHA-256 | Expected canonical SHA-256 | Digest check |
|---|---:|---|---|---|
| ARC v3 full controls | `9162165932` | `caa898f1ff046a337db9b5ddbffe1b332943a732868e2fd809abeda8ee89c30b` | `caa898f1ff046a337db9b5ddbffe1b332943a732868e2fd809abeda8ee89c30b` | **PASS** |
| ARC v3 matched supervised | `9003785715` | `13268856e9be2d9da91addc7935a9cd7bdc4bc7b0a59e527905c6de6fa0f87cc` | `13268856e9be2d9da91addc7935a9cd7bdc4bc7b0a59e527905c6de6fa0f87cc` | **PASS** |
| bounded pinned DeBERTa characterization | `9003740436` | `ff63544689d995c162b2eea3850fd06032115485b8007c6ccc5b01f8689c9b8d` | `ff63544689d995c162b2eea3850fd06032115485b8007c6ccc5b01f8689c9b8d` | **PASS** |

Raw JSON checksums from the downloaded archives:

- `arc-protocol-v3-full-controls-validation.json`: `76aad8b1327e21470aeed137bac341b75b4fcf1f37e5394047642d395e8070f8`
- `matched-v3-full-validation.json`: `7abec096d0875bf60f046f7ec4ec28c9580419f17137cccbb5a3f3311c85a78e`
- `arc-pretrained-v2-deberta.json`: `d5dd1beb32c7d875fb7d5d83ea4e7a57ba0c609c8e4b6595358a7368ac327e58`

## Frozen-protocol assertions

The independently parsed full-controls JSON satisfies:

- protocol ID `lam-jepa-arc-challenge-v3`;
- seeds `[1,2,3,4,5]`;
- epochs `20`;
- batch size `32`;
- learning rate `0.0003`;
- model steps `1`;
- 1,117 eligible train rows;
- 295 eligible validation rows;
- test split policy: `not downloaded or evaluated by this development command`.

The matched-supervised artifact additionally records `protocol.test_split_accessed=false`.

No ARC test artifact was used in this verification.

## Independently recomputed validation table

The table below is reconstructed directly from the downloaded raw JSON payloads, not copied from manuscript prose.

| System | Mean accuracy | Sample SD | n |
|---|---:|---:|---:|
| LAM-JEPA full | `0.2549152493` | `0.0129968006` | 5 |
| No planner | `0.2501694888` | `0.0129968006` | 5 |
| No target | `0.2616949081` | `0.0203953938` | 5 |
| Shuffled labels | `0.2630508393` | `0.0145011803` | 5 |
| Matched supervised | `0.2664406806` | `0.0154600003` | 5 |

Matched paired LAM-minus-supervised:

- mean `-0.0115254313`;
- sample SD `0.0140994057`;
- n `5`.

## Independently checked mechanism effects

### Full minus `no_planner`

- seed-level differences: `[0.0, 0.0, 0.0237288028, 0.0, 0.0]`;
- mean `+0.0047457606`;
- paired SD `0.0106118432`;
- retained bootstrap 95% interval `[0.0, 0.0142372817]`;
- frozen mechanism numeric criterion: **FAIL**.

### Full minus `no_target`

- seed-level differences: `[0.0, 0.0, -0.0169491470, 0.0, -0.0169491470]`;
- mean `-0.0067796588`;
- paired SD `0.0092834301`;
- retained bootstrap 95% interval `[-0.0135593176, 0.0]`;
- frozen mechanism numeric criterion: **FAIL**.

These values preserve the existing negative mechanism conclusion.

## Bounded pinned-pretrained characterization

The independently parsed retained DeBERTa smoke artifact reports:

- LAM mean `0.15625`, sample SD `0.0441941738`, n `2`;
- pinned pretrained mean `0.21875`, sample SD `0.0441941738`, n `2`;
- paired LAM-minus-pretrained mean `-0.0625`, sample SD `0.0883883476`, n `2`.

This remains **development characterization only**. It is not the five-seed matched baseline, is not compute matched, is not an independent reproduction, does not access the ARC test split, and does not establish model superiority/inferiority.

## Independent paper-asset regeneration

Using the same deterministic schema and formatting logic as `scripts/analysis/generate_arc_negative_paper_assets.py`, the three downloaded raw JSON files regenerated:

- `arc_validation_accuracy.csv` — SHA-256 `4c53665775e60832202ced9b143fc649486f6cbc786df399f77c47f367b17356`
- `ARC_NEGATIVE_RESULT_TABLES.generated.md` — SHA-256 `ae9ff3c346b99049789190ea8b4c33f5009fbd5cb186b28bca564dee12ef98b9`
- `arc_validation_accuracy.generated.svg` — SHA-256 `db1304b32a833cd904fe380279d9fa21aa27b5536561297aba75f5f24b3f05d6`

The regenerated SVG uses sample-standard-deviation error bars, exactly as labeled; they are not confidence intervals.

## Verification verdict

`GREEN — INTERNAL PAPER NUMERICAL / ASSET PROVENANCE VERIFIED`

What this closes:

- raw artifact identity for full controls;
- raw artifact identity for the gradient-active-parameter-matched supervised comparison;
- raw artifact identity for the bounded pinned pretrained characterization;
- independent digest validation of all three retained bundles;
- independent reconstruction of the reported primary table and mechanism effects;
- deterministic paper-asset generation from retained evidence.

What this **does not** close:

- owner-controlled license/authorship/`CITATION.cff` decisions;
- independent outside reproduction/review;
- submission/publication;
- any positive superiority or mechanism claim;
- the locked ARC test, which remains unused for this failed hypothesis line.

No scientific result, seed, threshold, dataset, protocol, source claim, or locked-test state was changed by this verification.
