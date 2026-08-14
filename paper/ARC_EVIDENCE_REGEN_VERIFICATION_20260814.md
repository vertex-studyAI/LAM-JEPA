# ARC Evidence Regeneration Verification — 2026-08-14

**Scope:** internal independent recomputation of the frozen ARC paper numbers from retained raw GitHub Actions artifacts.  
**Scientific outcome:** unchanged — `ARC_SUPERIORITY_AND_MECHANISM_HYPOTHESES_UNSUPPORTED`.  
**External validation:** **NO**. This is an independent recomputation path inside the project workflow, not reproduction by an outside researcher.  
**Locked ARC test:** not downloaded or evaluated.

## Retained inputs materialized

| Evidence | Workflow/artifact lineage | ZIP SHA256 | Raw JSON SHA256 |
|---|---|---|---|
| Full controls | run `31203337502`, artifact `9162165932` | `caa898f1ff046a337db9b5ddbffe1b332943a732868e2fd809abeda8ee89c30b` | `76aad8b1327e21470aeed137bac341b75b4fcf1f37e5394047642d395e8070f8` |
| Capacity matched | run `31203337225`, artifact `9003785715` | `13268856e9be2d9da91addc7935a9cd7bdc4bc7b0a59e527905c6de6fa0f87cc` | `7abec096d0875bf60f046f7ec4ec28c9580419f17137cccbb5a3f3311c85a78e` |
| Pinned pretrained smoke | artifact `9003740436` | `ff63544689d995c162b2eea3850fd06032115485b8007c6ccc5b01f8689c9b8d` | `d5dd1beb32c7d875fb7d5d83ea4e7a57ba0c609c8e4b6595358a7368ac327e58` |

The downloaded ZIP hashes matched the retained GitHub artifact digests. The extracted filenames used were:

- `arc-protocol-v3-full-controls-validation.json`
- `matched-v3-full-validation.json`
- `arc-pretrained-v2-deberta.json`

## Verification method

A separate local verifier did **not** trust the manuscript aggregate tables. It:

1. read the three retained raw JSON files;
2. checked the frozen full-controls protocol: seeds `1..5`, 20 epochs, batch 32, learning rate `0.0003`, one model step, 1,117 eligible train rows, 295 eligible validation rows, and an explicit `test_split_policy` stating the test split was not downloaded/evaluated;
3. recomputed every per-seed accuracy directly from each retained `prediction`/`label` row;
4. independently re-aggregated the retained per-seed float32 metrics with sample standard deviation;
5. independently recomputed `full - no_planner` and `full - no_target` paired effects;
6. independently reran the frozen 10,000-sample paired bootstrap with seeds `20260807` and `20260808`;
7. independently recomputed the matched-supervised and pretrained-smoke accuracies/effects from per-example predictions;
8. regenerated a Markdown evidence table, CSV, and SVG from those recomputed values.

**Checks:** `32/32` aggregate/protocol/statistical checks passed.

## Independently regenerated full-controls values

| Condition | Mean validation accuracy | Sample SD | n |
|---|---:|---:|---:|
| Full LAM-JEPA | `0.2549152493` | `0.0129968006` | 5 |
| `no_planner` | `0.2501694888` | `0.0129968006` | 5 |
| `no_target` | `0.2616949081` | `0.0203953938` | 5 |
| Shuffled-label control | `0.2630508393` | `0.0145011803` | 5 |

These exactly reproduce the retained float32 per-seed-metric aggregation used by Table 1.

## Independently regenerated mechanism effects

| Effect | Mean | Sample SD | Frozen bootstrap 95% CI | Criterion |
|---|---:|---:|---:|---|
| Full − `no_planner` | `+0.0047457606` | `0.0106118432` | `[0.0000000000, 0.0142372817]` | NOT MET |
| Full − `no_target` | `-0.0067796588` | `0.0092834301` | `[-0.0135593176, 0.0000000000]` | NOT MET |

The bootstrap intervals were regenerated from the five seed-level deltas, not copied from the stored aggregate fields.

## Independently regenerated capacity-matched values

Directly recounting correct predictions from the retained per-example rows gives:

| System / effect | Mean | Sample SD |
|---|---:|---:|
| LAM-JEPA | `0.2549152542` | `0.0129968064` |
| Capacity-matched supervised | `0.2664406780` | `0.0154600058` |
| Paired LAM − matched | `-0.0115254237` | `0.0140994131` |

These reproduce the canonical Table 3 values.

## Independently regenerated pinned-pretrained characterization

The retained two-seed development smoke recomputes to:

- LAM-JEPA: `0.15625`
- pinned DeBERTa-v3-xsmall: `0.21875`
- paired delta: `-0.06250`

This remains a tiny development characterization only; no final/capacity-matched/confirmatory inference is added.

## Float32 lineage clarification

A useful numerical-provenance issue surfaced during independent recomputation. Exact per-example correct-count ratios and the retained float32 accuracy metrics differ by at most:

`1.4749623966636705e-08`

That is expected low-level numeric representation drift and changes no pass/fail gate or scientific conclusion. It does, however, affect the last printed decimals if a generator silently mixes the two bases.

Accordingly, `scripts/analysis/generate_arc_negative_paper_assets.py` on this branch was changed to:

- re-aggregate per-seed records rather than trust stored aggregate summaries;
- verify each float32 metric against direct per-example correct counts;
- independently recompute the paired bootstrap intervals;
- use exact per-example prediction counts for the capacity-matched and pretrained tables;
- record the numeric basis explicitly in the generated CSV/manifest;
- fail if prediction-vs-metric drift exceeds `2e-8`;
- keep the locked-test boundary explicit.

## Independently generated local output hashes

The separate verifier produced:

| Output | SHA256 |
|---|---|
| Markdown evidence table | `cc6976b3a89f02f113b0f643046121270edeec8ccf6a57c2205c5227eaee58bc` |
| Validation CSV | `831868ede32c8f465e9423eb3d34eac0c8e1a9ae8d8ce4efbe410e09a5eb7c58` |
| Validation SVG | `1c36d05a48c104415d9be1e4109cb7f0f777320edde9a2cb5a7a20ef249acc64` |
| Verification JSON | `2d3f49be6610fcfead6745828d09013bd88cba4c3ac5649ca1d2a5086f1e2981` |

These hashes identify this verifier run only. They are not substituted for the canonical raw-artifact hashes above.

## Verdict

**`GREEN — INTERNAL QUANTITATIVE RECOMPUTATION`** for the frozen ARC paper numbers and mechanism intervals.

This does **not** imply:

- external independent reproduction;
- paper acceptance or publication;
- LAM-JEPA superiority;
- planner/EMA-target benefit;
- quantization benefit;
- a family-level JEPA conclusion;
- permission to access the locked ARC confirmatory test.

Remaining LAM paper gates are release metadata/ownership decisions and external skeptical review/reproduction.
