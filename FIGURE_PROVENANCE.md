# LAM-JEPA Figure and Table Provenance

**Date:** 2026-08-14  
**Principle:** publication graphics are views of retained experiment artifacts, never hand-entered evidence.

## Figure 1 — Frozen planner/target mechanism effects

**Question:** did either required mechanism satisfy its frozen contribution criterion?

**Source artifact:** `ci-evidence/arc-protocol-v3-full-controls-validation.json` from the frozen five-seed ARC-v3 scientific workflow. Retained independent rerun attempt 3 is GitHub Actions run `31203337502`, attempt `3`, job `94291056903`, artifact `9162165932`, artifact digest `sha256:caa898f1ff046a337db9b5ddbffe1b332943a732868e2fd809abeda8ee89c30b`.

**Fields consumed:**

- `paired_effects.no_planner.mean_full_minus_ablation`
- `paired_effects.no_planner.paired_bootstrap_ci95_low/high`
- `paired_effects.no_target.mean_full_minus_ablation`
- `paired_effects.no_target.paired_bootstrap_ci95_low/high`

**Expected scientific interpretation:** neither effect satisfies the predeclared mechanism gate. The graphic is descriptive and is not a significance claim.

**Generator:** `scripts/paper/generate_arc_negative_artifacts.py`

**Generation command after the retained controls and matched JSON files are materialized:**

```bash
python scripts/paper/generate_arc_negative_artifacts.py \
  --controls-json ci-evidence/arc-protocol-v3-full-controls-validation.json \
  --matched-json <RETAINED_FIVE_SEED_MATCHED_BASELINE_JSON> \
  --out-dir papers/generated/arc-negative
```

**Outputs:**

- `arc_mechanism_effects.csv` — plotted source data;
- `arc_mechanism_effects.svg` — publication figure;
- `arc_paper_artifact_manifest.json` — hashes of both input artifacts and generated outputs.

The generator refuses non-five-seed/non-20-epoch inputs and refuses controls that do not identify protocol `lam-jepa-arc-challenge-v3`.

## Table 1 — Primary frozen ARC validation results

**Source artifacts:**

1. the same frozen controls JSON for full/no-planner/no-target/shuffled-label values;
2. the separately retained five-seed gradient-active-parameter-matched comparison JSON for the matched supervised result.

The matched-baseline value is already preserved in `RESULTS.md` and `RESEARCH_STATUS.md`, but the final table should be generated from the raw matched JSON rather than copied from manuscript prose. The exact retained matched JSON artifact/path must be materialized and its digest recorded before the generated table is accepted into a release bundle.

**Generator output:** `papers/generated/arc-negative/arc_primary_results.md`.

**Important boundary:** the generator deliberately keeps the LAM-JEPA value from the matched-baseline run separate from the LAM-JEPA value from the controls run, even though they are numerically near-identical. It does not average across distinct artifacts.

## Figure 2 — Reproducibility drift (optional)

Only include if it materially helps the paper.

**Source:** retained attempt-2 vs attempt-3 artifact comparison documented in `RESULTS.md` / `REPRODUCE.md`.

**Permitted content:**

- number of numeric leaves that differ (`35,526`);
- maximum observed numeric drift (~`5.9186e-4`);
- exact equality of aggregate scientific summaries/verifier outputs.

**Do not imply:** byte-exact reproducibility, or that a plotted distribution has been reconstructed if the pairwise raw-difference vector is not retained. If only the summary statistics are available, report them in text/table form rather than fabricating a histogram.

## Figure 3 — Mechanism diagram (optional, non-evidentiary)

A source-locked architecture diagram may be drawn from scientific commit `760aa7f9a73a177d5ff4ba7eb470f7e68ace63cb`, but it must visually distinguish:

- hashed-token/numeric encoder;
- projector;
- VQ path;
- learned memory retrieval/correction;
- one-step latent-action transition;
- EMA target path;
- latent-summary + four-choice ARC head;
- ARC-specific loss terms.

The diagram must **not** depict the frozen token encoder as a Transformer, must not label latent action IDs as semantic reasoning operations, and must not present confidence/verifier/rubric heads as ARC-supervised evidence because those heads are outside the frozen ARC objective.

## Figure acceptance checklist

Every final figure/table must ship with:

- raw source artifact path;
- source artifact SHA-256;
- scientific code commit;
- generator code commit;
- exact generation command;
- generated file SHA-256;
- caption;
- one-sentence interpretation;
- one-sentence limitation.

A figure with no provenance record is not manuscript-ready.
