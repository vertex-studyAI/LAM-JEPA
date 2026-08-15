# LAM-JEPA External Validation Packet — 2026-08-14

**Purpose:** give a genuinely independent validator one bounded, immutable reproduction/review task. This file prepares external validation; it does **not** claim external validation has occurred.

## Immutable package identity

- Repository: `vertex-studyAI/LAM-JEPA`
- Paper/reproducibility package revision: `725ae2fb17de9c988938d4b03bd8a6be456b8e8b`
- Frozen ARC-v3 scientific source recorded by the retained protocol/evidence: `760aa7f9a73a177d5ff4ba7eb470f7e68ace63cb`
- Locked ARC test: **must remain unused** for this failed hypothesis line.

Do not validate a moving branch tip. Check out the exact package revision above.

## Claim to validate

> Under the frozen exactly-four-choice ARC-Challenge development-validation protocol, this specific small LAM-JEPA configuration does not outperform its gradient-active-parameter-matched supervised comparator, and its evaluated one-step latent-action-rollout and EMA target-path ablations do not meet the preregistered contribution criteria. The adverse conclusion is reproducible from retained artifacts without accessing the locked ARC test.

## Non-claims

This packet does **not** ask the validator to endorse or test claims that:

- JEPA broadly fails on reasoning;
- Transformers broadly fail/succeed;
- search/planning in general fails;
- latent actions are novel;
- EMA target networks are novel;
- LAM-JEPA is superior;
- the bounded pinned DeBERTa smoke is a compute-matched final baseline;
- educational/tutoring effectiveness is established;
- the work is already publishable, accepted, or externally validated.

## Retained raw evidence

### Full controls

- workflow run: `31203337502`, attempt 3
- artifact ID: `9162165932`
- artifact name: `arc-protocol-v3-full-controls-validation`
- ZIP SHA-256: `caa898f1ff046a337db9b5ddbffe1b332943a732868e2fd809abeda8ee89c30b`
- key raw file: `arc-protocol-v3-full-controls-validation.json`
- raw JSON SHA-256: `76aad8b1327e21470aeed137bac341b75b4fcf1f37e5394047642d395e8070f8`

### Gradient-active-parameter-matched supervised comparator

- workflow run: `31203337225`
- artifact ID: `9003785715`
- artifact name: `arc-protocol-v3-matched-baseline`
- ZIP SHA-256: `13268856e9be2d9da91addc7935a9cd7bdc4bc7b0a59e527905c6de6fa0f87cc`
- key raw file: `matched-v3-full-validation.json`
- raw JSON SHA-256: `7abec096d0875bf60f046f7ec4ec28c9580419f17137cccbb5a3f3311c85a78e`

### Bounded pinned-pretrained characterization

- workflow run: `31203337145`
- artifact ID: `9003740436`
- artifact name: `arc-pretrained-v2-deberta-smoke`
- ZIP SHA-256: `ff63544689d995c162b2eea3850fd06032115485b8007c6ccc5b01f8689c9b8d`
- key raw file: `arc-pretrained-v2-deberta.json`
- raw JSON SHA-256: `d5dd1beb32c7d875fb7d5d83ea4e7a57ba0c609c8e4b6595358a7368ac327e58`

The pretrained artifact is characterization only; its verifier explicitly does not make a superiority claim.

## Internal expected values

A clean independent parse should recover, within displayed precision:

| System | Mean accuracy | Sample SD | n |
|---|---:|---:|---:|
| LAM-JEPA full | `0.2549152493` | `0.0129968006` | 5 |
| No planner | `0.2501694888` | `0.0129968006` | 5 |
| No target | `0.2616949081` | `0.0203953938` | 5 |
| Shuffled labels | `0.2630508393` | `0.0145011803` | 5 |
| Matched supervised | `0.2664406806` | `0.0154600003` | 5 |

Paired effects:

- full − matched supervised: mean `-0.0115254313`, sample SD `0.0140994057`, n=5;
- full − no planner: mean `+0.0047457606`, retained bootstrap 95% interval `[0.0, 0.0142372817]` — frozen mechanism criterion fails;
- full − no target: mean `-0.0067796588`, retained bootstrap 95% interval `[-0.0135593176, 0.0]` — frozen mechanism criterion fails.

Bounded pinned-pretrained characterization:

- LAM mean `0.15625`, n=2;
- pinned pretrained mean `0.21875`, n=2;
- paired LAM − pretrained mean `-0.0625`, n=2.

## Requested independent actions

### A. Artifact integrity

1. Download the three retained artifacts independently from GitHub Actions.
2. Compute SHA-256 of each ZIP and compare with this packet.
3. Extract the key raw JSON and compare its SHA-256.
4. Report any missing/expired artifact rather than substituting a new run silently.

### B. Numerical recomputation

Without copying manuscript numbers:

1. parse the raw JSONs;
2. recompute/report the five primary means/sample SDs and n;
3. recompute/report seed-paired full-minus-matched mean/sample SD;
4. inspect the retained planner/target paired effects and bootstrap intervals;
5. verify the frozen criteria evaluate to FAIL;
6. verify the bounded DeBERTa characterization is exactly two seeds and separately scoped.

### C. Deterministic paper assets

Run the committed generator against the three extracted JSON files:

```bash
python scripts/analysis/generate_arc_negative_paper_assets.py \
  --full-controls /ABS/PATH/arc-protocol-v3-full-controls-validation.json \
  --matched /ABS/PATH/matched-v3-full-validation.json \
  --pretrained /ABS/PATH/arc-pretrained-v2-deberta.json \
  --out-dir /ABS/PATH/lam-paper-assets
```

Internal regenerated reference hashes are:

- `arc_validation_accuracy.csv`: `4c53665775e60832202ced9b143fc649486f6cbc786df399f77c47f367b17356`
- `ARC_NEGATIVE_RESULT_TABLES.generated.md`: `ae9ff3c346b99049789190ea8b4c33f5009fbd5cb186b28bca564dee12ef98b9`
- `arc_validation_accuracy.generated.svg`: `db1304b32a833cd904fe380279d9fa21aa27b5536561297aba75f5f24b3f05d6`

If hashes differ, report the exact runtime, file diff and reason. Do not normalize away a discrepancy without documenting it.

### D. Source/method skepticism

Inspect the exact evaluated ARC source path and answer:

1. Is the configured `no_planner` comparison actually a one-step latent-action-rollout ablation rather than beam/tree search?
2. Does the frozen token path contain a contextual Transformer block, or is it effectively embedding/position + normalization/mean-pooling under the retained source?
3. Does the EMA target path use a separately held-out/future semantic target, or the same serialized ARC input through the target encoder path?
4. Are advertised value/verifier/rubric educational heads causally evaluated by the ARC objective?
5. Does any current evidence justify a family-level claim about JEPA, planning or educational effectiveness?

The validator should disagree explicitly if the canonical manuscript overstates any answer.

### E. Locked-test integrity

Verify the retained protocol/evidence says the ARC test split was not downloaded/evaluated for this development line. Do not access the test split as part of this validation.

## Requested report format

Return one report containing:

### REPRODUCTION VERDICT
`PASS`, `PARTIAL`, or `FAIL`.

### ARTIFACT INTEGRITY
For each artifact: availability, observed ZIP SHA-256, observed raw JSON SHA-256, match/mismatch.

### NUMERICAL AGREEMENT
Observed values, deltas from the packet, and whether any discrepancy changes the scientific conclusion.

### SOURCE/METHOD AGREEMENT
Answers to questions D1–D5 with file/line references.

### CLAIM BOUNDARY
The strongest claim the validator believes the evidence supports, plus any manuscript claim they believe is too broad.

### REPRODUCIBILITY GAPS
Undocumented dependencies, environment sensitivity, missing artifacts, ambiguous commands, or assumptions.

### ACCEPTANCE-THREATENING CRITICISM
The single strongest remaining scientific/experimental/mechanism criticism.

## Public-release metadata boundary — 2026-08-15

These fields are intentionally unresolved rather than inferred from commit history:

- **Current repository head observed for packet maintenance:** `bf8311e1a4d240e2891e51af38eaf7754944e300`.
- **Reproduction checkout remains immutable:** `725ae2fb17de9c988938d4b03bd8a6be456b8e8b`; this metadata note does not change the frozen scientific package.
- **License / redistribution:** `BLOCKED_OWNER`. No repository `LICENSE` file was present at the observed head. Owner must approve the license and third-party redistribution boundary before public release.
- **Authorship / order:** `BLOCKED_OWNER`. Do not infer authorship or author order from Git history, issue authorship, or prior drafts.
- **Citation metadata:** `BLOCKED_OWNER`. `CITATION.cff` was absent at the observed head; create it only after author order and license decisions are approved.
- **External-validation state:** `NONE_RETURNED`. This packet is ready to send, but no genuinely independent reproduction/review report has been returned yet.

## External GREEN rule

This project becomes **externally validated** only after a genuinely independent validator returns evidence from this immutable packet. A sent email, pending review, internal ChatGPT critique, or author rerun is **not** external validation.
