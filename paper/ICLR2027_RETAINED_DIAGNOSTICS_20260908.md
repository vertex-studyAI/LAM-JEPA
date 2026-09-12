# LAM-JEPA ICLR 2027 retained diagnostics

**Date:** 2026-09-08  
**Scientific state:** **NEGATIVE / INCONCLUSIVE — UNCHANGED**  
**Scope:** paper-facing evidence disclosure only; no new model run, seed, split, threshold, locked-test access, or successor authorization.

This record closes two quantitative disclosure blockers identified by `ICLR2027_SUBMISSION_READINESS_AUDIT_20260908.md`: seed-level integer counts for the frozen full-controls result, and quantitative description of the 256-ID hash-collision limitation. It also rechecks the retained input-truncation evidence from a current, non-expired audit artifact.

## 1. Exact retained full-controls source rechecked

The current paper provenance identifies GitHub Actions artifact **9162165932** from run **31203337502**, digest:

`sha256:caa898f1ff046a337db9b5ddbffe1b332943a732868e2fd809abeda8ee89c30b`

The archive was downloaded again on 2026-09-08. Its archive SHA-256 re-matched that retained digest exactly. The raw result file inside the archive is:

`arc-protocol-v3-full-controls-validation.json`

with SHA-256:

`76aad8b1327e21470aeed137bac341b75b4fcf1f37e5394047642d395e8070f8`

The file itself records protocol `lam-jepa-arc-challenge-v3`, seeds `[1,2,3,4,5]`, 20 epochs, batch size 32, learning rate `0.0003`, one model step, 1,117 eligible train rows, 295 eligible validation rows, and a policy stating that the test split was not downloaded/evaluated.

## 2. Seed-level integer counts from the raw retained predictions

Counts below were recomputed directly from each record's retained `predictions` array as `sum(prediction == label)`. Prediction support is the number of distinct predicted classes across all 295 validation rows in that seed.

| Condition | Seed 1 | Seed 2 | Seed 3 | Seed 4 | Seed 5 | Prediction support per seed |
|---|---:|---:|---:|---:|---:|---|
| `full` | 71/295 | 78/295 | 78/295 | 71/295 | 78/295 | 1, 1, 1, 1, 1 |
| `no_planner` | 71/295 | 78/295 | 71/295 | 71/295 | 78/295 | 1, 1, 1, 1, 1 |
| `no_target` | 71/295 | 78/295 | 83/295 | 71/295 | 83/295 | 1, 1, 1, 1, 1 |
| shuffled-label control | 78/295 | 71/295 | 83/295 | 78/295 | 78/295 | 1, 1, 1, 1, 1 |

The corresponding frozen mechanism integer deltas are therefore:

- `full - no_planner`: `[0, 0, +7, 0, 0]` correct items;
- `full - no_target`: `[0, 0, -5, 0, -5]` correct items.

These counts do **not** create a new hypothesis test. They clarify the already-retained bootstrap summaries and make explicit that most seed-level mechanism differences are zero because the compared runs are constant classifiers selecting the same class.

The validation label base counts reported by the external review are `63, 71, 78, 83` for classes 0–3, consistent with the retained constant-classifier counts above. The mechanism result remains unsupported under the frozen protocol.

## 3. Quantized-path diagnosis boundary

The existing external-review correction remains the authoritative bounded interpretation: the reviewed runs had 295/295 distinct pre-quantizer latents, used one of 32 VQ codes per run, produced effectively constant post-quantizer representations and downstream predictions, and a quantizer-off diagnostic restored input dependence without establishing above-chance ARC performance.

Therefore the paper may report a reproducible failure-mechanism case study for this tested implementation. It may not claim that vector quantization is generally harmful, that removing VQ solves ARC, that JEPA methods fail generally, or that the architecture is superior.

## 4. Hash-collision diagnostic retained from the original external review

The original external-review note quantified the deterministic whitespace-token hashing limitation over the reviewed eligible ARC text surface as:

- **7,030 distinct word types**;
- mapped into **256 hash buckets**;
- mean **27.5 distinct word types per bucket**;
- maximum **47 distinct word types in one bucket**.

This is an **external-review diagnostic**, not a newly generated model outcome. It may be cited descriptively as an upstream representation limitation, but it must not replace the measured VQ bottleneck as the primary localized information-loss finding.

## 5. Input-truncation evidence independently rechecked

The current non-expired `ARC Canonical Input Visibility Audit` artifact **9884278396** from workflow run **33732128865** has retained archive digest:

`sha256:eaf1b43a781d52b74bcf14b23e36acec27e8f4870dd182b57c23badf5d9df8c3`

The archive was downloaded again on 2026-09-08 and re-hashed to the same digest. Its `input-visibility-audit.json` has SHA-256:

`1d894781e4476505adb1c6464cea0cf232f1dfa13cb4e69f0e0b2d1c313c5f7a`

Recomputation from its retained row records gives:

- train: **23 / 1,117** eligible prompts exceed the 96-whitespace-token limit;
- validation: **6 / 295** eligible prompts exceed the limit;
- validation: **3 / 295** rows have the fourth-answer marker `[3]` beyond the retained 96-token surface;
- test split accessed: **false**.

The same audit records 1,116 unique encoded token sequences across 1,117 eligible train examples and 295/295 unique encoded token sequences on validation. This is a representation/input-visibility diagnostic only and does not alter the negative result.

## 6. Manuscript disclosure consequence

The next manuscript revision can now safely state, without inventing numbers:

1. the exact five-seed integer correct counts for `full`, `no_planner`, `no_target`, and shuffled control;
2. that the five-seed bootstrap is a finite-seed summary and not an asymptotic/population uncertainty guarantee;
3. the exact truncation counts above;
4. the external-review hash-collision diagnostic above;
5. the exact gradient-active accounting definition already identified in `scripts/ci/measure_arc_gradient_capacity.py`;
6. **AI2 Reasoning Challenge (ARC-Challenge)** at first use.

No scientific rescue is authorized. The locked historical confirmatory ARC test remains closed, and the separate successor study remains separate.