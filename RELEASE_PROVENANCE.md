# LAM-JEPA release provenance and claim boundary

**Evidence cutoff:** 13 August 2026  
**Documentation head before this closure branch:** `6c6f5c10e8610239ce6c72a4fa7f549659662014`  
**Frozen full-controls scientific source:** `760aa7f9a73a177d5ff4ba7eb470f7e68ace63cb`  
**Scientific verdict:** `ARC_SUPERIORITY_AND_MECHANISM_HYPOTHESES_UNSUPPORTED`

This file records repository-verifiable provenance for the negative/inconclusive ARC research package. It does not turn reproducibility into a superiority claim and it does not authorize access to the locked ARC confirmatory test.

## 1. Canonical implementation and evidence surfaces

The installable package is defined by `pyproject.toml` and `src/lam_jepa/`. Frozen research protocols live under `protocols/`; executable research and verification entry points live under `scripts/`; tests live under `tests/`; retained machine-readable metadata for the 12–13 August reproducibility wave lives at:

`experiments/repro_wave_2026_08_12/experiment_metadata.json`.

The human-readable reproduction contract is `REPRODUCE.md`.

## 2. External benchmark boundary

The evaluated dataset is AI2 ARC-Challenge using checksum-addressed train and validation data only. The frozen eligibility rule keeps rows with exactly four answer choices:

- train: 1,117 / 1,119 eligible;
- validation: 295 / 299 eligible;
- locked ARC test evaluated: **false**.

The confirmatory test must remain unopened for the failed superiority/mechanism hypothesis line.

## 3. Frozen full-controls scientific protocol

The full scientific comparison uses:

- seeds: `[1, 2, 3, 4, 5]`;
- epochs: `20`;
- batch size: `32`;
- learning rate: `0.0003`;
- model steps: `1`;
- device class: CPU;
- all eligible train/validation rows;
- full LAM-JEPA, `no_planner`, `no_target`, deterministic shuffled-label control, and separately retained capacity-matched/pretrained comparator evidence.

The source revision for this frozen full-controls result is `760aa7f9a73a177d5ff4ba7eb470f7e68ace63cb`.

## 4. Independent full scientific reruns

The frozen full-controls workflow was independently rerun without retuning the model, data, seeds, thresholds, or decision rules.

### Attempt 2

- workflow run: `31203337502`;
- job: `94178988063`;
- artifact: `9149336081`;
- digest: `sha256:c45710b5dae6a767ccb6bab7f6e3d8e9578752d8cf9b79fd82a65ae824dded1b`;
- conclusion: success.

### Attempt 3

- workflow run: `31203337502`;
- job: `94291056903`;
- artifact: `9162165932`;
- digest: `sha256:caa898f1ff046a337db9b5ddbffe1b332943a732868e2fd809abeda8ee89c30b`;
- conclusion: success.

Each artifact retains 10 files. Eight files are byte-identical. Raw result JSON and normalized-input JSON are not byte-identical because low-order per-example floating-point values drift across runners. The maximum observed numeric drift is approximately `5.9186e-4`; no non-numeric leaf changed.

The following scientific summaries reproduce **exactly** across the two independent attempts:

- full validation aggregate;
- `no_planner` aggregate;
- `no_target` aggregate;
- paired mechanism effects;
- shuffled-label summary;
- verifier summary;
- strict verifier verdict.

Therefore the defensible wording is **semantic/scientific aggregate reproducibility**, not byte-for-byte numerical identity.

## 5. Frozen result

The retained full-controls summary is:

| Condition | Mean validation accuracy | n |
|---|---:|---:|
| Full LAM-JEPA | 0.2549152493 | 5 |
| `no_planner` | 0.2501694888 | 5 |
| `no_target` | 0.2616949081 | 5 |
| Shuffled-label control | 0.2630508393 | 5 |

Paired effects:

- full − `no_planner`: `+0.0047457606`, bootstrap 95% CI `[0.0, 0.0142372817]`, criterion **not met**;
- full − `no_target`: `-0.0067796588`, bootstrap 95% CI `[-0.0135593176, 0.0]`, criterion **not met**.

The separately retained capacity-matched supervised baseline has higher mean validation accuracy. No LAM-JEPA superiority, planner-benefit, target-benefit, or repaired-quantization-benefit claim is authorized.

## 6. Deterministic training replay provenance

A reproducibility defect was found because `train_single.py` instantiated `LAMJEPA` before applying the requested seed. The pre-fix evidence is retained as invalidated reproducibility evidence rather than deleted.

The narrow seed-order repair merged at `b72a97a99769b278eb8ec75bc5eab62dc9599f29` (PR #61) without changing the scientific protocol.

Independent deterministic replay metadata records six verified attempts. Within each attempt, model state, metrics, and RNG state are exact. Across attempts, final loss `11.704492568969727` and final accuracy `0.0` are exact, while secondary floats and serialized checkpoint bytes are not guaranteed identical.

Do not claim byte-identical checkpoints across independent runners.

## 7. Preserved reporting defect

The frozen raw full-controls payload contains a stale sentence saying the invocation is not the final five-seed/20-epoch protocol. The actual command, frozen protocol fields, and independent verifier establish that the final five-seed/20-epoch budget was executed.

Classification: `NON_INVALIDATING_REPORTING_METADATA_DEFECT`.

The raw artifact is intentionally not rewritten after the result. The discrepancy is documented rather than erased.

## 8. Reproduction entry points

Use `REPRODUCE.md` for executable commands. The canonical full-controls paths include:

- `scripts/benchmark/run_arc_protocol_v3_controls.py`;
- `scripts/ci/verify_arc_protocol_v3_full_controls.py`;
- `protocols/arc_challenge_v3.json`;
- `.github/workflows/arc-protocol-v3-full-controls-validation.yml`;
- `.github/workflows/arc-protocol-v3-full-controls-reverify.yml`.

The repaired-v5 line remains a separate negative/inconclusive result and must not be conflated with the frozen v3 full-controls result.

## 9. Legal and bibliographic blockers

The following are intentionally unresolved until the repository owner supplies/approves them:

1. root license and third-party compatibility review;
2. `CITATION.cff` author list/order and release metadata;
3. any formal historical code-origin attestation beyond repository history;
4. dataset redistribution/licensing review.

These blockers do not invalidate the negative scientific result, but they block a formal release package from being called publication-complete.

## 10. Release rule

Passing CI and reproducing aggregate results establish repository execution and reproducibility evidence. They do not establish novelty, model superiority, educational effectiveness, production readiness, AGI capability, or `RESEARCH_COMPLETE`.

A release-quality evidence package must link an immutable source revision, frozen protocol, retained raw artifacts and digests, independent verification, reproduction commands, explicit claim ledger, approved bibliographic/legal metadata, and the negative-result boundary above.
