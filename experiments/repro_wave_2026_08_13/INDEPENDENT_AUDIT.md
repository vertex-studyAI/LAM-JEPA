# LAM-JEPA Independent Reproducibility Audit — 13 August 2026

## Scope

This audit inspects retained GitHub Actions evidence for the frozen ARC Protocol v3 full-controls validation. It does **not** change the scientific protocol, model, data, seeds, thresholds, raw artifacts, or locked-test policy.

## Attempt 4 provenance

- scientific head SHA: `760aa7f9a73a177d5ff4ba7eb470f7e68ace63cb`
- workflow run: `31203337502`
- run attempt: `4`
- job: `94302727334`
- artifact: `9163503934`
- artifact digest: `sha256:14c315cd64b2b96d48af4b865bca700a101ea66842a78f35382a5f408805b10a`
- workflow window: `2026-08-13T00:10:11Z` → `2026-08-13T00:12:47Z` (156 s wall clock)
- environment: Ubuntu 24.04.4 LTS, Python 3.11.15, PyTorch 2.13.0+cpu, NumPy 2.4.6, CPU

The historical workflow run is a `pull_request` run. `actions/checkout` therefore resolved `refs/remotes/pull/48/merge` at checkout SHA `ed81a16c5b2e3379eb37c4d94a79941d0cd0ff10`, described in the log as a merge of scientific head `760aa7f9a73a177d5ff4ba7eb470f7e68ace63cb` into base `4de9c4298dd66fd70af883c52aab30cf663cda30`.

A GitHub compare from `760aa7f9a73a177d5ff4ba7eb470f7e68ace63cb` to `ed81a16c5b2e3379eb37c4d94a79941d0cd0ff10` reports one merge commit ahead and **zero changed files**. This is therefore recorded as a non-invalidating provenance clarification: the checked-out experiment tree is equivalent to the scientific head, but the literal checkout SHA should be retained in the evidence trail.

## Frozen protocol verified

| Field | Value |
|---|---:|
| Seeds | 1, 2, 3, 4, 5 |
| Epochs | 20 |
| Batch size | 32 |
| Learning rate | 0.0003 |
| Model steps | 1 |
| Eligible train rows | 1,117 |
| Eligible validation rows | 295 |
| Device | CPU |
| Locked test evaluated | No |

The independent repository verifier returned `PROTOCOL_V3_FULL_CONTROLS_VALIDATION_VERIFIED`, with `final_five_seed_20_epoch_protocol_executed=true`, `mechanism_claim_authorized=false`, and `research_complete=false`.

## Independent raw-metric recomputation

The downloaded attempt-4 artifact contains 10 files. The raw JSON was parsed independently and the accuracy mean and sample standard deviation were recomputed directly from the five seed-level records.

| System | Seed accuracies | Mean | Sample SD | Stored summary reproduced? |
|---|---|---:|---:|---|
| Full LAM-JEPA | 0.2406779677, 0.2644067705, 0.2644067705, 0.2406779677, 0.2644067705 | 0.2549152493 | 0.0129968006 | Yes, exactly |
| `no_planner` | 0.2406779677, 0.2644067705, 0.2406779677, 0.2406779677, 0.2644067705 | 0.2501694888 | 0.0129968006 | Yes, exactly |
| `no_target` | 0.2406779677, 0.2644067705, 0.2813559175, 0.2406779677, 0.2813559175 | 0.2616949081 | 0.0203953938 | Yes, exactly |
| Shuffled-label control | 0.2644067705, 0.2406779677, 0.2813559175, 0.2644067705, 0.2644067705 | 0.2630508393 | 0.0145011803 | Yes, exactly |

The negative control remains below its frozen `0.35` ceiling. Neither planner nor target-path mechanism criterion is authorized by the verifier.

## Attempt 3 → attempt 4 comparison

Attempt 3 artifact `9162165932` and attempt 4 artifact `9163503934` have the same 10-file set.

- 8 / 10 files are byte-identical.
- The two differing files are the raw full-results JSON and its normalized-input copy.
- Numeric leaf differences: `36,468`.
- Non-numeric leaf differences: `0`.
- Maximum observed numeric drift: `0.0007445961236953735` at a negative-control per-example probability.
- Stored aggregate accuracy summaries: exactly equal.
- Stored paired-effect summaries: exactly equal.
- Stored negative-control summary: exactly equal.
- Strict verifier report: byte-identical.
- Verification JSON: byte-identical.
- Verifier console output: byte-identical.

This supports the same bounded conclusion as the earlier reruns: the **aggregate scientific result and verifier decision reproduce exactly**, while per-example floating-point probabilities are not byte-exact across independent runners.

## Attempt-4 file hashes

| File | SHA-256 |
|---|---|
| `arc-data/arc-challenge-validation.parquet` | `395a5c88d1580d69855fbaee9450270578df1ad5af6259771cd0a42c20e99f05` |
| `arc-data/arc-challenge-train.parquet` | `e488c1587ffdcfc8443f916c53488a95cd471c5790e0746c6bfe4cecf20962cb` |
| `arc-download-full-controls-v3.json` | `f538ece60fc86a69d33d3a43cf9a4cc3b12f1ab32f503309645aa3e30e5d6656` |
| `arc-protocol-v3-full-controls-validation-output.txt` | `b2294ab339fbf6b11586c4cd56c00fabdb5fabbb3e90a5d4b44089857852dcb0` |
| `arc-protocol-v3-full-controls-validation-verification-normalized-input.json` | `384cd542e7aceead8553d0e99dc1965e3d250d6768a44c8d3f00479c29612cd5` |
| `arc-protocol-v3-full-controls-validation-verification-strict-report.json` | `b21c992c773ec58a390cb1bddf848c25b446b988ae6bdcd5a70c78ff13e9de5a` |
| `arc-protocol-v3-full-controls-validation-verification.json` | `10704c164a885169832175a68fba4e501b5370dada5ebf3b87f6c3c94a7031a2` |
| `arc-protocol-v3-full-controls-validation-verifier-output.txt` | `a43b3d1f911e3649d29f292a72d98e9ea9a1d6e8c701c2c0d546d5dce2703107` |
| `arc-protocol-v3-full-controls-validation.json` | `f341f301ba5b9ff90afe433d2e57bbea710b262b2872acf6fa22324daba31a34` |
| `arc-protocol-v3-verification.json` | `00a06ec6efb5d1b25d45f0afac8847ec3b91ffaa26fbce20e76d053e79a002b6` |

## Scientific disposition

**NEGATIVE / INCONCLUSIVE, REPRODUCED.** The frozen evidence does not support LAM-JEPA superiority over the capacity-matched supervised baseline, nor a validated planner, target-path, or quantization benefit. The locked ARC test remains unused and must not be opened to rescue this hypothesis line.

The stale raw `protocol.claim_boundary` sentence remains a reporting-metadata defect and is deliberately left untouched in the retained artifact.
