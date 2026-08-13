# LAM-JEPA Independent Reproducibility Audit — 13 August 2026

## Scope

This audit independently checks the retained frozen ARC-v3 five-seed full-controls evidence without changing the scientific protocol, data split, seeds, thresholds, architecture, or locked-test policy after observing results.

- audited repository head: `6c6f5c10e8610239ce6c72a4fa7f549659662014`
- frozen scientific revision: `760aa7f9a73a177d5ff4ba7eb470f7e68ace63cb`
- seed-order software repair: `b72a97a99769b278eb8ec75bc5eab62dc9599f29`
- audit environment: Python 3.13.5, Linux 6.18.35 x86_64
- audit method: download the two retained authenticated GitHub Actions artifacts, verify archive hashes, expand both archives, compare retained files, parse raw JSON independently, and inspect the retained verifier report.

A fresh repository clone/training execution could not be performed in the audit sandbox because outbound DNS resolution for `github.com` was unavailable. This is an audit-environment limitation, not evidence of a repository failure. No scientific setting was altered to compensate.

## Artifact integrity

| Attempt | Run / attempt | Job | Artifact | Advertised SHA-256 | Downloaded ZIP SHA-256 | Files |
|---|---|---|---|---|---|---:|
| 2 | `31203337502` / `2` | `94178988063` | `9149336081` | `c45710b5dae6a767ccb6bab7f6e3d8e9578752d8cf9b79fd82a65ae824dded1b` | exact match | 10 |
| 3 | `31203337502` / `3` | `94291056903` | `9162165932` | `caa898f1ff046a337db9b5ddbffe1b332943a732868e2fd809abeda8ee89c30b` | exact match | 10 |

The downloaded ZIP hashes exactly match the digests recorded in the scientific ledger.

Both archives contain the same ten file paths. Eight files are byte-identical, including the train and validation parquet files, data-download manifest, human-readable benchmark output, protocol verification report, strict verifier report, verifier summary, and verifier console output.

Only these two files differ bytewise between attempts:

1. `arc-protocol-v3-full-controls-validation.json`
2. `arc-protocol-v3-full-controls-validation-verification-normalized-input.json`

Independent recursive comparison of the raw result trees found:

- numeric leaf differences: `35,526`
- non-numeric leaf differences: `0`
- maximum absolute numeric drift: `0.0005918592214584351`
- one maximum-drift location: `/negative_control/records/1/predictions/97/probabilities/0`

The normalized-input copies show the same numeric-difference count, zero non-numeric differences, and the same maximum drift. This independently reproduces the repository's low-order floating-point-drift characterization.

## Frozen protocol checks

The retained verifier reports:

- verdict: `PROTOCOL_V3_FULL_CONTROLS_VALIDATION_VERIFIED`
- train source rows: `1,119`
- train eligible rows: `1,117`
- train used rows: `1,117`
- validation source rows: `299`
- validation eligible rows: `295`
- validation used rows: `295`
- seeds: `[1, 2, 3, 4, 5]`
- epochs: `20`
- batch size: `32`
- locked test evaluated: `false`
- final five-seed / 20-epoch protocol executed: `true`
- mechanism claim authorized: `false`
- research complete: `false`
- raw results preserved: `true`
- aggregate normalization only: `true`
- aggregate tolerance: `1e-6`
- maximum observed aggregate drift: `1.075914349280005e-08`

The locked ARC test boundary therefore remains intact for this failed hypothesis line.

## Independently parsed scientific results

| System / control | Mean validation accuracy | Sample SD | n |
|---|---:|---:|---:|
| Full LAM-JEPA | `0.2549152542372881` | `0.01299680644927512` | 5 |
| `no_planner` | `0.2501694915254237` | `0.01299680644927512` | 5 |
| `no_target` | `0.26169491525423727` | `0.02039540197490349` | 5 |
| Shuffled-label negative control | `0.2630508474576271` | `0.014501186194038939` | 5 |

Paired mechanism effects:

- full − `no_planner`: mean `+0.004745762711864404`, sample SD `0.010611848028812555`, bootstrap 95% CI `[0.0, 0.014237288135593213]`, criterion not met.
- full − `no_target`: mean `-0.006779661016949157`, sample SD `0.009283433178053668`, bootstrap 95% CI `[-0.013559322033898313, 0.0]`, criterion not met.

The shuffled-label control remains below the frozen `0.35` failure ceiling.

## Reviewer-facing conclusion

**Classification: `REPRODUCED_NEGATIVE_OR_INCONCLUSIVE`.**

The retained five-seed ARC-v3 full-controls result is internally traceable and survives cross-attempt artifact integrity checks. The independent audit supports the repository's claim that aggregate scientific conclusions and verifier decisions reproduce despite low-order floating-point probability drift.

It does **not** support LAM-JEPA superiority on ARC, planner benefit, target/EMA-path benefit, research completion, or use of the locked test. The negative/inconclusive outcome is a legitimate reproduced research result and should remain visible.

## Limitation

This audit did not generate a seventh scientific training sample. It independently verified two previously retained authenticated Actions artifacts. A new full training rerun should only be counted as additional execution evidence when it runs the unchanged frozen workflow at the frozen scientific revision and retains its own command, environment, logs, raw metrics, verifier output, and artifact digest.
