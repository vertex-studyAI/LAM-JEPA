# LAM-JEPA Research Status

**Evidence cutoff:** 13 August 2026, attempt-4 audit  
**Frozen scientific head:** `760aa7f9a73a177d5ff4ba7eb470f7e68ace63cb`  
**Seed-order reproducibility repair:** `b72a97a99769b278eb8ec75bc5eab62dc9599f29`  
**Classification:** `RESEARCH_ACTIVE / FROZEN_ARC_RESULT_REPRODUCED / ARC_SUPERIORITY_HYPOTHESIS_UNSUPPORTED / MECHANISM_HYPOTHESES_UNSUPPORTED / RESEARCH_COMPLETE_FALSE`

## Executive result

LAM-JEPA has a reproducible ARC-Challenge research pipeline with checksum-addressed train/validation acquisition, capacity-matched and pretrained comparison paths, five-seed validation, ablations, negative controls, retained raw evidence, and independent verification.

The scientific outcome is **negative / inconclusive**, not a superiority result. The current evidence does **not** support claims that LAM-JEPA outperforms the capacity-matched supervised baseline on ARC, that the planner or target mechanism provides a validated ARC benefit, or that the repaired quantizer provides a validated generalization/quantization advantage.

The locked ARC confirmatory test remains unused and must not be opened to rescue this failed hypothesis line.

## 13 August 2026 full-controls reproducibility audit

Historical workflow run `31203337502` was retained through run attempt 4 without changing the frozen scientific protocol.

Attempt 4:

- job: `94302727334`;
- artifact: `9163503934`;
- artifact digest: `sha256:14c315cd64b2b96d48af4b865bca700a101ea66842a78f35382a5f408805b10a`;
- workflow window: `2026-08-13T00:10:11Z` → `2026-08-13T00:12:47Z`;
- environment: Ubuntu 24.04.4 LTS, Python 3.11.15, PyTorch 2.13.0+cpu, NumPy 2.4.6, CPU;
- verifier: `PROTOCOL_V3_FULL_CONTROLS_VALIDATION_VERIFIED`;
- full five-seed/20-epoch budget: verified;
- locked test evaluated: `false`;
- mechanism claim authorized: `false`;
- research complete: `false`.

An independent parser recomputed the stored five-seed means and sample SDs directly from the raw artifact and matched all stored summaries exactly.

### Attempt-3 → attempt-4 reproducibility

The retained artifacts have the same 10-file set. Eight files are byte-identical. The two raw/normalized result JSONs differ only numerically:

- numeric leaf differences: `36,468`;
- non-numeric leaf differences: `0`;
- maximum observed numeric drift: `0.0007445961236953735`;
- aggregate model, ablation, paired-effect and negative-control summaries: exactly equal;
- strict verifier report, verification JSON and verifier output: byte-identical.

Therefore the defensible claim is exact reproduction of the **aggregate scientific conclusion and verifier decision**, not byte-exact reproduction of every floating-point probability.

### Checkout provenance clarification

Attempt 4 is a rerun of a historical `pull_request` workflow run. The literal checkout SHA was `ed81a16c5b2e3379eb37c4d94a79941d0cd0ff10` (`refs/remotes/pull/48/merge`), while the scientific head is `760aa7f9a73a177d5ff4ba7eb470f7e68ace63cb`.

GitHub comparison between those revisions reports zero changed files. The experiment tree is therefore equivalent for attempt 4, but both SHAs are retained in provenance instead of collapsing them.

Full independent audit:

- `experiments/repro_wave_2026_08_13/INDEPENDENT_AUDIT.md`
- `experiments/repro_wave_2026_08_13/independent_audit.json`

## Frozen ARC scientific result

| System / control | Validation accuracy, mean ± sample SD | n | Outcome |
|---|---:|---:|---|
| Full LAM-JEPA | 0.2549152493 ± 0.0129968006 | 5 | proposed model |
| Capacity-matched supervised | 0.2664406780 ± 0.0154600058 | 5 | stronger mean |
| `no_planner` | 0.2501694888 ± 0.0129968006 | 5 | planner criterion not met |
| `no_target` | 0.2616949081 ± 0.0203953938 | 5 | target criterion not met |
| Shuffled-label control | 0.2630508393 ± 0.0145011803 | 5 | passes frozen `< 0.35` control gate |

Paired mechanism effects:

- full minus `no_planner`: `+0.0047457606 ± 0.0106118432`, bootstrap 95% CI `[0.0, 0.0142372817]`;
- full minus `no_target`: `-0.0067796588 ± 0.0092834301`, bootstrap 95% CI `[-0.0135593176, 0.0]`.

No statistical-significance claim is made.

## Seed-order reproducibility defect

A separate software defect was found in `train_single.py`: model initialization occurred before the requested seed was applied. Pre-fix one-step replay produced differing losses (`10.853294372558594` and `10.34877872467041`). That invalidated deterministic-replay evidence is retained.

PR #61 applied the narrow repair without changing ARC data, scientific seeds, thresholds, metrics, architecture, or locked-test policy. Machine-readable replay metadata records six independently verified attempts. Final loss `11.704492568969727` and final accuracy `0.0` remain exact across attempts, while some secondary floating-point values and serialized checkpoint bytes are not exact across independent runners.

The software repair does not rescue the negative ARC scientific result.

## ARC-v5 repaired line

The train-only repair `arc-v5-stable-ema-residual-0.03125` restored its bounded trainability gate. Repaired validation remained `VALID_NEGATIVE_OR_INCONCLUSIVE_VALIDATION`; the generalization and quantization-benefit gates were not supported.

## Supported claims

The repository can defensibly state that:

1. the documented training/checkpoint/evaluation pipeline executes reproducibly;
2. ARC-Challenge train/validation plumbing and evidence retention are implemented;
3. capacity-matched and pinned pretrained comparison paths are implemented;
4. frozen five-seed ARC validation and required controls were executed;
5. the full-controls aggregate result and verifier verdict reproduce across retained independent attempts;
6. a separate independent parser reproduced attempt-4 seed aggregates exactly;
7. negative results were retained rather than tuned away;
8. the bounded v5 repair restored its train-only gate but repaired validation remained negative/inconclusive.

## Claims not supported

Do not claim:

- LAM-JEPA superiority on ARC;
- planner benefit on ARC;
- target/EMA mechanism benefit on ARC;
- quantization benefit from repaired validation;
- externally validated educational effectiveness;
- general benchmark superiority;
- AGI or general intelligence capability;
- `RESEARCH_COMPLETE`.

## Remaining publication/release gates

- Preserve raw artifacts and their digests; do not rewrite the stale raw claim-boundary field after observation.
- Keep the locked ARC test closed for this failed hypothesis line.
- Treat new architectures or hypotheses as new versioned protocols frozen before validation.
- Repository-level publication packaging remains distinct from the scientific result. Issue #14 tracks owner-approved licensing, citation metadata, provenance and release packaging; author identity or licensing must not be invented.

## Scientific stop rule

Do not tune the current architecture or thresholds against the locked ARC test. Any future architectural repair, new benchmark, or new scientific hypothesis must receive a new versioned protocol before its validation evidence is observed.

Negative results are first-class research artifacts and should remain visible in manuscripts, technical reports, and portfolio summaries.
