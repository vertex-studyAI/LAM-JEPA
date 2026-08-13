# LAM-JEPA Results Ledger

**Reproducibility wave:** 2026-08-12 to 2026-08-13  
**Frozen scientific source revision:** `760aa7f9a73a177d5ff4ba7eb470f7e68ace63cb`  
**Seed-order reproducibility repair:** `b72a97a99769b278eb8ec75bc5eab62dc9599f29`  
**Scientific status:** **negative / inconclusive, reproduced** on the frozen ARC-Challenge superiority and mechanism hypotheses  
**Confirmatory test status:** **LOCKED**; do not use it to rescue this failed validation hypothesis line.

## Research question

Under the frozen ARC-Challenge train/validation protocol, does LAM-JEPA improve validation accuracy over a gradient-active-parameter-matched supervised baseline, and do the planner, target/EMA path, or repaired quantized latent mechanism contribute a reproducible validation benefit?

## Dataset, protocol, and seed policy

AI2 ARC-Challenge multiple-choice reasoning. Protocol v3 retains exactly four-choice rows, preserves source order, and uses checksum-addressed train and validation data only. The locked ARC test is not downloaded or evaluated for this failed hypothesis line.

Frozen full-controls budget:

- seeds: `1 2 3 4 5`;
- epochs: `20`;
- batch size: `32`;
- learning rate: `0.0003`;
- model steps: `1`;
- eligible train rows: `1117`;
- eligible validation rows: `295`;
- device: CPU.

## Baselines and controls

- capacity-matched supervised baseline using gradient-active parameter matching;
- pinned `microsoft/deberta-v3-xsmall` comparator for bounded development characterization;
- deterministic shuffled-label negative control;
- `no_planner` ablation;
- `no_target` ablation.

## Canonical scientific result

| System / control | Validation accuracy, mean ± sample SD | n | Interpretation |
|---|---:|---:|---|
| Full LAM-JEPA | 0.2549152493 ± 0.0129968006 | 5 | Proposed model |
| Capacity-matched supervised | 0.2664406780 ± 0.0154600058 | 5 | Matched baseline; stronger mean |
| `no_planner` | 0.2501694888 ± 0.0129968006 | 5 | Planner ablation |
| `no_target` | 0.2616949081 ± 0.0203953938 | 5 | Target-path ablation |
| Shuffled-label control | 0.2630508393 ± 0.0145011803 | 5 | Below frozen 0.35 failure threshold |

Paired mechanism effects from the frozen full-controls run:

- full − `no_planner`: mean `+0.0047457606`, sample SD `0.0106118432`, bootstrap 95% CI `[0.0, 0.0142372817]`;
- full − `no_target`: mean `-0.0067796588`, sample SD `0.0092834301`, bootstrap 95% CI `[-0.0135593176, 0.0]`.

Neither predeclared mechanism criterion was met. No statistical-significance claim is made.

The separately retained capacity-matched comparison is adverse to LAM-JEPA (`0.2549152542 ± 0.0129968064` versus `0.2664406780 ± 0.0154600058`; paired LAM minus matched `-0.0115254237 ± 0.0140994131`). A bounded development comparison against the pinned pretrained comparator was also adverse (`0.15625` vs `0.21875`).

## Independent frozen full-controls reruns

Historical workflow run `31203337502` has now been retained through attempt 4. The scientific head is `760aa7f9a73a177d5ff4ba7eb470f7e68ace63cb`.

| Attempt | Job | Artifact | Artifact digest | Verdict |
|---:|---:|---:|---|---|
| 2 | `94178988063` | `9149336081` | `sha256:c45710b5dae6a767ccb6bab7f6e3d8e9578752d8cf9b79fd82a65ae824dded1b` | success |
| 3 | `94291056903` | `9162165932` | `sha256:caa898f1ff046a337db9b5ddbffe1b332943a732868e2fd809abeda8ee89c30b` | success |
| 4 | `94302727334` | `9163503934` | `sha256:14c315cd64b2b96d48af4b865bca700a101ea66842a78f35382a5f408805b10a` | success |

Attempt 4 ran on GitHub-hosted Ubuntu 24.04.4 LTS with Python 3.11.15, PyTorch 2.13.0+cpu and NumPy 2.4.6. The workflow window was `2026-08-13T00:10:11Z` to `2026-08-13T00:12:47Z` (156 seconds wall clock).

The verifier again returned `PROTOCOL_V3_FULL_CONTROLS_VALIDATION_VERIFIED`, with the full five-seed/20-epoch budget executed, the locked test unused, mechanism claims unauthorized, and `research_complete=false`.

## Independent attempt-4 raw checker

The attempt-4 artifact was downloaded and parsed independently of the repository verifier. The five seed-level accuracies were recomputed into their mean and sample SD; all four stored summaries reproduced exactly:

| System | Seed-level accuracies | Recomputed mean ± sample SD |
|---|---|---:|
| Full | 0.2406779677, 0.2644067705, 0.2644067705, 0.2406779677, 0.2644067705 | 0.2549152493 ± 0.0129968006 |
| `no_planner` | 0.2406779677, 0.2644067705, 0.2406779677, 0.2406779677, 0.2644067705 | 0.2501694888 ± 0.0129968006 |
| `no_target` | 0.2406779677, 0.2644067705, 0.2813559175, 0.2406779677, 0.2813559175 | 0.2616949081 ± 0.0203953938 |
| Shuffled-label | 0.2644067705, 0.2406779677, 0.2813559175, 0.2644067705, 0.2644067705 | 0.2630508393 ± 0.0145011803 |

Per-file SHA-256 values and the complete independent audit are retained in:

- `experiments/repro_wave_2026_08_13/INDEPENDENT_AUDIT.md`
- `experiments/repro_wave_2026_08_13/independent_audit.json`

## Attempt 3 → attempt 4 comparison

Both artifacts contain the same 10 retained files.

- 8 files are byte-identical;
- the raw full-results JSON and its normalized-input copy differ;
- numeric leaf differences: `36,468`;
- non-numeric leaf differences: `0`;
- maximum observed numeric drift: `0.0007445961236953735` in a negative-control per-example probability;
- aggregate full / `no_planner` / `no_target` accuracies are exactly equal;
- paired-effect summaries are exactly equal;
- negative-control summary is exactly equal;
- strict verifier report, verification JSON and verifier console output are byte-identical.

The defensible reproducibility claim is therefore exact replication of the frozen aggregate scientific conclusion and verifier decision, with low-order per-example floating-point drift that does not change the aggregate result. Byte-exact raw JSON identity across independent runners is not claimed.

## Checkout provenance clarification

Attempt 4 is a rerun of a historical `pull_request` workflow run. `actions/checkout` resolved `refs/remotes/pull/48/merge` at `ed81a16c5b2e3379eb37c4d94a79941d0cd0ff10`, not the head SHA as a literal checkout.

A GitHub compare from scientific head `760aa7f9a73a177d5ff4ba7eb470f7e68ace63cb` to checkout `ed81a16c5b2e3379eb37c4d94a79941d0cd0ff10` reports **zero changed files**. This is a non-invalidating provenance clarification: retain both SHAs in the evidence trail, but the experiment tree is equivalent to the scientific head for attempt 4.

## Reporting-metadata defect

The frozen runner's raw `protocol.claim_boundary` string says the control script is “not the final five-seed/20-epoch protocol.” That sentence is stale for this invocation. The workflow arguments and independent verifier show seeds 1–5, 20 epochs, all eligible train/validation rows, and `final_five_seed_20_epoch_protocol_executed=true`.

This remains a **non-invalidating reporting-metadata defect**. The raw artifact is preserved unchanged.

## Seed-order reproducibility bug and repair

A separate reproducibility defect was previously found in `train_single.py`: model initialization occurred before the requested seed was applied. Under nominally identical SHA / CLI / seed / CPU execution, one-step losses differed (`10.853294372558594` vs `10.34877872467041`). The pre-fix evidence remains preserved.

PR #61 applied the smallest software repair: seed before `LAMJEPA(cfg)` construction while retaining trainer-side seeding for the subsequent stream. No ARC split, seed set, scientific threshold, metric, architecture, or locked-test policy changed.

Machine-readable replay metadata records six independently verified replay attempts. Within each runner attempt, model state, final metrics and RNG state are exact; across attempts, final loss `11.704492568969727` and final accuracy `0.0` remain exact while some secondary floating-point values drift and PyTorch checkpoint bytes are not identical.

This software reproducibility repair does not rescue the negative ARC scientific result.

## Repaired ARC-v5 line

The separate train-only quantizer repair `arc-v5-stable-ema-residual-0.03125` restored its bounded trainability gate, but repaired validation remained `VALID_NEGATIVE_OR_INCONCLUSIVE_VALIDATION`. The generalization and quantization-benefit gates were not supported.

## Result

The defensible conclusion is **not** that LAM-JEPA beats ARC baselines. The full five-seed ARC controls result has survived multiple retained independent reruns and an independent raw-artifact recomputation with the same aggregate scientific conclusion. LAM-JEPA remains below its capacity-matched supervised baseline, and the planner/target mechanism criteria remain unsupported.

## Limitations

- Five validation seeds do not justify broad benchmark-general significance claims.
- ARC-Challenge is one benchmark family and the locked test remains intentionally unused for this failed hypothesis line.
- Independent runners exhibit low-order floating-point drift in per-example probabilities.
- Historical workflow reruns use a pull-request merge checkout; attempt 4 was verified to have zero file differences versus the scientific head.
- The stale raw claim-boundary string is a reporting defect and should not be read as the executed budget.
- The pretrained comparator is bounded development characterization, not a full matched confirmatory trial.
- No claim of educational effectiveness, general benchmark superiority, AGI, or general intelligence is supported.

## Stop rule

Do not tune the current architecture or thresholds against the locked ARC test. Any new architectural repair, benchmark, or scientific hypothesis must receive a new versioned protocol before its validation evidence is observed.
