# LAM-JEPA Research Status

**Evidence cutoff:** 13 August 2026  
**Current audit base:** `6c6f5c10e8610239ce6c72a4fa7f549659662014`  
**Frozen full ARC-v3 scientific source:** `760aa7f9a73a177d5ff4ba7eb470f7e68ace63cb`  
**Seed-order software repair:** `b72a97a99769b278eb8ec75bc5eab62dc9599f29`  
**Classification:** `RESEARCH_ACTIVE / SCIENTIFIC_RESULT_REPRODUCED / ARC_SUPERIORITY_AND_MECHANISM_HYPOTHESES_UNSUPPORTED / RESEARCH_COMPLETE_FALSE`

## Executive result

LAM-JEPA now has a traceable external-benchmark reproducibility package covering the frozen ARC-Challenge protocol, fixed seeds, exact commands, matched and pretrained comparison paths, required controls, retained raw artifacts and digests, a documented reproducibility bug/fix lineage, and independent full scientific reruns.

The scientific conclusion remains negative/inconclusive. The current evidence does **not** support LAM-JEPA superiority on ARC, a validated planner benefit, a validated target/EMA benefit, or a repaired quantization/generalization benefit. Negative evidence is intentionally retained rather than tuned away.

The locked ARC confirmatory test remains unavailable to this failed hypothesis line and must not be used to rescue it.

## Full scientific reproduction closure — 13 August 2026

The frozen full-controls workflow `.github/workflows/arc-protocol-v3-full-controls-validation.yml` was independently rerun on scientific source `760aa7f9a73a177d5ff4ba7eb470f7e68ace63cb` without changing the scientific protocol after observing the prior result.

### Attempt 2

- Actions run: `31203337502`, attempt `2`;
- job: `94178988063`;
- artifact: `9149336081`;
- artifact digest: `sha256:c45710b5dae6a767ccb6bab7f6e3d8e9578752d8cf9b79fd82a65ae824dded1b`;
- conclusion: success.

### Attempt 3

- Actions run: `31203337502`, attempt `3`;
- job: `94291056903`;
- artifact: `9162165932`;
- artifact digest: `sha256:caa898f1ff046a337db9b5ddbffe1b332943a732868e2fd809abeda8ee89c30b`;
- conclusion: success.

Both attempts execute the frozen five-seed, 20-epoch ARC validation using all 1,117 eligible train rows and all 295 eligible validation rows. The aggregate full-model, ablation, negative-control, paired-effect and independent-verifier results are exactly equal between attempts. The locked test remains absent.

The two retained artifacts each contain 10 files; eight files are byte-identical. The raw probability-bearing JSON payloads show low-order cross-runner floating-point drift: 35,526 numeric leaves differ, no non-numeric leaves differ, and the maximum observed numeric drift is approximately `5.9186e-4`. This drift does not change any aggregate scientific metric or verifier decision. The repository therefore claims reproducibility of the aggregate scientific conclusion, not byte-exact identity of every floating-point probability.

## Frozen ARC protocol and eligibility

Protocol v3 uses checksum-addressed ARC-Challenge train/validation data and a deterministic feature-only eligibility rule retaining exactly-four-choice rows. This correction was frozen before confirmatory test access.

- train: 1,117 / 1,119 rows eligible;
- validation: 295 / 299 rows eligible;
- excluded rows remain evidence;
- source order is preserved;
- test split is not used for this failed line.

Frozen full-controls budget:

- seeds: `1 2 3 4 5`;
- epochs: `20`;
- batch size: `32`;
- learning rate: `0.0003`;
- model steps: `1`;
- device: CPU.

## Canonical scientific results

### Full model and required controls

- full LAM-JEPA: `0.2549152493 ± 0.0129968006`, `n=5`;
- `no_planner`: `0.2501694888 ± 0.0129968006`, `n=5`;
- `no_target`: `0.2616949081 ± 0.0203953938`, `n=5`;
- shuffled-label control: `0.2630508393 ± 0.0145011803`, `n=5`.

Paired mechanism effects:

- full minus `no_planner`: `+0.0047457606`, sample SD `0.0106118432`, bootstrap 95% CI `[0.0, 0.0142372817]`;
- full minus `no_target`: `-0.0067796588`, sample SD `0.0092834301`, bootstrap 95% CI `[-0.0135593176, 0.0]`.

The shuffled-label control remains below the frozen `0.35` failure threshold, but neither mechanism criterion is met.

### Capacity-matched supervised comparison

The separately retained matched comparison uses gradient-active parameter matching:

- LAM-JEPA gradient-active parameters: `86,372`;
- matched supervised gradient-active parameters: `86,644`;
- ratio: `1.0031491687`.

Results:

- LAM-JEPA: `0.2549152542 ± 0.0129968064`;
- matched supervised: `0.2664406780 ± 0.0154600058`;
- paired LAM minus matched: `-0.0115254237 ± 0.0140994131`.

**Verdict:** LAM-JEPA does not beat the capacity-matched supervised baseline on frozen validation.

### Strong pretrained comparator

The development comparator remains pinned to `microsoft/deberta-v3-xsmall` at immutable revision `14809e4f1fe1895fcba8b258271a940c6ca45ec4`.

Bounded development characterization:

- LAM-JEPA: `0.15625`;
- DeBERTa: `0.21875`;
- paired delta: `-0.0625`.

This is characterization evidence, not a standalone confirmatory inferiority trial.

## Seed-order reproducibility defect and repair

A software reproducibility defect was found in `train_single.py`: the model was initialized before applying the requested seed. Before repair, nominally identical same-seed one-step runs produced different losses (`10.853294372558594` versus `10.34877872467041`). The discrepant pre-fix evidence remains preserved.

PR #61 applied the smallest repair: seed before `LAMJEPA(cfg)` construction while retaining trainer-side seeding for subsequent randomness. No ARC split, scientific seed set, architecture, metric, threshold, or locked-test policy changed.

Six independently verified replay attempts after the repair preserve exact final loss `11.704492568969727` and final accuracy `0.0` across attempts. Some secondary floating-point quantities drift and serialized PyTorch checkpoint bytes are not byte-identical across independent runners. The seed-order repair is an execution reproducibility repair only and does not change the negative ARC conclusion.

## Reporting-metadata defect

The frozen full-controls raw payload contains a stale `protocol.claim_boundary` sentence saying the invocation is not the final five-seed/20-epoch protocol. The executable arguments and independent verifier demonstrate that the final five-seed/20-epoch protocol did execute.

This is a **non-invalidating reporting-metadata defect**. The raw artifact remains preserved unchanged; the stale sentence must not override the actual command, protocol fields or verifier evidence.

## ARC-v5 repaired line

A train-only causal investigation localized a trainability problem to the quantized latent path. The narrow opt-in repair `arc-v5-stable-ema-residual-0.03125` restored its bounded trainability gate.

Versioned lineage:

- repair merge: `df249086e9171febaa77333a4c62888f35265c40`;
- validation protocol freeze: `168f6beb434610752da4cb2cb6161f15ee026663`;
- validation execution: `18bd608a05bc308056e6279b347ff3ddb2b751be`;
- verifier-only float32 tolerance fix: `05c039fcc02c09c0aa1c1487596dcdd741ee6d51`.

The independent repaired-validation verdict remains `VALID_NEGATIVE_OR_INCONCLUSIVE_VALIDATION`. The generalization and quantization-benefit gates are not supported. The repair therefore does not rescue the original hard-VQ mechanism claim.

## Supported claims

The repository can defensibly state that:

1. the core training/checkpoint/evaluation pipeline executes reproducibly at the semantic/aggregate level;
2. ARC-Challenge external-benchmark plumbing and evidence retention are implemented;
3. the full frozen five-seed ARC controls result has been independently rerun with the same aggregate scientific conclusion and strict verifier output;
4. matched-capacity and pinned pretrained comparison paths are implemented;
5. adverse and negative results are retained;
6. a real seed-order reproducibility defect was found, preserved, narrowly fixed and independently replayed;
7. the repaired v5 trainability gate passes while repaired validation remains negative/inconclusive;
8. command → commit → environment → seeds → raw artifact → aggregate/verifier → conclusion is traceable in `RESULTS.md`, `REPRODUCE.md`, `experiments/repro_wave_2026_08_12/experiment_metadata.json`, and `EVIDENCE_AUDIT_20260813.md`.

## Claims not supported

Do not claim:

- LAM-JEPA superiority on ARC;
- planner benefit on ARC;
- target/EMA mechanism benefit on ARC;
- quantization benefit from repaired validation;
- byte-exact cross-runner floating-point identity;
- externally validated educational effectiveness;
- general benchmark superiority;
- AGI or general intelligence capability;
- `RESEARCH_COMPLETE`.

## Scientific stop rule

Do not unlock or use the ARC confirmatory test to rescue the current failed superiority/mechanism hypothesis. Any future architectural repair, benchmark or scientific hypothesis must be versioned separately and preregistered before its validation evidence is observed.

Negative results remain first-class research artifacts and should remain visible in manuscripts, technical reports and portfolio summaries.

## Publication/package boundary

Scientific reproducibility and publication packaging are separate. Issue #14 remains the authoritative gate for owner-approved licensing, citation metadata, provenance and release packaging. A license, author identity or publication state must not be invented.

## Canonical evidence documents

- `RESULTS.md` — retained scientific values and rerun comparison;
- `REPRODUCE.md` — exact reproduction commands and failure policy;
- `experiments/repro_wave_2026_08_12/experiment_metadata.json` — machine-readable lineage;
- `EVIDENCE_AUDIT_20260813.md` — skeptical-reviewer traceability audit;
- `papers/MANUSCRIPT_RESULTS_20260813.md` — conservative manuscript-ready result text.
