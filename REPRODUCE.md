# Reproduce LAM-JEPA

This document separates three evidence classes: fast execution smoke, deterministic same-seed replay after the seed-order repair, and the frozen full five-seed ARC scientific validation. Do not substitute one for another.

## 1. Revisions and claim boundary

The software reproducibility repair merged at:

```bash
git checkout b72a97a99769b278eb8ec75bc5eab62dc9599f29
git status --short
git rev-parse HEAD
```

The frozen full-controls ARC scientific rerun is tied to:

```text
760aa7f9a73a177d5ff4ba7eb470f7e68ace63cb
```

Do not reinterpret the software seed-order fix as a new scientific ARC result. Do not change model, data, thresholds, seeds, splits, or evaluation policy after seeing a result. The locked ARC test must remain unused for this failed hypothesis line.

## 2. CPU environment

Equivalent setup for repository verification:

```bash
python -m pip install --upgrade pip
python -m pip install torch --index-url https://download.pytorch.org/whl/cpu
python -m pip install -e '.[external-benchmarks]'
python -c 'import torch; assert not torch.cuda.is_available()'
python -m compileall -q src scripts
```

The frozen ARC full-controls workflow uses GitHub-hosted Ubuntu with Python 3.11 and CPU PyTorch.

## 3. Frozen ARC data boundary

```bash
mkdir -p ci-evidence
python scripts/ci/verify_arc_protocol_v3.py \
  --protocol protocols/arc_challenge_v3.json \
  --dataset-manifest data/manifests/arc_challenge.json \
  --report ci-evidence/arc-protocol-v3-verification.json

python scripts/data/download_arc_challenge.py \
  --splits train validation \
  --out-dir ci-evidence/arc-data \
  | tee ci-evidence/arc-download-full-controls-v3.json

test ! -e ci-evidence/arc-data/arc-challenge-test.parquet
```

The final assertion is a scientific stop rule, not optional housekeeping.

## 4. Full frozen five-seed ARC controls validation

From scientific revision `760aa7f9a73a177d5ff4ba7eb470f7e68ace63cb`, execute exactly:

```bash
python scripts/benchmark/run_arc_protocol_v3_controls.py \
  --train ci-evidence/arc-data/arc-challenge-train.parquet \
  --validation ci-evidence/arc-data/arc-challenge-validation.parquet \
  --seeds 1 2 3 4 5 \
  --epochs 20 \
  --batch-size 32 \
  --learning-rate 0.0003 \
  --model-steps 1 \
  --train-limit 0 \
  --validation-limit 0 \
  --device cpu \
  --out ci-evidence/arc-protocol-v3-full-controls-validation.json \
  | tee ci-evidence/arc-protocol-v3-full-controls-validation-output.txt
```

Then independently verify:

```bash
python scripts/ci/verify_arc_protocol_v3_full_controls.py \
  --results ci-evidence/arc-protocol-v3-full-controls-validation.json \
  --protocol protocols/arc_challenge_v3.json \
  --train ci-evidence/arc-data/arc-challenge-train.parquet \
  --validation ci-evidence/arc-data/arc-challenge-validation.parquet \
  --report ci-evidence/arc-protocol-v3-full-controls-validation-verification.json \
  | tee ci-evidence/arc-protocol-v3-full-controls-validation-verifier-output.txt
```

Required budget assertions:

- seeds exactly `[1, 2, 3, 4, 5]`;
- 20 epochs;
- batch size 32;
- learning rate 0.0003;
- model steps 1;
- all 1,117 eligible train rows;
- all 295 eligible validation rows;
- locked test not evaluated;
- verifier verdict `PROTOCOL_V3_FULL_CONTROLS_VALIDATION_VERIFIED`.

### Retained independent reruns

Attempt 2:

- Actions run `31203337502`, attempt `2`;
- job `94178988063`;
- artifact `9149336081`;
- digest `sha256:c45710b5dae6a767ccb6bab7f6e3d8e9578752d8cf9b79fd82a65ae824dded1b`.

Attempt 3:

- Actions run `31203337502`, attempt `3`;
- job `94291056903`;
- artifact `9162165932`;
- digest `sha256:caa898f1ff046a337db9b5ddbffe1b332943a732868e2fd809abeda8ee89c30b`.

Both attempts succeeded. Aggregate model/ablation/negative-control summaries and verifier outputs are exactly equal between attempts. Raw per-example probabilities show low-order floating-point drift, so byte-exact raw JSON identity is not required.

The artifact comparison found 10 retained files in each attempt. Eight are byte-identical. The raw full-results JSON and normalized-input copy differ numerically at prediction-probability level; maximum observed numeric drift was approximately `5.9186e-4`. No non-numeric leaf changed, and the aggregate scientific conclusion did not change.

## 5. Frozen scientific result expected from the full rerun

- full LAM-JEPA validation accuracy: `0.2549152493 ± 0.0129968006`, `n=5`;
- `no_planner`: `0.2501694888 ± 0.0129968006`, `n=5`;
- `no_target`: `0.2616949081 ± 0.0203953938`, `n=5`;
- shuffled-label control: `0.2630508393 ± 0.0145011803`, `n=5`, pass under frozen `0.35` ceiling;
- full − `no_planner`: `+0.0047457606`, bootstrap 95% CI `[0.0, 0.0142372817]`;
- full − `no_target`: `-0.0067796588`, bootstrap 95% CI `[-0.0135593176, 0.0]`.

The separately retained capacity-matched supervised baseline remains stronger in mean accuracy. Do not report LAM-JEPA superiority or validated planner/target benefit.

## 6. Reporting-metadata defect in frozen raw output

The frozen `run_arc_protocol_v3_controls.py` payload contains a stale `protocol.claim_boundary` sentence saying the invocation is “not the final five-seed/20-epoch protocol.” Preserve that raw artifact unchanged.

For attempts 2 and 3, this sentence is inconsistent with the actual arguments and independent verifier: both runs use five seeds, 20 epochs, full eligible train/validation rows, and satisfy `final_five_seed_20_epoch_protocol_executed = true`.

Treat this as a non-invalidating reporting-metadata defect. Do not use the stale sentence to override the executable command, protocol fields, or verifier evidence.

## 7. Pre-fix deterministic-training failure

Before the seed-order repair, `train_single.py` instantiated `LAMJEPA` before applying the requested seed. Under nominally identical SHA / command / seed / CPU execution, one-step losses differed:

```text
10.853294372558594
10.34877872467041
```

Retain this as invalidated reproducibility evidence rather than deleting it.

## 8. Deterministic seed-order repair

PR #61 applied the narrow repair: apply the requested seed before model construction while retaining trainer-side seeding for subsequent data/training randomness.

The repair changed no ARC data split, scientific seed set, metric, threshold, architecture, or locked-test policy.

Current machine-readable replay metadata records six independently verified deterministic replay attempts. Within each runner attempt, model state, final metrics and RNG state are exact. Across attempts, final loss `11.704492568969727` and final accuracy `0.0` remain exact, while some secondary floating-point values drift and serialized PyTorch checkpoints are not byte-identical.

Do not claim byte-for-byte checkpoint identity across independent runners.

## 9. Fast smoke paths

A fast external benchmark smoke can be used to check plumbing, but not scientific effects:

```bash
python scripts/benchmark/run_arc_challenge.py \
  --train ci-evidence/arc-data/arc-challenge-train.parquet \
  --validation ci-evidence/arc-data/arc-challenge-validation.parquet \
  --seeds 1 2 \
  --epochs 1 \
  --batch-size 4 \
  --model-steps 1 \
  --train-limit 16 \
  --validation-limit 16 \
  --device cpu \
  --out ci-evidence/arc-challenge-smoke.json
```

Smoke success establishes executability only.

## 10. Failure policy

If a future scientific rerun differs:

1. retain the exact command, SHA, environment, seeds, logs, metrics and artifact;
2. classify the difference as environment, data, nondeterminism, evaluator, metadata, or scientific-result drift;
3. do not alter the frozen threshold or locked-test policy;
4. if a software bug invalidates execution, preserve the old evidence, make the smallest versioned fix, rerun, and distinguish old from new results;
5. do not select only favorable seeds.

## 11. Independent retained-artifact integrity check

A reviewer who has downloaded attempts 2 and 3 can verify the retained evidence without generating a new training sample:

```bash
sha256sum lam-jepa-arc-attempt2.zip lam-jepa-arc-attempt3.zip
rm -rf attempt2 attempt3
mkdir attempt2 attempt3
unzip -q lam-jepa-arc-attempt2.zip -d attempt2
unzip -q lam-jepa-arc-attempt3.zip -d attempt3
find attempt2 -type f -print | sort
find attempt3 -type f -print | sort
(cd attempt2 && find . -type f -print0 | sort -z | xargs -0 sha256sum)
(cd attempt3 && find . -type f -print0 | sort -z | xargs -0 sha256sum)
```

Expected ZIP SHA-256 values:

- attempt 2: `c45710b5dae6a767ccb6bab7f6e3d8e9578752d8cf9b79fd82a65ae824dded1b`;
- attempt 3: `caa898f1ff046a337db9b5ddbffe1b332943a732868e2fd809abeda8ee89c30b`.

Expected structural comparison:

- 10 retained file paths in each archive;
- 8 byte-identical files;
- differing files only `arc-protocol-v3-full-controls-validation.json` and `arc-protocol-v3-full-controls-validation-verification-normalized-input.json`;
- 35,526 numeric leaf differences and 0 non-numeric differences in each differing JSON tree;
- maximum observed absolute numeric drift `0.0005918592214584351`;
- unchanged aggregate scientific result and unchanged verifier decision.

The machine-readable reference for this audit is `audits/repro_wave_2026-08-13.json` and the reviewer-facing record is `audits/REPRO_WAVE_2026-08-13.md`.

This artifact-integrity procedure is not a substitute for a fresh scientific rerun. It verifies that retained evidence and the written ledger agree.

## Reporting policy

Report means, sample dispersion, paired deltas, sample count and confidence intervals where available. Do not claim significance without an appropriate predeclared analysis. Preserve negative results, low-order numerical drift, and reporting defects as part of the evidence trail.
