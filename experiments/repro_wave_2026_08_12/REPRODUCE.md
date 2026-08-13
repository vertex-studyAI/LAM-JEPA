# Reproduce the 2026-08-12 to 2026-08-13 LAM-JEPA wave

This file separates the frozen scientific reruns from the later reproducibility-plumbing repair. Do not use the locked ARC test split to rescue the failed validation hypothesis.

## 1. Frozen ARC protocol-v3 full-controls experiment

Scientific source SHA: `760aa7f9a73a177d5ff4ba7eb470f7e68ace63cb`.

Environment:

- Python 3.11
- CPU PyTorch
- `pip install -e '.[external-benchmarks]'`
- ARC-Challenge train and validation downloaded through the checksum-addressed repository downloader
- ARC test explicitly absent

Verify the frozen protocol and download train/validation only:

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

Run the frozen five-seed, twenty-epoch full-controls validation:

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

Independently verify the retained result:

```bash
python scripts/ci/verify_arc_protocol_v3_full_controls.py \
  --results ci-evidence/arc-protocol-v3-full-controls-validation.json \
  --protocol protocols/arc_challenge_v3.json \
  --train ci-evidence/arc-data/arc-challenge-train.parquet \
  --validation ci-evidence/arc-data/arc-challenge-validation.parquet \
  --report ci-evidence/arc-protocol-v3-full-controls-validation-verification.json \
  | tee ci-evidence/arc-protocol-v3-full-controls-validation-verifier-output.txt
```

Required invariants:

- seeds `[1,2,3,4,5]`;
- 20 epochs;
- batch size 32;
- learning rate `0.0003`;
- model steps 1;
- 1,117 eligible train rows used;
- 295 eligible validation rows used;
- locked test not evaluated;
- verifier verdict `PROTOCOL_V3_FULL_CONTROLS_VALIDATION_VERIFIED`;
- research complete remains false.

### Retained independent reruns

Workflow run `31203337502` has two successful retained reruns of the full scientific experiment:

- attempt 2: job `94178988063`, artifact `9149336081`, digest `sha256:c45710b5dae6a767ccb6bab7f6e3d8e9578752d8cf9b79fd82a65ae824dded1b`;
- attempt 3: job `94291056903`, artifact `9162165932`, digest `sha256:caa898f1ff046a337db9b5ddbffe1b332943a732868e2fd809abeda8ee89c30b`.

The aggregate scientific metrics, paired effects, negative-control summary and verifier verdict reproduce exactly. Low-order per-example floating-point values are not required to be byte-identical across independent runners.

## 2. Reverify the immutable retained full-controls artifact without retraining

The repository also contains a verifier-only workflow that downloads immutable artifact `9000793334` from source run `31195682685` and checks its recorded digest before recomputing the result from retained raw evidence.

Equivalent verifier command after downloading the exact artifact and metadata:

```bash
python scripts/ci/verify_arc_protocol_v3_full_controls_artifact.py \
  --results retained-evidence/arc-protocol-v3-full-controls-validation.json \
  --protocol protocols/arc_challenge_v3.json \
  --train retained-evidence/arc-data/arc-challenge-train.parquet \
  --validation retained-evidence/arc-data/arc-challenge-validation.parquet \
  --artifact-metadata verification-evidence/source-artifact-metadata.json \
  --report verification-evidence/full-controls-artifact-verification.json
```

This path must not retrain the model. Its purpose is evidence-chain verification.

## 3. Frozen ARC-v5 repaired-validation line

Historical result run: GitHub Actions `31459399884`, head SHA `1001fbe580b6369d5ff7ee7abb9e4edaae96121d`.

```bash
python scripts/benchmark/run_arc_v5_repaired_validation.py \
  --protocol protocols/arc_challenge_v5_repaired_validation.json \
  --train ci-evidence/arc-data/arc-challenge-train.parquet \
  --validation ci-evidence/arc-data/arc-challenge-validation.parquet \
  --device cpu \
  --out ci-evidence/arc-v5-repaired-validation.json

python scripts/ci/verify_arc_v5_repaired_validation.py \
  --results ci-evidence/arc-v5-repaired-validation.json \
  --protocol protocols/arc_challenge_v5_repaired_validation.json \
  --validation ci-evidence/arc-data/arc-challenge-validation.parquet \
  --report ci-evidence/arc-v5-repaired-validation-verification.json
```

Verdict remains `VALID_NEGATIVE_OR_INCONCLUSIVE_VALIDATION`. Do not download or inspect ARC test for this protocol.

## 4. Pre-fix reproducibility defect

At SHA `2f59b4297e5978d4ce769ebe95adb363e1e75d7a`, repeated execution of the same declared seed produced one-step losses `10.853294372558594` and `10.34877872467041`.

Root cause: model initialization occurred before `set_seed(seed)`.

Merged repair: SHA `b72a97a99769b278eb8ec75bc5eab62dc9599f29`, PR #61. The scientific protocol was not changed.

## 5. Post-fix exact same-seed replay

```bash
python scripts/train/train_single.py \
  --seed 1 --steps 1 --batch-size 2 --task parity --device cpu \
  --out-dir replay-evidence/first-checkpoints \
  --out replay-evidence/first.pt

python scripts/train/train_single.py \
  --seed 1 --steps 1 --batch-size 2 --task parity --device cpu \
  --out-dir replay-evidence/second-checkpoints \
  --out replay-evidence/second.pt

python scripts/ci/verify_deterministic_training_replay.py \
  --first replay-evidence/first.pt \
  --second replay-evidence/second.pt \
  --report replay-evidence/replay-verification.json
```

The verifier requires exact model tensors, final metrics, semantic checkpoint metadata and RNG-state values within an attempt. Output-path metadata is intentionally excluded because the two replay runs write to different directories.

Latest independent replay evidence: workflow run `31641305854`, six verified attempts, latest artifact `9160533550`, digest `sha256:84ef6f4a9c4274441a8e8a4b959620551cd37ae6fbd29a0efd07510553359354`.

Across independent CPU attempts, final loss `11.704492568969727` and final accuracy `0.0` remain exact. PyTorch checkpoint bytes and all secondary floats are not claimed byte-identical across runners.

## 6. Reporting metadata defect

Do not rewrite the frozen raw artifact to hide its stale claim-boundary sentence. Preserve it and record that workflow arguments plus the independent verifier establish the actual five-seed/20-epoch budget. This is a reporting-metadata defect, not a scientific-result repair.
