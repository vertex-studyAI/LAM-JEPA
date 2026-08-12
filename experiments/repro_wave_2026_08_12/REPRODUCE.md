# Reproduce the 2026-08-12 LAM-JEPA wave

## Frozen ARC-v5 validation

Historical result run: GitHub Actions `31459399884`, head SHA `1001fbe580b6369d5ff7ee7abb9e4edaae96121d`.

The canonical workflow uses Python 3.11 and CPU PyTorch, verifies the frozen protocol, downloads checksum-addressed ARC train and validation only, asserts the test split is absent, and runs:

```bash
python scripts/benchmark/run_arc_v5_repaired_validation.py \
  --protocol protocols/arc_challenge_v5_repaired_validation.json \
  --train ci-evidence/arc-data/arc-challenge-train.parquet \
  --validation ci-evidence/arc-data/arc-challenge-validation.parquet \
  --device cpu \
  --out ci-evidence/arc-v5-repaired-validation.json
```

Independent retained-row verification:

```bash
python scripts/ci/verify_arc_v5_repaired_validation.py \
  --results ci-evidence/arc-v5-repaired-validation.json \
  --protocol protocols/arc_challenge_v5_repaired_validation.json \
  --validation ci-evidence/arc-data/arc-challenge-validation.parquet \
  --report ci-evidence/arc-v5-repaired-validation-verification.json
```

Do not download or inspect ARC test for this protocol.

## Pre-fix reproduction defect

At SHA `2f59b4297e5978d4ce769ebe95adb363e1e75d7a`, rerun the existing Reproducibility CI training smoke twice. The wave observed one-step losses `10.853294372558594` and `10.34877872467041` under the same declared seed, exposing unseeded model initialization.

## Post-fix exact replay

On branch `repro-wave-2026-08-12`:

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

The verifier requires exact model tensors, final metrics, semantic checkpoint metadata, and RNG state values. Output-path metadata is intentionally excluded because the two replay runs write to different directories.
