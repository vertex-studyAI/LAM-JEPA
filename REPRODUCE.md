# Reproduce LAM-JEPA

This document separates the fast executable reproducibility gate from the retained full scientific validation result.

## Freeze the revision

```bash
git checkout 2f59b4297e5978d4ce769ebe95adb363e1e75d7a
git status --short
git rev-parse HEAD
```

Do not change model, data, thresholds, seeds, or evaluation policy after seeing a result. If an execution bug is found, retain the old failed evidence, make the smallest versioned fix, and rerun under a newly identified revision.

## Environment

The exact-head reproduction wave used the repository's `Reproducibility CI`:

- GitHub-hosted Ubuntu runner (`ubuntu-latest`)
- Python 3.11
- CPU-only PyTorch
- editable package install with `.[external-benchmarks]`
- CUDA explicitly asserted unavailable

Equivalent setup:

```bash
python -m pip install --upgrade pip
python -m pip install torch --index-url https://download.pytorch.org/whl/cpu
python -m pip install -e '.[external-benchmarks]'
python -c 'import torch; assert not torch.cuda.is_available()'
python -m compileall -q src scripts
```

## Frozen ARC protocol verification

```bash
mkdir -p ci-evidence
python scripts/ci/verify_arc_protocol.py \
  --protocol protocols/arc_challenge_v1.json \
  --dataset-manifest data/manifests/arc_challenge.json \
  --report ci-evidence/arc-protocol-verification.json

python scripts/data/download_arc_challenge.py \
  --splits train validation \
  --out-dir ci-evidence/arc-data \
  | tee ci-evidence/arc-download.json

test ! -e ci-evidence/arc-data/arc-challenge-test.parquet
```

The final line is a scientific boundary, not a convenience check: the locked test split must remain absent for the current failed hypothesis line.

## Exact CI external-benchmark smoke

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
  --out ci-evidence/arc-challenge-smoke.json \
  | tee ci-evidence/arc-challenge-output.txt

python scripts/ci/verify_arc_challenge_smoke.py \
  --results ci-evidence/arc-challenge-smoke.json \
  --report ci-evidence/arc-challenge-verification.json
```

This smoke tests execution and evidence structure only. It is not the full scientific validation.

## Multi-seed interface check

```bash
python scripts/bench/run_benchmarks.py \
  --variant full \
  --steps 1 \
  --batch-size 2 \
  --eval-batches 1 \
  --evaluation-seed 17 \
  --device cpu \
  --seeds 1 2 \
  --out ci-evidence/benchmark-multiseed.json
```

## Deterministic checkpoint and evaluation path

```bash
python scripts/train/train_single.py \
  --seed 1 \
  --steps 1 \
  --batch-size 2 \
  --task parity \
  --device cpu \
  --out-dir ci-evidence/checkpoints \
  --out ci-evidence/final.pt

python scripts/eval/eval_all.py \
  --checkpoint ci-evidence/final.pt \
  --device cpu \
  --batch-size 2 \
  --batches 1 \
  --seed 1 \
  --out ci-evidence/eval-all.json
```

## Reference baselines and exact-row comparison

```bash
python scripts/eval/eval_baselines.py \
  --batch-size 2 \
  --batches 1 \
  --seed 1 \
  --out ci-evidence/eval-baselines.json

python scripts/analysis/compare_model_to_baselines.py \
  --model ci-evidence/eval-all.json \
  --baselines ci-evidence/eval-baselines.json \
  --out ci-evidence/model-baseline-comparison.json
```

## Paper-results and component-ablation smoke

```bash
python scripts/paper/generate_results.py \
  --out-dir ci-evidence/paper-results \
  --seeds 1 2 \
  --steps 1 \
  --batch-size 2 \
  --eval-batches 1 \
  --evaluation-seed 17 \
  --device cpu \
  --training-task parity

python scripts/benchmark/run_ablations.py \
  --seeds 1 2 \
  --steps 1 \
  --batch-size 2 \
  --eval-batches 1 \
  --evaluation-seed 19 \
  --training-task parity \
  --device cpu \
  --out ci-evidence/ablation-results.json
```

Run the repository verifier scripts after each generated artifact, as done in `.github/workflows/reproducibility-ci.yml`.

## Full retained scientific protocol

The canonical ARC full-controls result uses the already frozen scientific protocol: five paired seeds, 20 epochs, batch size 32, learning rate 0.0003, model steps 1, all 1,117 eligible train rows, and all 295 eligible validation rows. Consult the versioned protocol and retained raw artifacts before rerunning; do not reconstruct missing details from memory or silently substitute the CI smoke settings.

## 2026-08-12 reproduction record

- source SHA: `2f59b4297e5978d4ce769ebe95adb363e1e75d7a`
- GitHub Actions workflow run: `31610608912`
- rerun attempt: `2`
- job: `94178401933`
- start: `2026-08-12T16:06:02Z`
- completion: `2026-08-12T16:07:43Z`
- conclusion: `success`
- evidence artifact name: `lam-jepa-training-evaluation-smoke`
- retention configured by workflow: 7 days

## Reporting policy

Report means, dispersion, paired deltas, sample count, and confidence intervals where they exist. Do not claim significance without a suitable predeclared test. Do not select only the best seed. Do not use the locked test set to rescue the current negative/inconclusive ARC line.
