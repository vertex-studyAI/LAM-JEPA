# Reproduce LAM-JEPA

This document separates the fast executable reproducibility gate from the retained full scientific validation result and preserves the pre-fix deterministic-training failure.

## Freeze the revision

For the repaired execution path:

```bash
git checkout b72a97a99769b278eb8ec75bc5eab62dc9599f29
git status --short
git rev-parse HEAD
```

The frozen ARC scientific estimates remain tied to their retained artifacts and protocol lineage; do not reinterpret the seed-order repair as a new ARC result. Do not change model, data, thresholds, seeds, or evaluation policy after seeing a result. If an execution bug is found, retain the old failed evidence, make the smallest versioned fix, and rerun under a newly identified revision.

## Environment

The repaired exact-head reproduction used GitHub-hosted Ubuntu runners with Python 3.11 and CPU-only PyTorch. Equivalent setup:

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

The final line is a scientific boundary: the locked test split must remain absent for the current failed hypothesis line.

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

Run the same seed twice into separate outputs. On revisions before the seed-order fix this check can fail because model initialization happened before the requested seed was applied.

```bash
mkdir -p ci-evidence/replay-a ci-evidence/replay-b

python scripts/train/train_single.py \
  --seed 1 \
  --steps 1 \
  --batch-size 2 \
  --task parity \
  --device cpu \
  --out-dir ci-evidence/replay-a/checkpoints \
  --out ci-evidence/replay-a/final.pt

python scripts/train/train_single.py \
  --seed 1 \
  --steps 1 \
  --batch-size 2 \
  --task parity \
  --device cpu \
  --out-dir ci-evidence/replay-b/checkpoints \
  --out ci-evidence/replay-b/final.pt
```

Use the repository deterministic replay verifier/workflow to require exact model-state, final-metric, and RNG-state equality **within the same runner attempt**. Then evaluate an accepted checkpoint:

```bash
python scripts/eval/eval_all.py \
  --checkpoint ci-evidence/replay-a/final.pt \
  --device cpu \
  --batch-size 2 \
  --batches 1 \
  --seed 1 \
  --out ci-evidence/eval-all.json
```

Do not require serialized `.pt` files to be byte-identical across independent GitHub runners. The retained independent replay found exact final loss and accuracy across attempts but low-order (`~1e-6` to `1e-7`) drift in some floating-point submetrics and non-identical checkpoint bytes.

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

## 2026-08-12 lineage record

### Pre-fix execution

- source SHA: `2f59b4297e5978d4ce769ebe95adb363e1e75d7a`
- Reproducibility CI run: `31610608912`, attempt `2`
- job: `94178401933`
- result: workflow success, but later same-seed replay exposed checkpoint nondeterminism
- observed one-step losses under nominally identical SHA / CLI / seed / CPU execution: `10.853294372558594` and `10.34877872467041`
- disposition: retain as invalidated reproducibility evidence; do not overwrite

### Minimal fix and PR-head rerun

- repaired PR head: `ced95ee10021d09419816aade3f5906a3d99663c`
- merged main commit: `b72a97a99769b278eb8ec75bc5eab62dc9599f29`
- Reproducibility CI: `31618228743` — success
- Deterministic training replay: `31618227708` — success
- ARC Protocol V2 QA: `31618228252` — success
- Research claim boundary: `31618228424` — success
- deterministic replay artifact ID: `9150159954`
- artifact SHA-256: `6ebd9a6e2d55b6cb2b06a65dc267cd354088ed314b0c41469fd5e76ddbd49c6c`

### Independent merged-main replay

- workflow run: `31620784264`
- merged repair SHA: `b72a97a99769b278eb8ec75bc5eab62dc9599f29`
- attempts verified: `2`
- within-attempt model state exact: `true`
- within-attempt final metrics exact: `true`
- within-attempt RNG state exact: `true`
- cross-attempt final loss exact: `true`
- cross-attempt final accuracy exact: `true`
- cross-attempt verifier JSON SHA-256: `1080efccc40d7a931451ec3fa5094113e877d54b4c16739cfe1861e22292f4af`
- cross-attempt checkpoint bytes exact: `false`
- cross-attempt low-order floating-point drift: `true`

The fix seeds before model construction and leaves the ARC scientific protocol, metrics, thresholds, seed set, data splits, architecture, and locked-test policy unchanged. Report this as semantic same-seed reproducibility under the documented CI path, not byte-for-byte identity across independent runners.

## Reporting policy

Report means, dispersion, paired deltas, sample count, and confidence intervals where they exist. Do not claim significance without a suitable predeclared test. Do not select only the best seed. Preserve pre-fix and post-fix lineage separately. Do not claim byte-exact cross-run checkpoint identity. Do not use the locked test set to rescue the current negative/inconclusive ARC line.
