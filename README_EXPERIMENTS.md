# LAM-JEPA experiment guide

This file is the executable companion to the repository README. It describes the smallest reproducible experiment ladder and the evidence required before making stronger research claims.

## 1. Environment

LAM-JEPA requires Python 3.10+.

```bash
python -m pip install -e .
```

A containerized environment is also available:

```bash
docker build -t lam-jepa .
docker run --rm lam-jepa
```

The Docker image is a reproducible research environment, not a production service.

## 2. Fast smoke run

Train one deterministic short run:

```bash
python scripts/train/train_single.py \
  --seed 1 \
  --steps 20 \
  --out-dir experiments/smoke/checkpoints \
  --out experiments/smoke/final.pt
```

Evaluate the resulting checkpoint:

```bash
python scripts/eval/eval_all.py \
  --checkpoint experiments/smoke/final.pt \
  --device cpu \
  --batch-size 32 \
  --batches 2 \
  --seed 7 \
  --out outputs/smoke-eval.json
```

A successful smoke run establishes that the training/checkpoint/evaluation path executes. It does not establish benchmark validity, model superiority, or educational effectiveness.

## 3. Minimum multi-seed run

Use multiple training seeds before interpreting performance differences:

```bash
python scripts/bench/run_benchmarks.py --steps 120 --seeds 1 2 3 4 5
```

Aggregate retained runs:

```bash
python scripts/analysis/aggregate_seeds.py \
  --runs-dir experiments \
  --out experiments/aggregate/summary.json
```

Record at minimum:

- training seed and evaluation seed;
- training steps and batch size;
- checkpoint path and commit SHA;
- task-level sample counts;
- accuracy/confidence metrics;
- mean and dispersion across training seeds;
- failures, NaNs, interrupted runs, or invalid checkpoints.

## 4. Paper-results package

Generate the repository's reproducible results bundle with shared evaluation rows across training seeds:

```bash
python scripts/paper/generate_results.py \
  --out-dir papers \
  --seeds 1 2 3 4 5 \
  --steps 80 \
  --batch-size 32 \
  --eval-batches 6 \
  --evaluation-seed 1007 \
  --device cpu \
  --training-task mixed
```

Keep raw seed records and manifests. Do not copy only the best seed into a manuscript.

## 5. Matched comparison standard

For any claim that LAM-JEPA improves on another architecture or ablation, keep the comparison matched on:

- dataset and exact split;
- preprocessing and target semantics;
- evaluation rows;
- training steps/epochs;
- optimizer and learning-rate policy where applicable;
- parameter budget or a clearly reported mismatch;
- seed count;
- device class and precision;
- metric implementation.

A valid comparison should include the proposed model, a credible baseline, and at least one mechanism-removing ablation.

## 6. ARC-v5 boundary

The repository's current externally grounded ARC-v5 development-validation result is negative or inconclusive under the frozen protocol. Preserve that result. Do not tune thresholds against the locked confirmatory test set, and do not describe development-validation evidence as confirmatory generalization.

## 7. Promotion gates

Use these evidence levels:

1. **RUNNABLE** — documented command executes.
2. **TESTED** — deterministic repository checks pass.
3. **EXPERIMENTED** — retained multi-seed outputs exist.
4. **ANALYZED** — baseline/ablation comparison and failure analysis are recorded.
5. **PAPER DRAFT** — methods/results/limitations are backed by retained artifacts.
6. **RELEASE CANDIDATE** — clean reproduction instructions and provenance are present.
7. **RELEASED** — public artifact is actually published and the released revision is identified.

Never promote a result merely because a training job finished or CI is green.

## 8. Failure handling

When a run fails:

1. retain the command and error;
2. identify whether the failure is environment, data, numerical, checkpoint, or evaluator related;
3. apply the smallest robust correction;
4. rerun the failing path;
5. rerun neighboring checks;
6. record whether the correction changes the scientific protocol or only execution plumbing.

Protocol-changing fixes must be disclosed and should not be silently folded into earlier results.
