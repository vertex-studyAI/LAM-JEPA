# LAM-JEPA

LAM-JEPA is a latent-action joint-embedding predictive architecture for adaptive educational reasoning, verification, and tutoring.

This repo includes:

- reproducible single-run training
- seed sweeps and aggregation
- ed-tech task generators for math, science, reading, tutoring, and reasoning
- student-state modeling
- misconception diagnosis
- curriculum and intervention selection
- ablations, calibration, OOD, and seed-level statistics
- paper-ready result generation and visualization hooks

## Install

LAM-JEPA requires Python 3.10 or newer.

```bash
python -m pip install -e .
```

Verify the installation with the repository reproducibility gate:

```bash
python -m compileall -q src scripts tests
python -m unittest discover -s tests -v
```

## Core commands

Train one reproducible run:

```bash
python scripts/train/train_single.py --seed 1 --steps 200 --out-dir experiments/seed_1
```

The canonical resumable checkpoint is written to:

```text
experiments/seed_1/final.pt
```

It includes the model, optimizer, scheduler, step, random-number-generator state, model configuration, and training configuration. Use `--checkpoint-dir` only as the legacy alias for `--out-dir`. Use `--out PATH` when an additional byte-for-byte copy of the canonical checkpoint is needed.

Run a one-step CPU smoke test:

```bash
python scripts/train/train_single.py \
  --seed 1 \
  --steps 1 \
  --batch-size 2 \
  --task parity \
  --device cpu \
  --out-dir experiments/smoke
```

Evaluate a checkpoint across all benchmark tasks:

```bash
python scripts/eval/eval_all.py --checkpoint experiments/seed_1/final.pt
```

Run the full benchmark suite:

```bash
python scripts/bench/run_benchmarks.py --steps 120 --seeds 1 2 3 4 5
```

Aggregate seed runs:

```bash
python scripts/analysis/aggregate_seeds.py --runs-dir experiments --out experiments/aggregate/summary.json
```

Generate paper-ready summaries:

```bash
python scripts/paper/generate_results.py --out-dir papers
```

## Benchmark tasks

The suite covers both classic synthetic reasoning and ed-tech style tasks:

- parity
- modular addition
- algebraic solving
- chained arithmetic
- GSM8K-style arithmetic
- equation solving
- science/physics reasoning
- reading comprehension
- tutoring diagnosis
- abstract reasoning

## Reproducibility boundary

The automated gate proves that the documented CPU training command executes and produces a checkpoint that can be loaded through LAM-JEPA's own checkpoint API. It does not establish benchmark quality, scientific novelty, or task-level superiority. Those claims require declared datasets, sufficient training budgets, seed-level statistics, baselines, ablations, and independently inspectable experiment artifacts.
