# LAM-JEPA

LAM-JEPA is a latent-action joint-embedding predictive architecture for adaptive educational reasoning, verification, and tutoring.

This repo now includes:

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

## Core commands

Train one reproducible run:

```bash
python scripts/train/train_single.py \
  --seed 1 \
  --steps 200 \
  --out-dir experiments/seed_1/checkpoints \
  --out experiments/seed_1/final.pt
```

The trainer writes its canonical resumable checkpoint to `experiments/seed_1/checkpoints/final.pt`. It contains the model, optimizer, scheduler, training step, metrics, RNG state, model configuration, and training configuration. `--checkpoint-dir` is an equivalent legacy name for `--out-dir`. When `--out` is supplied, the CLI creates a byte-for-byte copy of that same canonical checkpoint rather than a second incompatible format.

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

## Reproducibility gate

Every pull request and push to `main` runs a CPU-only smoke experiment that:

1. installs the package from `pyproject.toml` using PyTorch's CPU wheel channel;
2. confirms the environment is CPU-only and compiles the source and scripts;
3. trains the actual LAM-JEPA model for one deterministic parity step;
4. verifies finite model tensors plus optimizer, scheduler, RNG, model-configuration, and training-configuration state;
5. reloads the artifact through LAM-JEPA's own checkpoint API; and
6. uploads the canonical checkpoint, structured training output, and verification report as short-lived workflow evidence.

This gate proves that the documented installation, primary training path, and checkpoint reload execute end to end. It does **not** establish benchmark quality or scientific performance.

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

## Notes

The repo is structured to support reproducible research, but benchmark quality still depends on the actual training budget, dataset quality, and evaluation protocol you run. Passing CI is an execution check, not evidence of model superiority, novelty, or educational effectiveness.
