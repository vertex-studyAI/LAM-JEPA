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
python scripts/eval/eval_all.py \
  --checkpoint experiments/seed_1/final.pt \
  --device cpu \
  --batch-size 64 \
  --batches 8 \
  --seed 7 \
  --out outputs/eval_all.json
```

The evaluator loads the canonical checkpoint through the repository checkpoint API, places sampled inputs on the model's device, evaluates every task in `EDTECH_TASKS`, and records the checkpoint step, device, seed, batch settings, task list, sample counts, accuracy, confidence, distinct input count, distinct prompt count, distinct label count, and target semantics. Generated text-style evaluation rows are sampled one example at a time instead of repeating one example across an entire batch. A fixed seed makes repeated smoke runs comparable, but it does not replace multi-seed statistical evaluation.

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
5. reloads the artifact through LAM-JEPA's own checkpoint API;
6. evaluates one seeded batch for every declared benchmark task;
7. independently verifies complete task coverage, finite metrics, confidence and accuracy bounds, exact sample counts, input diversity, generated-prompt diversity, and declared target semantics; and
8. uploads the canonical checkpoint, training output, evaluation output, and structured verification reports as short-lived workflow evidence.

This gate proves that the documented installation, primary training path, checkpoint reload, and all-task evaluation path execute end to end. It does **not** establish benchmark quality or scientific performance.

## Benchmark tasks and metric semantics

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

Metrics are not semantically identical across all tasks. `parity`, `modadd`, `algebra`, `chain`, `equation`, and `science` currently use answer-class targets. `gsm8k`, `reading`, `tutoring`, and `reasoning` currently use concept-proxy labels produced by their synthetic generators. Accuracy on those four proxy tasks measures classification of the generated concept label; it is **not** evidence that the model produced or verified the natural-language answer. Evaluation JSON records `target_semantics` per task so downstream tables cannot silently present proxy accuracy as answer correctness.

## Notes

The repo is structured to support reproducible research, but benchmark quality still depends on the actual training budget, dataset quality, evaluation protocol, baselines, and seed-level statistics you run. Passing CI is an execution check, not evidence of model superiority, novelty, or educational effectiveness. Distinct-input counts expose evaluation diversity, but they do not turn synthetic proxy objectives into validated educational reasoning benchmarks.
