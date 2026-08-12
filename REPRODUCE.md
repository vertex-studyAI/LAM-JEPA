# LAM-JEPA — REPRODUCE

## Environment

LAM-JEPA requires Python 3.10+.

```bash
python -m pip install -e .
```

## Minimum multi-seed benchmark

```bash
python scripts/bench/run_benchmarks.py --steps 120 --seeds 1 2 3 4 5
python scripts/analysis/aggregate_seeds.py \
  --runs-dir experiments \
  --out experiments/aggregate/summary.json
```

## Paper-results package

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

## Comparison contract

For any comparison, keep the dataset/split, preprocessing, target semantics, evaluation rows, training budget, optimizer policy, parameter budget or explicitly reported mismatch, seed count, device/precision and metric implementation matched.

Capture commit SHA, command, environment, training/evaluation seed, runtime, logs, checkpoint paths, task-level sample counts, raw metrics and all failures. Preserve all retained seeds; never select only the best seed.

## Failure and negative-result handling

Do not silently change the scientific protocol after observing an outcome. If a bug is execution-only, document the old failure, apply the smallest fix and rerun. If the fix changes the protocol, version it as a new experiment. The current ARC-v5 validation result is negative/inconclusive and the confirmatory test remains locked for that hypothesis line.
