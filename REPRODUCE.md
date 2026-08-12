# LAM-JEPA Reproduction Protocol

**Audit date:** 12 August 2026  
**Pinned repository head:** `2f59b4297e5978d4ce769ebe95adb363e1e75d7a`

This protocol separates three evidence classes:

1. **fresh execution** — command actually rerun in the current environment;
2. **retained canonical evidence** — raw/derived evidence already committed and independently checked by repository workflows;
3. **blocked reproduction** — command identified and pinned, but the current environment cannot execute it without changing the environment or acquiring missing data/dependencies.

Never relabel retained evidence as a fresh rerun.

## 1. Environment

Required by the repository:

```bash
python --version
python -m pip install -e .
```

Python 3.10+ is required. A repository container is also available:

```bash
docker build -t lam-jepa .
docker run --rm lam-jepa
```

Before any scientific run, capture:

```bash
git rev-parse HEAD
python --version
python -m pip freeze > environment-pip-freeze.txt
uname -a > environment-platform.txt
```

Record accelerator model, driver/runtime versions, precision mode and peak memory when a GPU is used.

## 2. Smallest deterministic smoke path

Training:

```bash
python scripts/train/train_single.py \
  --seed 1 \
  --steps 20 \
  --out-dir experiments/smoke/checkpoints \
  --out experiments/smoke/final.pt
```

Evaluation:

```bash
python scripts/eval/eval_all.py \
  --checkpoint experiments/smoke/final.pt \
  --device cpu \
  --batch-size 32 \
  --batches 2 \
  --seed 7 \
  --out outputs/smoke-eval.json
```

A passing smoke test establishes only executable train/checkpoint/eval plumbing.

## 3. Minimum multi-seed path

```bash
python scripts/bench/run_benchmarks.py --steps 120 --seeds 1 2 3 4 5
python scripts/analysis/aggregate_seeds.py \
  --runs-dir experiments \
  --out experiments/aggregate/summary.json
```

Retain per-seed raw outputs. Report mean, dispersion and sample count; do not manuscript-select the best seed.

## 4. Paper-results package

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

For any claimed comparison, match dataset/split, preprocessing, evaluation rows, optimization budget, seed count, device/precision and metric implementation. Report parameter-budget mismatches explicitly.

## 5. ARC scientific boundary

The current ARC development-validation line is already negative/inconclusive under a frozen protocol. Reproduction work must preserve all of the following:

- no use of the locked ARC confirmatory test to rescue the failed validation hypothesis;
- no threshold tuning after observing confirmatory data;
- no silent substitution of a repaired architecture into the old hypothesis;
- no omission of adverse seeds, failed controls or invalid runs;
- no claim that the trainability repair validates the original hard-VQ mechanism.

A new architectural repair or scientific hypothesis requires a new versioned protocol frozen before its validation evidence is observed.

## 6. Failure / bug protocol

If a run fails:

1. retain command, commit, environment, stdout/stderr and exit code;
2. classify the failure as environment, dependency/data, numerical, checkpoint, evaluator or scientific;
3. do not overwrite the failed artifact;
4. implement the smallest correction;
5. state whether the correction changes only execution plumbing or changes the scientific protocol;
6. rerun the failed path and neighboring checks;
7. label pre-fix and post-fix results separately.

Protocol-changing fixes require a new experiment identifier.

## 7. Required run metadata

Each run record should contain at least:

- experiment ID;
- timestamp and operator/environment identifier;
- repository and commit SHA;
- exact command;
- training and evaluation seeds;
- dataset/version/split and eligibility counts;
- model/baseline/ablation identity;
- optimizer, learning rate, epochs/steps and batch size;
- device, precision, runtime and peak memory where measurable;
- stdout/stderr/log paths;
- checkpoint/result artifact paths plus hashes;
- metrics and uncertainty;
- exit status;
- scientific verdict and claim boundary;
- any bug/fix lineage.

## 8. Current-wave execution note

The 12 August 2026 ChatGPT execution sandbox had no outbound GitHub/dataset network from the runtime. Repository source/evidence could be inspected through the connected GitHub integration, but ARC could not be freshly downloaded and the full external-benchmark retrain was therefore **blocked by environment**, not treated as a scientific failure.

At pinned head `2f59b4297e5978d4ce769ebe95adb363e1e75d7a`, push-triggered GitHub workflows for research-claim boundaries, ARC protocol QA, reproducibility and container smoke completed successfully. Treat that as repository/packaging verification, not a new performance result.
