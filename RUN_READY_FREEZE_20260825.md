# LAM-JEPA Run-Ready Freeze — 25 August 2026

## Current truth

LAM-JEPA already has a reproducible internal package and a frozen negative/inconclusive ARC validation result. This freeze does **not** attempt to turn that result positive. It makes the repository easier to reproduce, independently review and hand to an executor without violating the locked-test boundary.

## Scientific boundary

- ARC-Challenge test remains absent for the frozen validation protocols.
- The retained ARC-v5 verdict remains `VALID_NEGATIVE_OR_INCONCLUSIVE_VALIDATION`.
- CI/reproducibility success does not establish novelty, educational effectiveness, external generalization, or model superiority.
- No metric/seed/budget switching is permitted to rescue the frozen result.
- Any new scientific hypothesis must live under a new versioned protocol rather than silently mutating the frozen line.

## Canonical executable surfaces already present

The repository README documents:

- editable install: `python -m pip install -e .`
- deterministic single-run training: `scripts/train/train_single.py`
- all-task evaluation: `scripts/eval/eval_all.py`
- benchmark suite: `scripts/bench/run_benchmarks.py`
- seed aggregation: `scripts/analysis/aggregate_seeds.py`
- paper-results generation: `scripts/paper/generate_results.py`

The retained reproduction guide at `experiments/repro_wave_2026_08_12/REPRODUCE.md` additionally defines the frozen ARC protocol-v3 controls experiment, verifier-only retained-artifact path, ARC-v5 repaired-validation line, and deterministic same-seed replay.

## Morning run-ready checklist

### Environment

- [ ] Python 3.11 environment available for the historical ARC reproduction line
- [ ] CPU PyTorch available
- [ ] `pip install -e '.[external-benchmarks]'` succeeds in a clean environment
- [ ] environment package list captured before/after reproduction
- [ ] source SHA captured in the run manifest

### Dataset boundary

- [ ] `data/manifests/arc_challenge.json` present
- [ ] `scripts/data/download_arc_challenge.py` available
- [ ] only `train validation` requested for frozen ARC scientific lines
- [ ] explicit assertion that ARC test parquet is absent
- [ ] train/validation file hashes recorded in run manifest

### Frozen protocol-v3 full-controls path

Preflight only:

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

Frozen result reproduction command (run only when intentionally reproducing the existing scientific result):

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

Independent verifier:

```bash
python scripts/ci/verify_arc_protocol_v3_full_controls.py \
  --results ci-evidence/arc-protocol-v3-full-controls-validation.json \
  --protocol protocols/arc_challenge_v3.json \
  --train ci-evidence/arc-data/arc-challenge-train.parquet \
  --validation ci-evidence/arc-data/arc-challenge-validation.parquet \
  --report ci-evidence/arc-protocol-v3-full-controls-validation-verification.json \
  | tee ci-evidence/arc-protocol-v3-full-controls-validation-verifier-output.txt
```

### ARC-v5 repaired validation

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

Expected scientific classification remains negative/inconclusive unless a *new, separately frozen* study says otherwise.

### Determinism smoke

A cheap pre-outcome/reproducibility check may replay one deterministic parity step twice and verify exact model tensors/semantic checkpoint state using `scripts/ci/verify_deterministic_training_replay.py`. This is an execution/reproducibility check, not a scientific outcome.

### Paper/results surface

Before any publication update:

- [ ] `CLAIM_LEDGER.md` agrees with the frozen result
- [ ] `RELEASE_PROVENANCE.md` source/protocol/dataset claims remain accurate
- [ ] `MANUSCRIPT_PROVENANCE.md` traces every reported number/table/figure
- [ ] existing negative-result manuscript remains negative where the evidence is negative
- [ ] regenerated tables/figures consume machine outputs rather than hand-entered values
- [ ] license/authorship/citation/redistribution decisions are explicitly owner-approved rather than invented

## Executor contract

A Percy/LabOS/direct-agent job for this repo must:

1. record source SHA before execution;
2. create a fresh output directory rather than overwrite retained evidence;
3. never download or inspect ARC test for frozen validation protocols;
4. preserve raw outputs even when a verifier fails;
5. run the independent verifier after the producer command;
6. classify result according to the frozen protocol, not desired direction;
7. report `CURRENT_TRUTH`, commands, exit codes, hashes, artifacts and blockers.

## Final morning status target

`RUN_READY / EXTERNAL_VALIDATION`: internal reproduction paths are executable and documented; the frozen negative/inconclusive result is preserved; the next genuinely new evidence should be independent outside reproduction/skeptical review or a separately versioned new hypothesis.