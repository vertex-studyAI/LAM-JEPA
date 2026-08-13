# Reproduce LAM-JEPA

This document separates four evidence classes: fast execution smoke, deterministic same-seed replay after the seed-order repair, the frozen five-seed ARC scientific validation, and an independent retained-artifact checker. Do not substitute one for another.

## 1. Revisions and claim boundary

The software reproducibility repair merged at:

```bash
git checkout b72a97a99769b278eb8ec75bc5eab62dc9599f29
git status --short
git rev-parse HEAD
```

The frozen ARC scientific head is:

```text
760aa7f9a73a177d5ff4ba7eb470f7e68ace63cb
```

Do not reinterpret the software seed-order fix as a new scientific ARC result. Do not change model, data, thresholds, seeds, splits, or evaluation policy after seeing a result. The locked ARC test must remain unused for this failed hypothesis line.

## 2. CPU environment

Equivalent setup for repository verification:

```bash
python -m pip install --upgrade pip
python -m pip install torch --index-url https://download.pytorch.org/whl/cpu
python -m pip install -e '.[external-benchmarks]'
python -c 'import torch; assert not torch.cuda.is_available()'
python -m compileall -q src scripts
```

Attempt 4 of the retained full-controls workflow used Ubuntu 24.04.4 LTS, Python 3.11.15, PyTorch 2.13.0+cpu and NumPy 2.4.6 on CPU.

## 3. Frozen ARC data boundary

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

The final assertion is a scientific stop rule, not optional housekeeping.

Known retained source-file digests from attempt 4:

```text
train parquet      e488c1587ffdcfc8443f916c53488a95cd471c5790e0746c6bfe4cecf20962cb
validation parquet 395a5c88d1580d69855fbaee9450270578df1ad5af6259771cd0a42c20e99f05
```

## 4. Full frozen five-seed ARC controls validation

From scientific revision `760aa7f9a73a177d5ff4ba7eb470f7e68ace63cb`, execute exactly:

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

Then independently verify:

```bash
python scripts/ci/verify_arc_protocol_v3_full_controls.py \
  --results ci-evidence/arc-protocol-v3-full-controls-validation.json \
  --protocol protocols/arc_challenge_v3.json \
  --train ci-evidence/arc-data/arc-challenge-train.parquet \
  --validation ci-evidence/arc-data/arc-challenge-validation.parquet \
  --report ci-evidence/arc-protocol-v3-full-controls-validation-verification.json \
  | tee ci-evidence/arc-protocol-v3-full-controls-validation-verifier-output.txt
```

Required budget assertions:

- seeds exactly `[1, 2, 3, 4, 5]`;
- 20 epochs;
- batch size 32;
- learning rate 0.0003;
- model steps 1;
- all 1,117 eligible train rows;
- all 295 eligible validation rows;
- locked test not evaluated;
- verifier verdict `PROTOCOL_V3_FULL_CONTROLS_VALIDATION_VERIFIED`.

## 5. Retained independent reruns

All three retained reruns below conclude successfully:

| Attempt | Job | Artifact | Digest |
|---:|---:|---:|---|
| 2 | `94178988063` | `9149336081` | `sha256:c45710b5dae6a767ccb6bab7f6e3d8e9578752d8cf9b79fd82a65ae824dded1b` |
| 3 | `94291056903` | `9162165932` | `sha256:caa898f1ff046a337db9b5ddbffe1b332943a732868e2fd809abeda8ee89c30b` |
| 4 | `94302727334` | `9163503934` | `sha256:14c315cd64b2b96d48af4b865bca700a101ea66842a78f35382a5f408805b10a` |

The historical workflow run ID is `31203337502`.

### Provenance note for attempt 4

The run is a historical `pull_request` workflow run. `actions/checkout` checked out merge ref SHA:

```text
ed81a16c5b2e3379eb37c4d94a79941d0cd0ff10
```

rather than the head SHA as a literal checkout. A GitHub compare from `760aa7f9a73a177d5ff4ba7eb470f7e68ace63cb` to `ed81a16c5b2e3379eb37c4d94a79941d0cd0ff10` reports zero changed files. Retain both SHAs in provenance. Attempt 4 is therefore tree-equivalent to the scientific head, but do not hide the literal checkout SHA.

## 6. Expected frozen scientific result

- full LAM-JEPA validation accuracy: `0.2549152493 ± 0.0129968006`, `n=5`;
- `no_planner`: `0.2501694888 ± 0.0129968006`, `n=5`;
- `no_target`: `0.2616949081 ± 0.0203953938`, `n=5`;
- shuffled-label control: `0.2630508393 ± 0.0145011803`, `n=5`, pass under frozen `0.35` ceiling;
- full − `no_planner`: `+0.0047457606`, bootstrap 95% CI `[0.0, 0.0142372817]`;
- full − `no_target`: `-0.0067796588`, bootstrap 95% CI `[-0.0135593176, 0.0]`.

The separately retained capacity-matched supervised baseline remains stronger in mean accuracy. Do not report LAM-JEPA superiority or validated planner/target benefit.

## 7. Independent retained-artifact checker

Download the retained attempt-4 artifact into an empty directory. With GitHub CLI, one suitable command is:

```bash
mkdir -p /tmp/lam-jepa-attempt4
cd /tmp/lam-jepa-attempt4
gh run download 31203337502 \
  --repo vertex-studyAI/LAM-JEPA \
  --name arc-protocol-v3-full-controls-validation
```

If GitHub returns more than one artifact with that repeated name, select artifact ID `9163503934` explicitly through the GitHub Actions UI/API and verify its digest before analysis.

Then recompute the accuracy summaries directly from the raw retained records:

```bash
python - <<'PY'
import json
from pathlib import Path
from statistics import mean, stdev

p = Path('arc-protocol-v3-full-controls-validation.json')
raw = json.loads(p.read_text())

expected = {
    'full': (0.2549152493476868, 0.01299680055624953),
    'no_planner': (0.25016948878765105, 0.01299680055624953),
    'no_target': (0.26169490814208984, 0.02039539375324249),
}

for name, target in expected.items():
    values = [r['metrics']['accuracy'] for r in raw['variants'][name]['records']]
    got = (mean(values), stdev(values))
    assert got == target, (name, got, target)
    assert raw['variants'][name]['accuracy']['n'] == 5
    assert raw['variants'][name]['accuracy']['mean'] == got[0]
    assert raw['variants'][name]['accuracy']['std'] == got[1]
    print(name, values, got)

neg = [r['metrics']['accuracy'] for r in raw['negative_control']['records']]
neg_got = (mean(neg), stdev(neg))
neg_expected = (0.263050839304924, 0.014501180290680909)
assert neg_got == neg_expected, (neg_got, neg_expected)
assert raw['negative_control']['pass'] is True
print('negative_control', neg, neg_got)
PY
```

The independent audit performed on 13 August 2026 reproduced these stored summaries exactly.

The audit and all attempt-4 per-file SHA-256 hashes are retained at:

```text
experiments/repro_wave_2026_08_13/INDEPENDENT_AUDIT.md
experiments/repro_wave_2026_08_13/independent_audit.json
```

## 8. Attempt 3 vs attempt 4 artifact behavior

The two artifacts contain the same 10 files. Eight are byte-identical. The raw full-results JSON and normalized-input copy differ only in numeric leaves:

- numeric leaf differences: `36,468`;
- non-numeric leaf differences: `0`;
- maximum observed numeric drift: `0.0007445961236953735`;
- aggregate accuracy summaries: exactly equal;
- paired-effect summaries: exactly equal;
- negative-control summary: exactly equal;
- strict verifier report, verification JSON, and verifier output: byte-identical.

Do not require byte-exact per-example probability tensors across independent runners. Do require exact aggregate conclusions and a passing independent verifier under the frozen tolerance policy.

## 9. Reporting-metadata defect in frozen raw output

The frozen `run_arc_protocol_v3_controls.py` payload contains a stale `protocol.claim_boundary` sentence saying the invocation is “not the final five-seed/20-epoch protocol.” Preserve that raw artifact unchanged.

For the retained full-budget attempts, this sentence is inconsistent with the actual arguments and independent verifier: five seeds, 20 epochs, full eligible train/validation rows, and `final_five_seed_20_epoch_protocol_executed=true`.

Treat this as a non-invalidating reporting-metadata defect. Do not use the stale sentence to override the executable command, protocol fields, or verifier evidence.

## 10. Pre-fix deterministic-training failure

Before the seed-order repair, `train_single.py` instantiated `LAMJEPA` before applying the requested seed. Under nominally identical SHA / command / seed / CPU execution, one-step losses differed:

```text
10.853294372558594
10.34877872467041
```

Retain this as invalidated reproducibility evidence rather than deleting it.

## 11. Deterministic seed-order repair

PR #61 applied the narrow repair: apply the requested seed before model construction while retaining trainer-side seeding for subsequent data/training randomness.

The repair changed no ARC data split, scientific seed set, metric, threshold, architecture, or locked-test policy.

Current machine-readable replay metadata records six independently verified deterministic replay attempts. Within each runner attempt, model state, final metrics and RNG state are exact. Across attempts, final loss `11.704492568969727` and final accuracy `0.0` remain exact, while some secondary floating-point values drift and serialized PyTorch checkpoints are not byte-identical.

Do not claim byte-for-byte checkpoint identity across independent runners.

## 12. Fast smoke path

A fast external benchmark smoke can check plumbing, but not scientific effects:

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
  --out ci-evidence/arc-challenge-smoke.json
```

Smoke success establishes executability only.

## 13. Failure policy

If a future scientific rerun differs:

1. retain the exact command, SHA, literal checkout SHA, environment, seeds, logs, metrics and artifact;
2. classify the difference as environment, data, nondeterminism, evaluator, metadata, provenance, or scientific-result drift;
3. do not alter the frozen threshold or locked-test policy;
4. if a software bug invalidates execution, preserve the old evidence, make the smallest versioned fix, rerun, and distinguish old from new results;
5. do not select only favorable seeds.

## Reporting policy

Report means, sample dispersion, paired deltas, sample count and confidence intervals where available. Do not claim significance without an appropriate predeclared analysis. Preserve negative results, low-order numerical drift, reporting defects and provenance clarifications as part of the evidence trail.
