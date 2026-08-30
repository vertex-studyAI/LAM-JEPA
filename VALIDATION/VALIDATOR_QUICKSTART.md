# External Validator Quickstart

**Purpose:** make the external reproduction target unambiguous.

There are two different revisions in this validation process. Do not collapse them.

## A. Validator packet revision
The `validation-freeze-20260830` branch freezes the validation instructions, report template, and campaign record. Use it to obtain the current validator documents.

## B. Scientific execution revision
The frozen five-seed ARC scientific experiment itself is tied to this scientific source revision:

```text
760aa7f9a73a177d5ff4ba7eb470f7e68ace63cb
```

**Run the scientific experiment from that exact revision.** Do not substitute the validator-packet branch head for the frozen scientific source SHA.

## Reproduction procedure

```bash
git clone https://github.com/vertex-studyAI/LAM-JEPA.git
cd LAM-JEPA
git checkout 760aa7f9a73a177d5ff4ba7eb470f7e68ace63cb
git status --short
git rev-parse HEAD

python -m pip install --upgrade pip
python -m pip install torch --index-url https://download.pytorch.org/whl/cpu
python -m pip install -e '.[external-benchmarks]'
python -c 'import torch; assert not torch.cuda.is_available()'
python -m compileall -q src scripts

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

python scripts/ci/verify_arc_protocol_v3_full_controls.py \
  --results ci-evidence/arc-protocol-v3-full-controls-validation.json \
  --protocol protocols/arc_challenge_v3.json \
  --train ci-evidence/arc-data/arc-challenge-train.parquet \
  --validation ci-evidence/arc-data/arc-challenge-validation.parquet \
  --report ci-evidence/arc-protocol-v3-full-controls-validation-verification.json \
  | tee ci-evidence/arc-protocol-v3-full-controls-validation-verifier-output.txt
```

## Required execution facts
- seeds exactly `[1,2,3,4,5]`
- 20 epochs
- batch size 32
- learning rate 0.0003
- one model step
- all eligible training and validation rows
- CPU execution
- ARC confirmatory test remains absent/unopened
- verifier expected to emit `PROTOCOL_V3_FULL_CONTROLS_VALIDATION_VERIFIED`

## Expected retained aggregate result
Expected values are supplied only to enable discrepancy detection, not to encourage result matching:

- LAM-JEPA accuracy: about `0.254915 ± 0.012997`
- no-planner: about `0.250169 ± 0.012997`
- no-target: about `0.261695 ± 0.020395`
- shuffled-label control: about `0.263051 ± 0.014501`

A different result must be reported, not tuned away.

## Report
After the run, return a completed copy of `VALIDATION/VALIDATOR_REPORT_TEMPLATE.md` from the validator packet, plus raw/log artifacts when possible.

Verdicts allowed:
- `REPRODUCED`
- `PARTIALLY_REPRODUCED`
- `NOT_REPRODUCED`
- `BLOCKED`
- `METHODS_REVIEW_ONLY`

A failed reproduction is a valid result and must not be suppressed.