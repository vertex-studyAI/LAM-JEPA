from __future__ import annotations

import argparse
import hashlib
import json
import re
import subprocess
import zipfile
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
SCIENTIFIC_REVISION = "760aa7f9a73a177d5ff4ba7eb470f7e68ace63cb"

FIXED_PATHS = (
    "pyproject.toml",
    "data/manifests/arc_challenge.json",
    "protocols/arc_challenge_v3.json",
    "scripts/data/download_arc_challenge.py",
    "scripts/benchmark/run_arc_protocol_v3_controls.py",
    "scripts/benchmark/run_arc_matched_baseline.py",
    "scripts/benchmark/run_arc_matched_baseline_v3.py",
    "scripts/ci/verify_arc_protocol_v3.py",
    "scripts/ci/verify_arc_protocol_v3_controls.py",
    "scripts/ci/verify_arc_protocol_v3_full_controls.py",
    "scripts/ci/verify_arc_matched_baseline.py",
    "scripts/ci/verify_arc_matched_baseline_v3.py",
    "scripts/ci/measure_arc_gradient_capacity.py",
)

BANNED_FRAGMENTS = (
    "vertex-studyai",
    "build-the-future-11",
    "ryan gomez",
    "ryangomez",
    "oakridge",
    "openreview",
    "aistats",
)
EMAIL_RE = re.compile(r"(?i)\\b[A-Z0-9._%+-]+@(gmail|outlook|hotmail|yahoo)\\.[A-Z]{2,}\\b")

README = """# Anonymous reproduction supplement

This archive contains the code and instructions needed to reproduce the paper's
main ARC-Challenge development-validation result without exposing author,
institution, repository, or submission metadata.

## Scientific boundary

- Use ARC-Challenge train and validation only.
- Do not download or evaluate the ARC test split for this failed hypothesis line.
- Use seeds 1, 2, 3, 4, and 5.
- Use 20 epochs, batch size 32, learning rate 0.0003, one model step, and CPU.
- Preserve null/adverse outcomes. Do not change thresholds, seeds, splits, or
  evaluation policy after seeing a result.
- Low-order floating-point probabilities may drift across runners. The retained
  claim is aggregate-result and verifier reproduction, not byte-identical raw
  probabilities or checkpoints.

## Environment

The retained scientific runs used Python 3.11 and CPU PyTorch. Exact physical
CPU model metadata was not retained, so this supplement does not claim
hardware-bitwise reproduction.

Create an isolated environment and install the declared dependencies:

~~~bash
python -m venv .venv
. .venv/bin/activate
python -m pip install --upgrade pip
python -m pip install torch --index-url https://download.pytorch.org/whl/cpu
python -m pip install -e '.[external-benchmarks]'
python -c 'import torch; assert not torch.cuda.is_available()'
~~~

## Obtain the allowed data

~~~bash
mkdir -p work/arc-data
python scripts/data/download_arc_challenge.py \\
  --splits train validation \\
  --out-dir work/arc-data
test ! -e work/arc-data/arc-challenge-test.parquet
~~~

Verify the frozen data/protocol boundary:

~~~bash
python scripts/ci/verify_arc_protocol_v3.py \\
  --protocol protocols/arc_challenge_v3.json \\
  --dataset-manifest data/manifests/arc_challenge.json \\
  --report work/arc-protocol-v3-verification.json
~~~

## Reproduce the five-seed full controls result

~~~bash
python scripts/benchmark/run_arc_protocol_v3_controls.py \\
  --train work/arc-data/arc-challenge-train.parquet \\
  --validation work/arc-data/arc-challenge-validation.parquet \\
  --seeds 1 2 3 4 5 \\
  --epochs 20 \\
  --batch-size 32 \\
  --learning-rate 0.0003 \\
  --model-steps 1 \\
  --train-limit 0 \\
  --validation-limit 0 \\
  --device cpu \\
  --out work/full-controls.json

python scripts/ci/verify_arc_protocol_v3_full_controls.py \\
  --results work/full-controls.json \\
  --protocol protocols/arc_challenge_v3.json \\
  --train work/arc-data/arc-challenge-train.parquet \\
  --validation work/arc-data/arc-challenge-validation.parquet \\
  --report work/full-controls-verification.json
~~~

## Reproduce the capacity-matched supervised comparison

~~~bash
python scripts/benchmark/run_arc_matched_baseline_v3.py \\
  --run-stage validation_stage \\
  --train work/arc-data/arc-challenge-train.parquet \\
  --validation work/arc-data/arc-challenge-validation.parquet \\
  --seeds 1 2 3 4 5 \\
  --epochs 20 \\
  --batch-size 32 \\
  --learning-rate 0.0003 \\
  --model-steps 1 \\
  --device cpu \\
  --match-tolerance 0.01 \\
  --out work/matched-baseline.json

python scripts/ci/verify_arc_matched_baseline.py \\
  --results work/matched-baseline.json \\
  --report work/matched-baseline-base-verification.json

python scripts/ci/verify_arc_matched_baseline_v3.py \\
  --results work/matched-baseline.json \\
  --base-verification work/matched-baseline-base-verification.json \\
  --protocol protocols/arc_challenge_v3.json \\
  --train work/arc-data/arc-challenge-train.parquet \\
  --validation work/arc-data/arc-challenge-validation.parquet \\
  --expected-stage validation_stage \\
  --report work/matched-baseline-v3-verification.json
~~~

Both matched-baseline verifiers must pass. Compare the aggregate outputs against EXPECTED_RESULTS.json. Small
representation-level floating-point differences are acceptable only when the
aggregate scientific conclusion and verifier verdict remain unchanged.

## Interpretation boundary

The expected result is negative/inconclusive for this frozen implementation and
protocol. It does not establish general failure of JEPA methods, vector
quantization, planners, or representation learning. The locked ARC test remains
unused.
"""

EXPECTED = {
    "full_lam_jepa_accuracy_mean": 0.25491525,
    "matched_supervised_accuracy_mean": 0.26644068,
    "paired_lam_minus_matched_mean": -0.01152542,
    "full_minus_no_planner_mean": 0.00474576,
    "full_minus_no_planner_bootstrap_ci95": [0.0, 0.01423729],
    "full_minus_no_target_mean": -0.00677966,
    "full_minus_no_target_bootstrap_ci95": [-0.01355932, 0.0],
    "shuffled_label_accuracy_mean": 0.26305084,
    "shuffled_label_ceiling": 0.35,
    "locked_test_evaluated": False,
    "interpretation": "Frozen superiority and mechanism-attribution gates are not met.",
}


def git_bytes(revision: str, path: str) -> bytes:
    proc = subprocess.run(
        ["git", "show", f"{revision}:{path}"],
        cwd=ROOT,
        check=False,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
    )
    if proc.returncode:
        raise SystemExit(f"missing frozen source path {path}: {proc.stderr.decode(errors='replace')}")
    return proc.stdout


def frozen_python_paths() -> list[str]:
    proc = subprocess.run(
        ["git", "ls-tree", "-r", "--name-only", SCIENTIFIC_REVISION, "src/lam_jepa"],
        cwd=ROOT,
        check=True,
        text=True,
        stdout=subprocess.PIPE,
    )
    return sorted(
        path for path in proc.stdout.splitlines()
        if path.endswith(".py") and "__pycache__" not in path
    )


def scan_text(path: str, data: bytes) -> None:
    try:
        text = data.decode("utf-8")
    except UnicodeDecodeError:
        return
    lower = text.lower()
    for fragment in BANNED_FRAGMENTS:
        if fragment in lower:
            raise SystemExit(f"identity/submission fragment {fragment!r} found in bundle file {path}")
    if EMAIL_RE.search(text):
        raise SystemExit(f"personal email address found in bundle file {path}")


def zip_info(path: str) -> zipfile.ZipInfo:
    info = zipfile.ZipInfo(path, date_time=(1980, 1, 1, 0, 0, 0))
    info.compress_type = zipfile.ZIP_DEFLATED
    info.external_attr = 0o644 << 16
    return info


def build(output: Path) -> None:
    payloads: dict[str, bytes] = {}
    for path in sorted(set(FIXED_PATHS) | set(frozen_python_paths())):
        data = git_bytes(SCIENTIFIC_REVISION, path)
        scan_text(path, data)
        payloads[path] = data

    generated = {
        "README.md": README.encode(),
        "EXPECTED_RESULTS.json": (json.dumps(EXPECTED, indent=2, sort_keys=True) + "\n").encode(),
    }
    for path, data in generated.items():
        scan_text(path, data)
        payloads[path] = data

    manifest = {
        path: {"sha256": hashlib.sha256(data).hexdigest(), "bytes": len(data)}
        for path, data in sorted(payloads.items())
    }
    manifest_bytes = (json.dumps({"files": manifest}, indent=2, sort_keys=True) + "\n").encode()
    scan_text("BUNDLE_MANIFEST.json", manifest_bytes)
    payloads["BUNDLE_MANIFEST.json"] = manifest_bytes

    output.parent.mkdir(parents=True, exist_ok=True)
    with zipfile.ZipFile(output, "w") as archive:
        for path, data in sorted(payloads.items()):
            archive.writestr(zip_info(path), data)

    verify(output)


def verify(output: Path) -> None:
    with zipfile.ZipFile(output, "r") as archive:
        names = archive.namelist()
        if names != sorted(names):
            raise SystemExit("bundle member order is not deterministic")
        required = {
            "README.md",
            "EXPECTED_RESULTS.json",
            "BUNDLE_MANIFEST.json",
            "pyproject.toml",
            "protocols/arc_challenge_v3.json",
            "data/manifests/arc_challenge.json",
            "scripts/data/download_arc_challenge.py",
            "scripts/benchmark/run_arc_protocol_v3_controls.py",
            "scripts/benchmark/run_arc_matched_baseline_v3.py",
            "scripts/ci/verify_arc_protocol_v3_full_controls.py",
            "src/lam_jepa/model.py",
            "src/lam_jepa/benchmarking/arc_challenge.py",
        }
        missing = required - set(names)
        if missing:
            raise SystemExit(f"anonymous bundle missing required files: {sorted(missing)}")
        if any(name.startswith(".git") or "__pycache__" in name or name.endswith(".pyc") for name in names):
            raise SystemExit("anonymous bundle contains forbidden repository/cache metadata")
        for name in names:
            scan_text(name, archive.read(name))

        manifest = json.loads(archive.read("BUNDLE_MANIFEST.json"))
        for name, record in manifest["files"].items():
            data = archive.read(name)
            if hashlib.sha256(data).hexdigest() != record["sha256"] or len(data) != record["bytes"]:
                raise SystemExit(f"manifest mismatch for {name}")


def main() -> None:
    parser = argparse.ArgumentParser(description="Build the anonymous paper reproduction supplement.")
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--verify-only", action="store_true")
    args = parser.parse_args()
    if args.verify_only:
        verify(args.output)
    else:
        build(args.output)
    print(json.dumps({"bundle": str(args.output), "status": "ANONYMOUS_BUNDLE_VERIFIED"}, indent=2))


if __name__ == "__main__":
    main()
