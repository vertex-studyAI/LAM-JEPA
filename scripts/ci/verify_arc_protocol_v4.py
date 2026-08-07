from __future__ import annotations

import argparse
import json
import subprocess
from pathlib import Path


def require(condition: bool, message: str) -> None:
    if not condition:
        raise SystemExit(message)


def main() -> None:
    parser = argparse.ArgumentParser(description="Verify ARC v4 is a runtime-only pre-test addendum over immutable v3.")
    parser.add_argument("--protocol", type=Path, default=Path("protocols/arc_challenge_v4.json"))
    parser.add_argument("--report", type=Path, required=True)
    args = parser.parse_args()

    v4 = json.loads(args.protocol.read_text(encoding="utf-8"))
    require(v4.get("protocol_id") == "lam-jepa-arc-challenge-v4", "wrong protocol id")
    require(v4.get("supersedes") == "lam-jepa-arc-challenge-v3", "v4 must supersede v3")
    require(v4.get("status") == "FROZEN_BEFORE_CONFIRMATORY_TEST", "v4 not frozen pre-test")
    require(v4.get("confirmatory_test_accessed") is False, "confirmatory test already accessed")

    base = Path(v4.get("base_protocol_file", ""))
    require(base.is_file(), "v3 base file missing")
    actual_blob = subprocess.check_output(["git", "hash-object", str(base)], text=True).strip()
    require(actual_blob == v4.get("base_protocol_git_blob_sha"), f"v3 drift: {actual_blob}")
    v3 = json.loads(base.read_text(encoding="utf-8"))
    require(v3.get("protocol_id") == "lam-jepa-arc-challenge-v3", "wrong base protocol")

    v3_strong = ((v3.get("models") or {}).get("strong_pretrained_baseline") or {})
    runtime = ((v4.get("runtime_addendum") or {}).get("strong_pretrained_baseline") or {})
    for key in ("model", "revision", "license"):
        require(runtime.get(key) == v3_strong.get(key), f"scientific identity drift: {key}")

    expected = {
        "model": "microsoft/deberta-v3-xsmall",
        "revision": "14809e4f1fe1895fcba8b258271a940c6ca45ec4",
        "license": "MIT",
        "transformers_version": "4.57.6",
        "sentencepiece_version": "0.2.1",
        "tokenizer_max_length": 96,
        "tokenizer_use_fast": False,
        "trust_remote_code": False,
        "checkpoint_format": "pytorch_bin",
        "minimum_torch_version": "2.6.0",
        "weights_only": True,
    }
    require(runtime == expected, f"runtime contract drift: {runtime}")
    reason = str(v4.get("change_reason", ""))
    require("already-verified PR #23 development path" in reason, "runtime values are not tied to prior evidence")
    require("before any confirmatory ARC test access" in reason, "pre-test boundary missing")

    report = {
        "status": "passed",
        "protocol_id": v4["protocol_id"],
        "base_protocol_id": v3["protocol_id"],
        "base_protocol_git_blob_sha": actual_blob,
        "runtime_only_addendum": True,
        "confirmatory_test_accessed": False,
        "runtime": runtime,
    }
    args.report.parent.mkdir(parents=True, exist_ok=True)
    args.report.write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
