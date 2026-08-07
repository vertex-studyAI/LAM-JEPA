from __future__ import annotations

import argparse
import json
import subprocess
import sys
from pathlib import Path


def require(condition: bool, message: str) -> None:
    if not condition:
        raise SystemExit(message)


def main() -> None:
    parser = argparse.ArgumentParser(description="Verify ARC protocol-v3 DeBERTa development smoke.")
    parser.add_argument("--results", type=Path, required=True)
    parser.add_argument("--protocol", type=Path, default=Path("protocols/arc_challenge_v3.json"))
    parser.add_argument("--report", type=Path, required=True)
    args = parser.parse_args()

    require(args.results.is_file(), f"results missing: {args.results}")
    require(args.protocol.is_file(), f"protocol missing: {args.protocol}")
    v3 = json.loads(args.protocol.read_text(encoding="utf-8"))
    runtime = ((v3.get("runtime_addendum") or {}).get("strong_pretrained_baseline") or {})

    v2_report = args.report.with_name(args.report.stem + "-v2-base.json")
    subprocess.run(
        [
            sys.executable,
            "scripts/ci/verify_arc_pretrained_v2_deberta.py",
            "--results", str(args.results),
            "--protocol", str(v3.get("base_protocol_file")),
            "--report", str(v2_report),
        ],
        check=True,
    )
    base_report = json.loads(v2_report.read_text(encoding="utf-8"))
    require(base_report.get("verdict") == "PROTOCOL_V2_PRETRAINED_BASELINE_EXECUTION_VERIFIED_ONLY", "base verifier failed")

    payload = json.loads(args.results.read_text(encoding="utf-8"))
    protocol = payload.get("protocol")
    require(isinstance(protocol, dict), "result protocol missing")
    require(protocol.get("frozen_protocol_id") == "lam-jepa-arc-challenge-v3", "result is not bound to v3")
    require(protocol.get("pretrained_model_id") == runtime.get("model"), "v3 model mismatch")
    require(protocol.get("pretrained_model_revision") == runtime.get("revision"), "v3 revision mismatch")
    require(protocol.get("resolved_pretrained_revision") == runtime.get("revision"), "resolved revision mismatch")
    require(protocol.get("pretrained_model_license") == runtime.get("license"), "v3 license mismatch")
    require(int(protocol.get("max_length", 0)) == int(runtime.get("tokenizer_max_length", 0)), "max-length mismatch")
    require(int(protocol.get("tokenizer_max_length", 0)) == int(runtime.get("tokenizer_max_length", 0)), "v3 tokenizer metadata mismatch")
    require(protocol.get("tokenizer_use_fast") is runtime.get("tokenizer_use_fast"), "tokenizer implementation drift")
    require(protocol.get("trust_remote_code") is runtime.get("trust_remote_code"), "remote-code policy drift")
    require(protocol.get("checkpoint_format") == runtime.get("checkpoint_format"), "checkpoint format drift")
    require(protocol.get("minimum_torch_version") == runtime.get("minimum_torch_version"), "torch safety floor drift")
    require(protocol.get("weights_only") is runtime.get("weights_only"), "weights-only policy drift")
    require(protocol.get("test_split_policy") == "not downloaded or evaluated by this development command", "locked-test boundary weakened")

    report = {
        "verdict": "PROTOCOL_V3_PRETRAINED_BASELINE_EXECUTION_VERIFIED_ONLY",
        "protocol_id": v3["protocol_id"],
        "base_verifier_verdict": base_report["verdict"],
        "model_id": runtime["model"],
        "model_revision": runtime["revision"],
        "tokenizer_max_length": runtime["tokenizer_max_length"],
        "locked_test_evaluated": False,
        "independent_reproduction": False,
        "claim_boundary_preserved": True,
    }
    args.report.parent.mkdir(parents=True, exist_ok=True)
    args.report.write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
