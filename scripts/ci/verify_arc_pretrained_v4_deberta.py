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
    parser = argparse.ArgumentParser(description="Verify the protocol-v4 DeBERTa development smoke.")
    parser.add_argument("--results", type=Path, required=True)
    parser.add_argument("--protocol", type=Path, default=Path("protocols/arc_challenge_v4.json"))
    parser.add_argument("--report", type=Path, required=True)
    args = parser.parse_args()

    require(args.results.is_file(), f"results missing: {args.results}")
    require(args.protocol.is_file(), f"protocol missing: {args.protocol}")
    v4 = json.loads(args.protocol.read_text(encoding="utf-8"))
    runtime = ((v4.get("runtime_addendum") or {}).get("strong_pretrained_baseline") or {})

    base_report = args.report.with_name(args.report.stem + "-v2-evidence.json")
    subprocess.run(
        [
            sys.executable,
            "scripts/ci/verify_arc_pretrained_v2_deberta.py",
            "--results", str(args.results),
            "--protocol", "protocols/arc_challenge_v2.json",
            "--report", str(base_report),
        ],
        check=True,
    )
    base = json.loads(base_report.read_text(encoding="utf-8"))
    require(base.get("verdict") == "PROTOCOL_V2_PRETRAINED_BASELINE_EXECUTION_VERIFIED_ONLY", "base DeBERTa evidence verifier failed")

    payload = json.loads(args.results.read_text(encoding="utf-8"))
    protocol = payload.get("protocol")
    require(isinstance(protocol, dict), "result protocol missing")
    require(protocol.get("frozen_protocol_id") == "lam-jepa-arc-challenge-v4", "result is not bound to v4")
    require(protocol.get("pretrained_model_id") == runtime.get("model"), "model id drift")
    require(protocol.get("pretrained_model_revision") == runtime.get("revision"), "model revision drift")
    require(protocol.get("resolved_pretrained_revision") == runtime.get("revision"), "resolved revision drift")
    require(protocol.get("pretrained_model_license") == runtime.get("license"), "license drift")
    require(int(protocol.get("max_length", 0)) == runtime.get("tokenizer_max_length"), "runner max length drift")
    require(protocol.get("tokenizer_max_length") == runtime.get("tokenizer_max_length"), "v4 max-length metadata drift")
    require(protocol.get("tokenizer_use_fast") is runtime.get("tokenizer_use_fast"), "tokenizer implementation drift")
    require(protocol.get("trust_remote_code") is runtime.get("trust_remote_code"), "remote-code policy drift")
    require(protocol.get("checkpoint_format") == runtime.get("checkpoint_format"), "checkpoint format drift")
    require(protocol.get("minimum_torch_version") == runtime.get("minimum_torch_version"), "Torch safety floor drift")
    require(protocol.get("weights_only") is runtime.get("weights_only"), "weights-only policy drift")
    require(protocol.get("test_split_policy") == "not downloaded or evaluated by this development command", "locked-test boundary weakened")

    report = {
        "verdict": "PROTOCOL_V4_PRETRAINED_RUNTIME_VERIFIED_ONLY",
        "protocol_id": v4["protocol_id"],
        "base_evidence_verdict": base["verdict"],
        "model": runtime["model"],
        "revision": runtime["revision"],
        "tokenizer_max_length": runtime["tokenizer_max_length"],
        "locked_test_evaluated": False,
        "final_five_seed_20_epoch_protocol_executed": False,
        "independent_reproduction": False,
        "research_complete": False,
    }
    args.report.parent.mkdir(parents=True, exist_ok=True)
    args.report.write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
