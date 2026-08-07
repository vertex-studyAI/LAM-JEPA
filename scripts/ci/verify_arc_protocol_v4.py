from __future__ import annotations

import argparse
import json
import subprocess
from pathlib import Path


def require(condition: bool, message: str) -> None:
    if not condition:
        raise SystemExit(message)


def main() -> None:
    parser = argparse.ArgumentParser(description="Verify ARC protocol-v4 runtime addendum and immutable v3 base.")
    parser.add_argument("--protocol", type=Path, default=Path("protocols/arc_challenge_v4.json"))
    parser.add_argument("--report", type=Path, default=Path("ci-evidence/arc-protocol-v4-verification.json"))
    args = parser.parse_args()

    v4 = json.loads(args.protocol.read_text(encoding="utf-8"))
    require(v4.get("schema_version") == 4, "unsupported v4 schema")
    require(v4.get("protocol_id") == "lam-jepa-arc-challenge-v4", "wrong v4 protocol id")
    require(v4.get("supersedes") == "lam-jepa-arc-challenge-v3", "v4 must supersede v3")
    require(v4.get("status") == "FROZEN_BEFORE_CONFIRMATORY_TEST", "v4 must remain frozen pre-test")
    require(v4.get("confirmatory_test_accessed") is False, "confirmatory-test boundary violated")

    base_path = Path(str(v4.get("base_protocol_file", "")))
    require(base_path.is_file(), "v4 base protocol file missing")
    actual_blob = subprocess.check_output(["git", "hash-object", str(base_path)], text=True).strip()
    expected_blob = str(v4.get("base_protocol_git_blob_sha", ""))
    require(actual_blob == expected_blob, f"v3 base protocol drift: expected={expected_blob} actual={actual_blob}")
    v3 = json.loads(base_path.read_text(encoding="utf-8"))
    require(v3.get("protocol_id") == "lam-jepa-arc-challenge-v3", "unexpected base protocol")
    require(v3.get("status") == "FROZEN_BEFORE_CONFIRMATORY_TEST", "base protocol is not frozen")

    strong_v3 = ((v3.get("models") or {}).get("strong_pretrained_baseline") or {})
    runtime = ((v4.get("runtime_addendum") or {}).get("strong_pretrained_baseline") or {})
    for key in ("model", "revision", "license"):
        require(runtime.get(key) == strong_v3.get(key), f"v4 changed pretrained scientific identity: {key}")
    require(runtime.get("model") == "microsoft/deberta-v3-xsmall", "unexpected DeBERTa model")
    require(runtime.get("revision") == "14809e4f1fe1895fcba8b258271a940c6ca45ec4", "unexpected DeBERTa revision")
    require(runtime.get("license") == "MIT", "unexpected DeBERTa license")
    require(runtime.get("transformers_version") == "4.57.6", "transformers pin drift")
    require(runtime.get("sentencepiece_version") == "0.2.2", "SentencePiece pin drift")
    require(runtime.get("tokenizer_max_length") == 96, "tokenizer max length drift")
    require(runtime.get("tokenizer_use_fast") is False, "slow tokenizer requirement removed")
    require(runtime.get("trust_remote_code") is False, "remote code must remain disabled")
    require(runtime.get("checkpoint_format") == "pytorch_bin", "checkpoint format drift")
    require(runtime.get("minimum_torch_version") == "2.6.0", "safe Torch floor drift")
    require(runtime.get("weights_only") is True, "weights-only loader requirement removed")

    reason = str(v4.get("change_reason", ""))
    require("before any confirmatory ARC test access" in reason, "pre-test correction rationale missing")
    require("already-verified development DeBERTa path" in reason, "runtime settings must be evidence-backed, not newly tuned")
    rule = str(v4.get("protocol_change_rule", ""))
    require("V1, V2, and V3 remain immutable" in rule, "prior protocol audit trails are not protected")
    require("new protocol version" in rule, "future material changes must be versioned")

    report = {
        "status": "passed",
        "protocol_id": v4["protocol_id"],
        "base_protocol_id": v3["protocol_id"],
        "base_protocol_git_blob_sha": actual_blob,
        "runtime_only_addendum": True,
        "confirmatory_test_accessed": False,
        "model": runtime["model"],
        "revision": runtime["revision"],
        "tokenizer_max_length": runtime["tokenizer_max_length"],
        "weights_only": runtime["weights_only"],
    }
    args.report.parent.mkdir(parents=True, exist_ok=True)
    args.report.write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
