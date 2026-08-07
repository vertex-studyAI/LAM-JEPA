from __future__ import annotations

import argparse
import json
from pathlib import Path


def require(condition: bool, message: str) -> None:
    if not condition:
        raise SystemExit(message)


def main() -> None:
    parser = argparse.ArgumentParser(description="Verify that ARC protocol v3 is a pre-test implementation-alignment correction only.")
    parser.add_argument("--v2", type=Path, default=Path("protocols/arc_challenge_v2.json"))
    parser.add_argument("--v3", type=Path, default=Path("protocols/arc_challenge_v3.json"))
    parser.add_argument("--report", type=Path, default=Path("ci-evidence/arc-protocol-v3-verification.json"))
    args = parser.parse_args()

    v2 = json.loads(args.v2.read_text(encoding="utf-8"))
    v3 = json.loads(args.v3.read_text(encoding="utf-8"))

    require(v2.get("protocol_id") == "lam-jepa-arc-challenge-v2", "unexpected protocol v2 identity")
    require(v3.get("schema_version") == 3, "unsupported protocol v3 schema")
    require(v3.get("protocol_id") == "lam-jepa-arc-challenge-v3", "unexpected protocol v3 identity")
    require(v3.get("supersedes") == "lam-jepa-arc-challenge-v2", "v3 must supersede v2")
    require(v3.get("status") == "FROZEN_BEFORE_CONFIRMATORY_TEST", "v3 must remain frozen before test access")
    reason = str(v3.get("change_reason", ""))
    require("runner still pinned DistilRoBERTa" in reason, "implementation-mismatch correction rationale missing")
    require("No confirmatory ARC test access occurred" in reason, "pre-test correction boundary missing")

    for section in (
        "scientific_question", "claim_scope", "dataset", "training_budget", "metrics",
        "robustness", "negative_control", "ablations", "claim_gate", "artifact_contract",
    ):
        require(v3.get(section) == v2.get(section), f"scientific contract drift outside pretrained runtime: {section}")

    v2_models = v2.get("models") or {}
    v3_models = v3.get("models") or {}
    for model_key in ("lam_jepa", "majority_reference", "matched_capacity_supervised_baseline"):
        require(v3_models.get(model_key) == v2_models.get(model_key), f"non-pretrained model contract drift: {model_key}")

    pretrained_v2 = v2_models.get("strong_pretrained_baseline") or {}
    pretrained_v3 = v3_models.get("strong_pretrained_baseline") or {}
    for key in ("model", "revision", "license", "role", "parameter_matching"):
        require(pretrained_v3.get(key) == pretrained_v2.get(key), f"pretrained scientific identity drift: {key}")
    require(pretrained_v3.get("model") == "microsoft/deberta-v3-xsmall", "v3 pretrained model mismatch")
    require(pretrained_v3.get("revision") == "14809e4f1fe1895fcba8b258271a940c6ca45ec4", "v3 pretrained revision mismatch")
    require(pretrained_v3.get("license") == "MIT", "v3 pretrained license mismatch")
    require(pretrained_v3.get("transformers_version") == "4.57.6", "transformers runtime pin drift")
    require(pretrained_v3.get("sentencepiece_version") == "0.2.2", "sentencepiece runtime pin drift")
    require(pretrained_v3.get("tokenizer_max_length") == 128, "tokenizer max length drift")
    require(pretrained_v3.get("trust_remote_code") is False, "remote code must remain disabled")
    require(pretrained_v3.get("safetensors_required") is True, "safetensors requirement removed")

    change_rule = str(v3.get("protocol_change_rule", ""))
    require("V1 and V2 remain immutable" in change_rule, "prior protocol audit trails are not protected")
    require("V3 supersedes V2 before confirmatory test access" in change_rule, "v3 pre-test supersession missing")
    require("new protocol version" in change_rule, "future protocol changes must be versioned")

    report = {
        "status": "passed",
        "protocol_id": v3["protocol_id"],
        "supersedes": v3["supersedes"],
        "scientific_sections_unchanged_from_v2": True,
        "pretrained_model": pretrained_v3["model"],
        "pretrained_revision": pretrained_v3["revision"],
        "tokenizer_max_length": pretrained_v3["tokenizer_max_length"],
        "confirmatory_test_accessed": False,
    }
    args.report.parent.mkdir(parents=True, exist_ok=True)
    args.report.write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
