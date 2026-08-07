from __future__ import annotations

import argparse
import json
import subprocess
from pathlib import Path


def require(condition: bool, message: str) -> None:
    if not condition:
        raise SystemExit(message)


def main() -> None:
    parser = argparse.ArgumentParser(description="Verify ARC protocol-v3 pre-test dataset/runtime addenda and immutable v2 base.")
    parser.add_argument("--protocol", type=Path, default=Path("protocols/arc_challenge_v3.json"))
    parser.add_argument("--report", type=Path, default=Path("ci-evidence/arc-protocol-v3-verification.json"))
    args = parser.parse_args()

    v3 = json.loads(args.protocol.read_text(encoding="utf-8"))
    require(v3.get("schema_version") == 3, "unsupported v3 schema")
    require(v3.get("protocol_id") == "lam-jepa-arc-challenge-v3", "wrong v3 protocol id")
    require(v3.get("supersedes") == "lam-jepa-arc-challenge-v2", "v3 must supersede v2")
    require(v3.get("status") == "FROZEN_BEFORE_CONFIRMATORY_TEST", "v3 must remain frozen pre-test")
    require(v3.get("confirmatory_test_accessed") is False, "confirmatory-test boundary violated")

    base_path = Path(str(v3.get("base_protocol_file", "")))
    require(base_path.is_file(), "v3 base protocol file missing")
    actual_blob = subprocess.check_output(["git", "hash-object", str(base_path)], text=True).strip()
    expected_blob = str(v3.get("base_protocol_git_blob_sha", ""))
    require(actual_blob == expected_blob, f"v2 base protocol drift: expected={expected_blob} actual={actual_blob}")
    v2 = json.loads(base_path.read_text(encoding="utf-8"))
    require(v2.get("protocol_id") == "lam-jepa-arc-challenge-v2", "unexpected base protocol")
    require(v2.get("status") == "FROZEN_BEFORE_CONFIRMATORY_TEST", "base protocol is not frozen")
    require(((v2.get("models") or {}).get("lam_jepa") or {}).get("answer_head") == "four-choice ARC classifier", "v2 four-choice head changed")

    dataset = v3.get("dataset_addendum") or {}
    require(dataset.get("selection_rule") == "retain rows with exactly four answer choices", "structural filter changed")
    require(dataset.get("decision_basis") == "choice cardinality only; never label, question content, prediction, metric, or outcome", "filter is not label/outcome independent")
    require(dataset.get("applies_to") == ["train", "validation", "test"], "structural filter must apply to every split")
    require(dataset.get("preserve_source_order") is True, "source ordering must be preserved")
    require(dataset.get("renumber_or_substitute_rows") is False, "rows may not be substituted or renumbered")
    budget = dataset.get("training_budget_override") or {}
    require("four-choice filter" in str(budget.get("train_examples", "")), "train budget override missing filter")
    require("four-choice filter" in str(budget.get("validation_examples", "")), "validation budget override missing filter")
    require("four-choice filter" in str(budget.get("test_examples", "")), "test budget override missing filter")

    audit = dataset.get("known_pretest_audit") or {}
    require(audit.get("test_split_accessed") is False, "test split was accessed during pretest audit")
    train = audit.get("train") or {}
    validation = audit.get("validation") or {}
    require((train.get("input_rows"), train.get("retained_rows")) == (1119, 1117), "train structural counts drift")
    require((validation.get("input_rows"), validation.get("retained_rows")) == (299, 295), "validation structural counts drift")
    require(train.get("excluded_rows") == [
        {"id": "NYSEDREGENTS_2004_4_8", "choice_count": 3},
        {"id": "TIMSS_1995_8_N3", "choice_count": 5},
    ], "train excluded-row identity drift")
    require(validation.get("excluded_rows") == [
        {"id": "NYSEDREGENTS_2014_4_4", "choice_count": 3},
        {"id": "NYSEDREGENTS_2014_4_19", "choice_count": 3},
        {"id": "TIMSS_2003_8_pg29", "choice_count": 5},
        {"id": "NYSEDREGENTS_2014_4_28", "choice_count": 3},
    ], "validation excluded-row identity drift")
    require(train.get("excluded_identity_digest") == "860fe5ba2cc99943a0fa7e0e679d63c7885038dedba2e0166f91e632b8e2b905", "train exclusion digest drift")
    require(validation.get("excluded_identity_digest") == "3ff837fd582144446de95786bcec2e6b3e0a555086843470a92411df6bef8a8a", "validation exclusion digest drift")

    strong_v2 = ((v2.get("models") or {}).get("strong_pretrained_baseline") or {})
    runtime = ((v3.get("runtime_addendum") or {}).get("strong_pretrained_baseline") or {})
    for key in ("model", "revision", "license"):
        require(runtime.get(key) == strong_v2.get(key), f"v3 changed pretrained scientific identity: {key}")
    require(runtime.get("model") == "microsoft/deberta-v3-xsmall", "unexpected DeBERTa model")
    require(runtime.get("revision") == "14809e4f1fe1895fcba8b258271a940c6ca45ec4", "unexpected DeBERTa revision")
    require(runtime.get("license") == "MIT", "unexpected DeBERTa license")
    require(runtime.get("transformers_version") == "4.57.6", "transformers pin drift")
    require(runtime.get("sentencepiece_version") == "0.2.2", "SentencePiece pin drift")
    require(runtime.get("tokenizer_max_length") == 96, "tokenizer max length drift")
    require(runtime.get("tokenizer_use_fast") is False, "slow SentencePiece tokenizer requirement removed")
    require(runtime.get("trust_remote_code") is False, "remote-code execution must remain disabled")
    require(runtime.get("checkpoint_format") == "pytorch_bin", "checkpoint format drift")
    require(runtime.get("minimum_torch_version") == "2.6.0", "safe torch floor drift")
    require(runtime.get("weights_only") is True, "weights-only loader requirement removed")

    reason = str(v3.get("change_reason", ""))
    require("before any ARC test access" in reason, "pre-test correction rationale missing")
    require("four-choice" in reason and "runtime" in reason, "both v3 correction reasons must be explicit")
    rule = str(v3.get("protocol_change_rule", ""))
    require("V1 and V2 remain immutable" in rule, "prior protocol audit trail not protected")
    require("new protocol version" in rule, "future material changes must be versioned")

    report = {
        "status": "passed",
        "protocol_id": v3["protocol_id"],
        "base_protocol_id": v2["protocol_id"],
        "base_protocol_git_blob_sha": actual_blob,
        "scientific_contract_source": str(base_path),
        "dataset_filter": dataset["selection_rule"],
        "train_retained_rows": train["retained_rows"],
        "validation_retained_rows": validation["retained_rows"],
        "confirmatory_test_accessed": False,
        "pretrained_model": runtime["model"],
        "pretrained_revision": runtime["revision"],
        "tokenizer_max_length": runtime["tokenizer_max_length"],
    }
    args.report.parent.mkdir(parents=True, exist_ok=True)
    args.report.write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
