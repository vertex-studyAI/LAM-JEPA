from __future__ import annotations

import argparse
import json
from pathlib import Path

from lam_jepa.benchmarking.arc_challenge import load_arc_split


def require(condition: bool, message: str) -> None:
    if not condition:
        raise SystemExit(message)


def verify_report(report: dict, *, split: str, input_rows: int, retained_rows: int, excluded_rows: list[dict], excluded_digest: str) -> None:
    require(report.get("protocol_id") == "lam-jepa-arc-challenge-v3", f"{split}: wrong protocol id")
    require(report.get("dataset") == "AI2 ARC-Challenge", f"{split}: wrong dataset")
    require(report.get("split") == split, f"{split}: wrong split")
    require(report.get("selection_rule") == "retain rows with exactly four answer choices", f"{split}: filter rule drift")
    require(report.get("decision_basis") == "choice cardinality only", f"{split}: decision basis drift")
    require(report.get("test_split_accessed") is False, f"{split}: confirmatory test access detected")
    require(report.get("input_rows") == input_rows, f"{split}: input row count drift")
    require(report.get("retained_rows") == retained_rows, f"{split}: retained row count drift")
    require(report.get("excluded_rows") == excluded_rows, f"{split}: excluded row identities drift")
    require(report.get("excluded_identity_digest") == excluded_digest, f"{split}: exclusion digest drift")
    require(report.get("preserved_source_order") is True, f"{split}: source order not preserved")
    output_sha = report.get("output_sha256")
    require(isinstance(output_sha, str) and len(output_sha) == 64, f"{split}: invalid filtered output digest")


def verify_filtered(path: Path, *, expected_rows: int, split: str) -> None:
    require(path.is_file(), f"{split}: filtered parquet missing: {path}")
    examples = load_arc_split(path)
    require(len(examples) == expected_rows, f"{split}: filtered parquet row count drift")
    require(all(len(example.choices) == 4 for example in examples), f"{split}: non-four-choice row survived filter")
    ids = [example.item_id for example in examples]
    require(len(ids) == len(set(ids)), f"{split}: duplicate filtered IDs")


def main() -> None:
    parser = argparse.ArgumentParser(description="Verify protocol-v3 train/validation structural filter evidence.")
    parser.add_argument("--protocol", type=Path, default=Path("protocols/arc_challenge_v3.json"))
    parser.add_argument("--train-report", type=Path, required=True)
    parser.add_argument("--validation-report", type=Path, required=True)
    parser.add_argument("--train-filtered", type=Path, required=True)
    parser.add_argument("--validation-filtered", type=Path, required=True)
    parser.add_argument("--report", type=Path, required=True)
    args = parser.parse_args()

    protocol = json.loads(args.protocol.read_text(encoding="utf-8"))
    require(protocol.get("protocol_id") == "lam-jepa-arc-challenge-v3", "wrong protocol")
    require(protocol.get("status") == "FROZEN_BEFORE_CONFIRMATORY_TEST", "protocol is not frozen")
    require(protocol.get("confirmatory_test_accessed") is False, "protocol records confirmatory test access")
    audit = ((protocol.get("dataset_addendum") or {}).get("known_pretest_audit") or {})

    train_report = json.loads(args.train_report.read_text(encoding="utf-8"))
    validation_report = json.loads(args.validation_report.read_text(encoding="utf-8"))
    train_expected = audit.get("train") or {}
    validation_expected = audit.get("validation") or {}

    verify_report(
        train_report,
        split="train",
        input_rows=1119,
        retained_rows=1117,
        excluded_rows=train_expected.get("excluded_rows"),
        excluded_digest=str(train_expected.get("excluded_identity_digest")),
    )
    verify_report(
        validation_report,
        split="validation",
        input_rows=299,
        retained_rows=295,
        excluded_rows=validation_expected.get("excluded_rows"),
        excluded_digest=str(validation_expected.get("excluded_identity_digest")),
    )
    verify_filtered(args.train_filtered, expected_rows=1117, split="train")
    verify_filtered(args.validation_filtered, expected_rows=295, split="validation")

    report = {
        "verdict": "PROTOCOL_V3_STRUCTURAL_FILTER_VERIFIED",
        "protocol_id": protocol["protocol_id"],
        "selection_rule": "retain rows with exactly four answer choices",
        "decision_basis": "choice cardinality only",
        "train_input_rows": 1119,
        "train_retained_rows": 1117,
        "validation_input_rows": 299,
        "validation_retained_rows": 295,
        "confirmatory_test_accessed": False,
        "source_order_preserved": True,
    }
    args.report.parent.mkdir(parents=True, exist_ok=True)
    args.report.write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
