#!/usr/bin/env python3
"""Fail-closed deterministic development-split freezer for the LAM-JEPA ARC successor.

The successor may use only the historically contaminated ARC-Challenge train split for
development. This tool never touches ARC validation/test, never observes model outcomes,
and never authorizes held-out execution. It deterministically partitions an already
eligibility-filtered ordered ID list after a split seed and exact dev count are frozen.

Partitioning is library-independent: each item is ranked by
SHA256("lam-successor-dev-split-v1\\0" + seed + "\\0" + item_id), with item_id as a
stable tie-breaker. The first dev_count ranked IDs form dev; the remaining IDs form
train. Output lists preserve the original eligible-source order.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import re
import sys
from pathlib import Path
from typing import Any, Mapping, Sequence

HASH_RE = re.compile(r"^[0-9a-f]{64}$")
PLACEHOLDER_RE = re.compile(
    r"(?i)(?:^|[^a-z0-9])(?:tbd|todo|placeholder|unknown|unset|fill[-_ ]?me|null)(?:$|[^a-z0-9])"
)
EXPECTED_SOURCE_SHA256 = "e488c1587ffdcfc8443f916c53488a95cd471c5790e0746c6bfe4cecf20962cb"
EXPECTED_SOURCE_ROWS = 1119
EXPECTED_ELIGIBLE_ROWS = 1117
DOMAIN = "lam-successor-dev-split-v1"


def canonical_sha256(value: Any) -> str:
    raw = json.dumps(value, sort_keys=True, separators=(",", ":"), ensure_ascii=False).encode("utf-8")
    return hashlib.sha256(raw).hexdigest()


def _concrete(value: Any) -> bool:
    return isinstance(value, str) and bool(value.strip()) and PLACEHOLDER_RE.search(value.strip()) is None


def _hash(value: Any) -> bool:
    return isinstance(value, str) and bool(HASH_RE.fullmatch(value.strip()))


def validate_config(config: Any) -> list[str]:
    errors: list[str] = []
    if not isinstance(config, Mapping):
        return ["config must be a JSON object"]

    if config.get("schema_version") != 1:
        errors.append("schema_version must equal 1")
    if config.get("status") != "FROZEN_PREOUTCOME_DEVELOPMENT_ONLY":
        errors.append("status must equal FROZEN_PREOUTCOME_DEVELOPMENT_ONLY")
    if config.get("execution_authorized") is not False:
        errors.append("development split freeze must not authorize execution")
    if config.get("outcomes_observed") is not False:
        errors.append("outcomes_observed must be false before development split freeze")

    source = config.get("source")
    if not isinstance(source, Mapping):
        errors.append("source must be an object")
        source = {}
    if source.get("dataset") != "AI2 ARC-Challenge":
        errors.append("source.dataset must equal AI2 ARC-Challenge")
    if source.get("split") != "train":
        errors.append("source.split must equal train")
    if source.get("source_sha256") != EXPECTED_SOURCE_SHA256:
        errors.append("source.source_sha256 must match the frozen ARC-Challenge train hash")
    if source.get("source_rows") != EXPECTED_SOURCE_ROWS:
        errors.append(f"source.source_rows must equal {EXPECTED_SOURCE_ROWS}")
    if source.get("eligible_rows") != EXPECTED_ELIGIBLE_ROWS:
        errors.append(f"source.eligible_rows must equal {EXPECTED_ELIGIBLE_ROWS}")
    if source.get("eligibility_rule") != "exactly_four_answer_choices":
        errors.append("source.eligibility_rule must equal exactly_four_answer_choices")
    if not _hash(source.get("eligibility_artifact_sha256")):
        errors.append("source.eligibility_artifact_sha256 must be a lowercase 64-hex SHA-256")

    construction = config.get("construction")
    if not isinstance(construction, Mapping):
        errors.append("construction must be an object")
        construction = {}
    if construction.get("method") != "sha256_rank_v1":
        errors.append("construction.method must equal sha256_rank_v1")
    if not _concrete(construction.get("seed")):
        errors.append("construction.seed must be a concrete non-placeholder string")
    dev_count = construction.get("dev_count")
    if isinstance(dev_count, bool) or not isinstance(dev_count, int) or not (1 <= dev_count < EXPECTED_ELIGIBLE_ROWS):
        errors.append(f"construction.dev_count must be an integer in [1, {EXPECTED_ELIGIBLE_ROWS - 1}]")
    if construction.get("label_blind") is not True:
        errors.append("construction.label_blind must be true")
    if construction.get("preserve_source_order_in_outputs") is not True:
        errors.append("construction.preserve_source_order_in_outputs must be true")

    boundary = config.get("boundary")
    if not isinstance(boundary, Mapping):
        errors.append("boundary must be an object")
        boundary = {}
    expected_bool = {
        "development_only": True,
        "arc_validation_used": False,
        "arc_test_used": False,
        "same_partition_for_B0_B1_T1_T2": True,
        "partition_selection_informed_by_treatment_outcomes": False,
    }
    for key, expected in expected_bool.items():
        if boundary.get(key) is not expected:
            errors.append(f"boundary.{key} must be {str(expected).lower()}")
    if not _hash(boundary.get("metric_selection_policy_sha256")):
        errors.append("boundary.metric_selection_policy_sha256 must be a lowercase 64-hex SHA-256")

    return errors


def validate_eligible_ids(ids: Any) -> list[str]:
    errors: list[str] = []
    if not isinstance(ids, list):
        return ["eligible IDs must be a JSON list"]
    if len(ids) != EXPECTED_ELIGIBLE_ROWS:
        errors.append(f"eligible ID list must contain exactly {EXPECTED_ELIGIBLE_ROWS} IDs")
    if any(not _concrete(item) for item in ids):
        errors.append("every eligible ID must be a concrete non-placeholder string")
    if all(isinstance(item, str) for item in ids) and len(set(ids)) != len(ids):
        errors.append("eligible ID list contains duplicates")
    return errors


def rank_key(seed: str, item_id: str) -> tuple[str, str]:
    payload = f"{DOMAIN}\0{seed}\0{item_id}".encode("utf-8")
    return hashlib.sha256(payload).hexdigest(), item_id


def freeze_partition(config: Mapping[str, Any], eligible_ids: Sequence[str]) -> dict[str, Any]:
    seed = str(config["construction"]["seed"])
    dev_count = int(config["construction"]["dev_count"])
    ranked = sorted(eligible_ids, key=lambda item_id: rank_key(seed, item_id))
    dev_members = set(ranked[:dev_count])
    train_ids = [item_id for item_id in eligible_ids if item_id not in dev_members]
    dev_ids = [item_id for item_id in eligible_ids if item_id in dev_members]
    assert len(train_ids) + len(dev_ids) == len(eligible_ids)
    assert len(set(train_ids).intersection(dev_ids)) == 0
    manifest_core = {
        "schema_version": 1,
        "status": "FROZEN_PREOUTCOME_DEVELOPMENT_ONLY",
        "execution_authorized": False,
        "outcomes_observed": False,
        "source": dict(config["source"]),
        "construction": dict(config["construction"]),
        "boundary": dict(config["boundary"]),
        "partition": {
            "eligible_count": len(eligible_ids),
            "train_count": len(train_ids),
            "dev_count": len(dev_ids),
            "overlap_count": 0,
            "coverage_count": len(train_ids) + len(dev_ids),
            "eligible_ids_sha256": canonical_sha256(list(eligible_ids)),
            "train_ids_sha256": canonical_sha256(train_ids),
            "dev_ids_sha256": canonical_sha256(dev_ids),
            "train_ids": train_ids,
            "dev_ids": dev_ids,
        },
    }
    return {
        **manifest_core,
        "freeze_receipt_sha256": canonical_sha256({"development_split_manifest": manifest_core}),
    }


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--eligible-ids", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args(argv)

    config = json.loads(args.config.read_text(encoding="utf-8"))
    eligible_ids = json.loads(args.eligible_ids.read_text(encoding="utf-8"))

    errors = validate_config(config) + validate_eligible_ids(eligible_ids)
    if errors:
        print(json.dumps({"frozen": False, "execution_authorized": False, "errors": errors}, indent=2, sort_keys=True))
        return 2

    manifest = freeze_partition(config, eligible_ids)
    if args.output.exists():
        raise FileExistsError(f"refusing to overwrite existing frozen split manifest: {args.output}")
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(manifest, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    print(
        json.dumps(
            {
                "frozen": True,
                "execution_authorized": False,
                "development_only": True,
                "train_count": manifest["partition"]["train_count"],
                "dev_count": manifest["partition"]["dev_count"],
                "freeze_receipt_sha256": manifest["freeze_receipt_sha256"],
            },
            indent=2,
            sort_keys=True,
        )
    )
    return 0


if __name__ == "__main__":
    sys.exit(main())
