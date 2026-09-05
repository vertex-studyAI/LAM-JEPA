#!/usr/bin/env python3
"""Fail-closed pre-outcome freezer for the LAM-JEPA successor confirmatory dataset.

This tool does not choose a dataset and does not authorize model execution. It only
validates that a human-selected confirmatory surface is immutable, task-compatible,
non-overlapping with the successor development surface, not one of the historical ARC
splits, outcome-unobserved, and independently reviewed before producing a deterministic
selection receipt.
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

ARC_SPLIT_HASHES = {
    "train": "e488c1587ffdcfc8443f916c53488a95cd471c5790e0746c6bfe4cecf20962cb",
    "validation": "395a5c88d1580d69855fbaee9450270578df1ad5af6259771cd0a42c20e99f05",
    "test": "62f03257e737aed263f55c6abf87c7bb0028a44a6bdd2a26eb1279eb42c1d1e9",
}
DISALLOWED_DATASET_NAMES = {
    "ai2 arc-challenge",
    "arc-challenge",
    "allenai/ai2_arc arc-challenge",
}


def sha256_bytes(raw: bytes) -> str:
    return hashlib.sha256(raw).hexdigest()


def canonical_sha256(value: Any) -> str:
    raw = json.dumps(value, sort_keys=True, separators=(",", ":"), ensure_ascii=False).encode("utf-8")
    return sha256_bytes(raw)


def _concrete(value: Any) -> bool:
    return isinstance(value, str) and bool(value.strip()) and PLACEHOLDER_RE.search(value.strip()) is None


def _hash(value: Any) -> bool:
    return isinstance(value, str) and bool(HASH_RE.fullmatch(value.strip()))


def _mapping(value: Any, path: str, errors: list[str]) -> Mapping[str, Any]:
    if not isinstance(value, Mapping):
        errors.append(f"{path} must be an object")
        return {}
    return value


def _require_concrete(errors: list[str], obj: Mapping[str, Any], key: str, path: str) -> None:
    if not _concrete(obj.get(key)):
        errors.append(f"{path}.{key} must be a concrete non-placeholder string")


def _require_hash(errors: list[str], obj: Mapping[str, Any], key: str, path: str) -> None:
    if not _hash(obj.get(key)):
        errors.append(f"{path}.{key} must be a lowercase 64-hex SHA-256")


def validate_manifest(manifest: Any) -> list[str]:
    errors: list[str] = []
    if not isinstance(manifest, Mapping):
        return ["manifest must be a JSON object"]

    if manifest.get("schema_version") != 1:
        errors.append("schema_version must equal 1")
    if manifest.get("status") != "FROZEN_PREOUTCOME":
        errors.append("status must equal FROZEN_PREOUTCOME")
    if manifest.get("execution_authorized") is not False:
        errors.append("confirmatory dataset freeze must not authorize execution")
    if manifest.get("outcomes_observed") is not False:
        errors.append("outcomes_observed must be false before the dataset is frozen")

    dataset = _mapping(manifest.get("dataset"), "dataset", errors)
    for key in ("name", "source", "revision", "license", "task_type"):
        _require_concrete(errors, dataset, key, "dataset")
    for key in ("content_sha256", "provenance_sha256"):
        _require_hash(errors, dataset, key, "dataset")

    dataset_name = str(dataset.get("name", "")).strip().casefold()
    if dataset_name in DISALLOWED_DATASET_NAMES or "arc-challenge" in dataset_name:
        errors.append("historical AI2 ARC-Challenge cannot be the successor confirmatory dataset")
    content_hash = str(dataset.get("content_sha256", "")).lower()
    if content_hash in ARC_SPLIT_HASHES.values():
        errors.append("dataset.content_sha256 matches a historical ARC-Challenge split")

    if dataset.get("task_type") != "four_choice_science_multiple_choice":
        errors.append("dataset.task_type must equal four_choice_science_multiple_choice")
    if dataset.get("answer_choice_count") != 4:
        errors.append("dataset.answer_choice_count must equal 4")

    adapter = _mapping(manifest.get("adapter"), "adapter", errors)
    _require_concrete(errors, adapter, "path", "adapter")
    _require_hash(errors, adapter, "sha256", "adapter")
    if adapter.get("label_blind") is not True:
        errors.append("adapter.label_blind must be true")
    if adapter.get("deterministic") is not True:
        errors.append("adapter.deterministic must be true")

    selection = _mapping(manifest.get("selection"), "selection", errors)
    _require_hash(errors, selection, "selected_item_ids_sha256", "selection")
    _require_hash(errors, selection, "selection_rationale_sha256", "selection")
    count = selection.get("item_count")
    if isinstance(count, bool) or not isinstance(count, int) or count <= 0:
        errors.append("selection.item_count must be a positive integer")
    if selection.get("labels_hidden_from_development") is not True:
        errors.append("selection.labels_hidden_from_development must be true")
    if selection.get("selection_completed_before_treatment_outcomes") is not True:
        errors.append("selection.selection_completed_before_treatment_outcomes must be true")

    freshness = _mapping(manifest.get("freshness"), "freshness", errors)
    if freshness.get("project_treatment_family_tuned_on_dataset") is not False:
        errors.append("freshness.project_treatment_family_tuned_on_dataset must be false")
    if freshness.get("project_outcomes_previously_observed") is not False:
        errors.append("freshness.project_outcomes_previously_observed must be false")
    if freshness.get("prior_access_audit_completed") is not True:
        errors.append("freshness.prior_access_audit_completed must be true")
    _require_hash(errors, freshness, "prior_access_audit_sha256", "freshness")

    overlap = _mapping(manifest.get("overlap"), "overlap", errors)
    if overlap.get("development_overlap_count") != 0:
        errors.append("overlap.development_overlap_count must equal 0")
    if overlap.get("historical_arc_overlap_count") != 0:
        errors.append("overlap.historical_arc_overlap_count must equal 0")
    _require_hash(errors, overlap, "audit_sha256", "overlap")
    if overlap.get("semantic_overlap_reviewed") is not True:
        errors.append("overlap.semantic_overlap_reviewed must be true")

    policy = _mapping(manifest.get("one_shot_policy"), "one_shot_policy", errors)
    if policy.get("maximum_confirmatory_runs") != 1:
        errors.append("one_shot_policy.maximum_confirmatory_runs must equal 1")
    if policy.get("hyperparameter_updates_after_access") is not False:
        errors.append("one_shot_policy.hyperparameter_updates_after_access must be false")
    if policy.get("architecture_changes_after_access") is not False:
        errors.append("one_shot_policy.architecture_changes_after_access must be false")
    if policy.get("retain_all_outputs") is not True:
        errors.append("one_shot_policy.retain_all_outputs must be true")
    if policy.get("retain_negative_or_inconclusive_result") is not True:
        errors.append("one_shot_policy.retain_negative_or_inconclusive_result must be true")

    review = _mapping(manifest.get("independent_review"), "independent_review", errors)
    if review.get("approved") is not True:
        errors.append("independent_review.approved must be true")
    for key in ("reviewer", "reviewed_at"):
        _require_concrete(errors, review, key, "independent_review")
    _require_hash(errors, review, "artifact_sha256", "independent_review")
    if review.get("outcomes_available_to_reviewer") is not False:
        errors.append("independent_review.outcomes_available_to_reviewer must be false")

    return errors


def freeze_receipt(manifest: Mapping[str, Any]) -> str:
    return canonical_sha256({"confirmatory_dataset_manifest": manifest})


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument("--receipt", type=Path)
    args = parser.parse_args(argv)

    manifest = json.loads(args.manifest.read_text(encoding="utf-8"))
    errors = validate_manifest(manifest)
    if errors:
        print(json.dumps({"frozen": False, "execution_authorized": False, "errors": errors}, indent=2, sort_keys=True))
        return 2

    receipt = freeze_receipt(manifest)
    output = {
        "frozen": True,
        "execution_authorized": False,
        "confirmatory_dataset_receipt_sha256": receipt,
    }
    if args.receipt is not None:
        if args.receipt.exists():
            raise FileExistsError(f"refusing to overwrite existing receipt: {args.receipt}")
        args.receipt.write_text(json.dumps(output, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    print(json.dumps(output, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    sys.exit(main())
