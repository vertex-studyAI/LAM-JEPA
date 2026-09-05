#!/usr/bin/env python3
"""Fail-closed authorization gate for the LAM-JEPA ARC successor study.

The gate never reads model outputs. It only verifies that the separately
versioned successor protocol and its pre-outcome execution manifest are frozen,
hash-bound, independently reviewed, and internally consistent before held-out
treatment evaluation can be authorized.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import re
import sys
from pathlib import Path
from typing import Any, Mapping, Sequence

HASH_RE = re.compile(r"^[0-9a-fA-F]{64}$")
PLACEHOLDER_RE = re.compile(
    r"(?i)(?:^|[^a-z0-9])(?:tbd|todo|placeholder|unknown|unset|fill[-_ ]?me)(?:$|[^a-z0-9])"
)
UNRESOLVED_PROTOCOL_RE = re.compile(
    r"(?im)(?:\bNOT\s+(?:YET\s+)?FROZEN\b|\bNOT\s+AUTHORIZED\b|\bDRAFT_NOT_FROZEN\b|\bCOLLAPSE_THRESHOLDS_TBD\b|\bCONFIRMATORY_DATASET\b[^\n]*\bunresolved\b)"
)
EXPECTED_SYSTEMS = {"B0", "B1", "T1", "T2"}
REQUIRED_PROTOCOL_BLOCKERS = {
    "DATA_FRESHNESS_AUDIT",
    "CONFIRMATORY_DATASET",
    "ENCODER_FAMILY_AND_REVISION",
    "CONTEXT_TARGET_CONSTRUCTION",
    "DELTA_PRIMARY",
    "SEED_WIN_FRACTION",
    "UNCERTAINTY_RULE",
    "COLLAPSE_THRESHOLDS",
    "PARAMETER_MATCH_TOLERANCE",
    "MAX_COMPUTE_RATIO",
    "EXACT_REPRODUCE_COMMAND",
}


def sha256_bytes(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def _concrete_string(value: Any) -> bool:
    if not isinstance(value, str) or not value.strip():
        return False
    return PLACEHOLDER_RE.search(value.strip()) is None


def _valid_hash(value: Any) -> bool:
    return isinstance(value, str) and bool(HASH_RE.fullmatch(value.strip()))


def _require_concrete(errors: list[str], obj: Mapping[str, Any], key: str, path: str) -> None:
    if not _concrete_string(obj.get(key)):
        errors.append(f"{path}.{key} must be a concrete non-placeholder string")


def _require_hash(errors: list[str], obj: Mapping[str, Any], key: str, path: str) -> None:
    if not _valid_hash(obj.get(key)):
        errors.append(f"{path}.{key} must be a 64-hex SHA-256")


def _as_mapping(value: Any, path: str, errors: list[str]) -> Mapping[str, Any]:
    if not isinstance(value, Mapping):
        errors.append(f"{path} must be an object")
        return {}
    return value


def _frozen_seeds(protocol: Mapping[str, Any], errors: list[str]) -> list[int]:
    seeds = protocol.get("seeds", protocol.get("proposed_seeds"))
    if (
        not isinstance(seeds, list)
        or not seeds
        or any(isinstance(seed, bool) or not isinstance(seed, int) for seed in seeds)
        or len(set(seeds)) != len(seeds)
    ):
        errors.append("protocol seeds must be a non-empty list of unique integers")
        return []
    return list(seeds)


def _protocol_errors(protocol_bytes: bytes, protocol: Any) -> tuple[list[str], list[int]]:
    errors: list[str] = []
    if not isinstance(protocol, Mapping):
        return ["protocol JSON must be an object"], []

    if protocol.get("status") != "FROZEN":
        errors.append("protocol.status must equal FROZEN")
    if protocol.get("execution_authorized") is not True:
        errors.append("protocol.execution_authorized must be true")

    blockers = protocol.get("hard_blockers")
    if not isinstance(blockers, list):
        errors.append("protocol.hard_blockers must be an empty list after freeze")
    elif blockers:
        errors.append("protocol.hard_blockers must be empty after freeze")

    integrity = _as_mapping(protocol.get("integrity_rules"), "protocol.integrity_rules", errors)
    required_integrity = {
        "retain_failed_seeds": True,
        "posthoc_seed_exclusion": False,
        "heldout_threshold_tuning": False,
        "architecture_shopping_after_failure": False,
        "secondary_metric_rescue": False,
        "vq_rescue": False,
    }
    for key, expected in required_integrity.items():
        if integrity.get(key) is not expected:
            errors.append(f"protocol.integrity_rules.{key} must be {str(expected).lower()}")

    systems = protocol.get("systems")
    if not isinstance(systems, Mapping) or not EXPECTED_SYSTEMS.issubset(systems.keys()):
        errors.append("protocol.systems must retain B0, B1, T1, and T2")

    raw_text = protocol_bytes.decode("utf-8", errors="replace")
    if UNRESOLVED_PROTOCOL_RE.search(raw_text):
        errors.append("protocol contains unresolved draft/freeze markers")

    resolved = protocol.get("resolved_blockers")
    if not isinstance(resolved, Mapping):
        errors.append("protocol.resolved_blockers must map every former blocker to evidence")
    else:
        missing = sorted(REQUIRED_PROTOCOL_BLOCKERS - set(resolved.keys()))
        if missing:
            errors.append("protocol.resolved_blockers missing: " + ", ".join(missing))
        for key in sorted(REQUIRED_PROTOCOL_BLOCKERS & set(resolved.keys())):
            evidence = resolved.get(key)
            if not isinstance(evidence, Mapping):
                errors.append(f"protocol.resolved_blockers.{key} must be an object")
                continue
            _require_concrete(errors, evidence, "artifact", f"protocol.resolved_blockers.{key}")
            _require_hash(errors, evidence, "sha256", f"protocol.resolved_blockers.{key}")

    return errors, _frozen_seeds(protocol, errors)


def validate_authorization(protocol_bytes: bytes, manifest: Any) -> list[str]:
    errors: list[str] = []
    try:
        protocol = json.loads(protocol_bytes)
    except json.JSONDecodeError as exc:
        return [f"protocol is not valid JSON: {exc.msg}"]

    protocol_errors, protocol_seeds = _protocol_errors(protocol_bytes, protocol)
    errors.extend(protocol_errors)
    if not isinstance(manifest, Mapping):
        return errors + ["manifest JSON must be an object"]

    actual_protocol_hash = sha256_bytes(protocol_bytes)
    manifest_protocol_hash = manifest.get("protocol_sha256")
    if not _valid_hash(manifest_protocol_hash):
        errors.append("manifest.protocol_sha256 must be a 64-hex SHA-256")
    elif manifest_protocol_hash.lower() != actual_protocol_hash:
        errors.append("manifest.protocol_sha256 does not match protocol bytes")

    _require_concrete(errors, manifest, "protocol_version", "manifest")
    for key in (
        "dataset_snapshot_sha256",
        "split_manifest_sha256",
        "context_target_sha256",
        "optimizer_contract_sha256",
        "budget_contract_sha256",
        "environment_sha256",
        "analysis_sha256",
    ):
        _require_hash(errors, manifest, key, "manifest")
    _require_concrete(errors, manifest, "exact_reproduce_command", "manifest")

    encoder = _as_mapping(manifest.get("encoder"), "manifest.encoder", errors)
    for key in ("family", "revision", "runtime"):
        _require_concrete(errors, encoder, key, "manifest.encoder")
    _require_hash(errors, encoder, "tokenizer_sha256", "manifest.encoder")

    systems = manifest.get("systems")
    if not isinstance(systems, Mapping) or set(systems.keys()) != EXPECTED_SYSTEMS:
        errors.append("manifest.systems must contain exactly B0, B1, T1, and T2")
    else:
        for system_id in sorted(EXPECTED_SYSTEMS):
            path = f"manifest.systems.{system_id}"
            binding = _as_mapping(systems.get(system_id), path, errors)
            for key in ("implementation", "revision", "runtime"):
                _require_concrete(errors, binding, key, path)
            _require_hash(errors, binding, "config_sha256", path)

    manifest_seeds = manifest.get("seeds")
    if manifest_seeds != protocol_seeds:
        errors.append("manifest.seeds must exactly match the frozen protocol seed list")

    budget = _as_mapping(manifest.get("matched_budget"), "manifest.matched_budget", errors)
    for key in ("optimizer_steps", "hyperparameter_trials", "accelerator_seconds_ceiling"):
        value = budget.get(key)
        if isinstance(value, bool) or not isinstance(value, (int, float)) or value <= 0:
            errors.append(f"manifest.matched_budget.{key} must be a positive number")

    if manifest.get("outcomes_observed") is not False:
        errors.append("manifest.outcomes_observed must be false before authorization")

    review = _as_mapping(manifest.get("independent_review"), "manifest.independent_review", errors)
    if review.get("approved") is not True:
        errors.append("manifest.independent_review.approved must be true")
    for key in ("reviewer", "reviewed_at"):
        _require_concrete(errors, review, key, "manifest.independent_review")
    _require_hash(errors, review, "artifact_sha256", "manifest.independent_review")

    return errors


def authorization_receipt(protocol_bytes: bytes, manifest: Mapping[str, Any]) -> str:
    canonical = json.dumps(
        {"protocol_sha256": sha256_bytes(protocol_bytes), "manifest": manifest},
        sort_keys=True,
        separators=(",", ":"),
        ensure_ascii=False,
    ).encode("utf-8")
    return sha256_bytes(canonical)


def _load_json(path: Path) -> Any:
    with path.open("r", encoding="utf-8") as handle:
        return json.load(handle)


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--protocol", type=Path, required=True)
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument("--receipt", type=Path)
    args = parser.parse_args(argv)

    protocol_bytes = args.protocol.read_bytes()
    manifest = _load_json(args.manifest)
    errors = validate_authorization(protocol_bytes, manifest)
    if errors:
        print(json.dumps({"authorized": False, "errors": errors}, indent=2, sort_keys=True))
        return 2

    receipt = authorization_receipt(protocol_bytes, manifest)
    output = {"authorized": True, "receipt_sha256": receipt}
    if args.receipt is not None:
        args.receipt.write_text(json.dumps(output, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    print(json.dumps(output, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    sys.exit(main())
