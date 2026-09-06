#!/usr/bin/env python3
"""Fail-closed control-plane entrypoint for the LAM-JEPA ARC successor.

Current repository state permits pre-outcome verification only. The ``execute``
mode exists solely to prove that scientific execution cannot start while the
protocol is draft, the runner is unbound, or any authorization/evidence gate is
missing. This module never imports model, dataset, or training code.
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any, Mapping

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "tools"))

from verify_arc_successor_execution_contract import (  # noqa: E402
    CONTRACT,
    EVIDENCE_SCHEMA,
    PROTOCOL,
    ContractError,
    verify_repository,
)

EXIT_EXECUTION_REFUSED = 4


def _load(path: Path) -> Any:
    with path.open("r", encoding="utf-8") as handle:
        return json.load(handle)


def execution_refusal_reasons(
    protocol: Mapping[str, Any],
    contract: Mapping[str, Any],
    *,
    authorization_receipt: Path | None = None,
    execution_manifest: Path | None = None,
) -> list[str]:
    reasons: list[str] = []
    if protocol.get("status") != "FROZEN":
        reasons.append("protocol_not_frozen")
    if protocol.get("execution_authorized") is not True:
        reasons.append("protocol_execution_not_authorized")
    blockers = protocol.get("hard_blockers")
    if not isinstance(blockers, list) or blockers:
        reasons.append("protocol_hard_blockers_present")

    entrypoint = contract.get("preoutcome_entrypoint")
    if not isinstance(entrypoint, Mapping) or entrypoint.get("scientific_runner_bound") is not True:
        reasons.append("scientific_runner_unbound")

    if contract.get("scientific_execution_authorized") is not True:
        reasons.append("execution_contract_not_authorized")
    if contract.get("outcome_access_authorized") is not True:
        reasons.append("outcome_access_not_authorized")

    if authorization_receipt is None:
        reasons.append("authorization_receipt_missing")
    elif not authorization_receipt.is_file():
        reasons.append("authorization_receipt_not_found")

    if execution_manifest is None:
        reasons.append("execution_manifest_missing")
    elif not execution_manifest.is_file():
        reasons.append("execution_manifest_not_found")

    if protocol.get("old_locked_test_access") != "PROHIBITED" or contract.get("old_locked_test_access") != "PROHIBITED":
        reasons.append("old_locked_test_policy_invalid")

    return reasons


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--mode", choices=("verify-preoutcome", "execute"), default="verify-preoutcome")
    parser.add_argument("--authorization-receipt", type=Path)
    parser.add_argument("--execution-manifest", type=Path)
    parser.add_argument("--report", type=Path)
    args = parser.parse_args(argv)

    try:
        verified = verify_repository()
        protocol = _load(PROTOCOL)
        contract = _load(CONTRACT)
        _ = _load(EVIDENCE_SCHEMA)
    except (ContractError, json.JSONDecodeError, OSError) as exc:
        print(json.dumps({"status": "FAIL_PREOUTCOME_EXECUTION_LOCK", "error": str(exc)}, indent=2, sort_keys=True))
        return 2

    if args.mode == "execute":
        reasons = execution_refusal_reasons(
            protocol,
            contract,
            authorization_receipt=args.authorization_receipt,
            execution_manifest=args.execution_manifest,
        )
        # Current contract deliberately has no scientific runner binding. Even if
        # caller-supplied files exist, this lane must refuse until a separately
        # reviewed change binds the runner and all scientific authorization gates.
        if not reasons:
            reasons.append("scientific_execution_handler_intentionally_absent")
        payload = {
            "status": "SCIENTIFIC_EXECUTION_REFUSED_PREOUTCOME",
            "scientific_execution_started": False,
            "outcome_accessed": False,
            "reasons": sorted(reasons),
        }
        print(json.dumps(payload, indent=2, sort_keys=True))
        return EXIT_EXECUTION_REFUSED

    payload = {
        **verified,
        "entrypoint_mode": "verify-preoutcome",
        "scientific_execution_started": False,
        "outcome_accessed": False,
    }
    text = json.dumps(payload, indent=2, sort_keys=True) + "\n"
    if args.report:
        args.report.parent.mkdir(parents=True, exist_ok=True)
        args.report.write_text(text, encoding="utf-8")
    print(text, end="")
    return 0


if __name__ == "__main__":
    sys.exit(main())
