from __future__ import annotations

import argparse
import json
import re
from pathlib import Path

HEX64 = re.compile(r"^[0-9a-f]{64}$")
HEX40 = re.compile(r"^[0-9a-f]{40}$")

EXPECTED_SPLITS = {
    "train": {
        "sha256": "e488c1587ffdcfc8443f916c53488a95cd471c5790e0746c6bfe4cecf20962cb",
        "source_rows": 1119,
        "confirmatory_fresh": False,
        "successor_use": "DEVELOPMENT_ONLY",
    },
    "validation": {
        "sha256": "395a5c88d1580d69855fbaee9450270578df1ad5af6259771cd0a42c20e99f05",
        "source_rows": 299,
        "confirmatory_fresh": False,
        "successor_use": "HISTORICAL_OR_EXPLICIT_DEVELOPMENT_ONLY",
    },
    "test": {
        "sha256": "62f03257e737aed263f55c6abf87c7bb0028a44a6bdd2a26eb1279eb42c1d1e9",
        "source_rows": 1172,
        "confirmatory_fresh": None,
        "successor_use": "PROHIBITED_OLD_LOCKED_TEST",
    },
}


def _load_json(path: Path) -> dict:
    return json.loads(path.read_text(encoding="utf-8"))


def validate_audit(root: Path, audit: dict) -> list[str]:
    errors: list[str] = []

    if audit.get("schema_version") != 1:
        errors.append("schema_version must equal 1")
    if audit.get("audit_id") != "lam-arc-successor-data-freshness-v1":
        errors.append("unexpected audit_id")
    if audit.get("status") != "COMPLETE_PENDING_INDEPENDENT_REVIEW":
        errors.append("status must remain COMPLETE_PENDING_INDEPENDENT_REVIEW before review")
    if audit.get("execution_authorized") is not False:
        errors.append("freshness audit must not authorize execution")
    if audit.get("whole_split_conservative_rule") is not True:
        errors.append("whole_split_conservative_rule must be true")

    cutoff = audit.get("repository_evidence_cutoff")
    if not isinstance(cutoff, str) or not HEX40.fullmatch(cutoff):
        errors.append("repository_evidence_cutoff must be a lowercase 40-hex commit SHA")

    arc = audit.get("arc_challenge")
    if not isinstance(arc, dict):
        errors.append("arc_challenge must be an object")
        return errors
    if arc.get("manifest") != "data/manifests/arc_challenge.json":
        errors.append("arc_challenge.manifest path drift")
    if arc.get("dataset") != "AI2 ARC-Challenge":
        errors.append("arc_challenge.dataset drift")

    splits = arc.get("splits")
    if not isinstance(splits, dict) or set(splits) != set(EXPECTED_SPLITS):
        errors.append("arc_challenge.splits must contain exactly train, validation, and test")
    else:
        for name, expected in EXPECTED_SPLITS.items():
            split = splits[name]
            if not isinstance(split, dict):
                errors.append(f"{name} split must be an object")
                continue
            sha = split.get("sha256")
            if not isinstance(sha, str) or not HEX64.fullmatch(sha):
                errors.append(f"{name}.sha256 must be lowercase 64-hex")
            if sha != expected["sha256"]:
                errors.append(f"{name}.sha256 does not match frozen ARC manifest")
            if split.get("source_rows") != expected["source_rows"]:
                errors.append(f"{name}.source_rows does not match frozen ARC manifest")
            if split.get("confirmatory_fresh") is not expected["confirmatory_fresh"]:
                errors.append(f"{name}.confirmatory_fresh violates the freshness boundary")
            if split.get("successor_use") != expected["successor_use"]:
                errors.append(f"{name}.successor_use violates the successor boundary")

        if splits["train"].get("historical_access") != "ACCESSED":
            errors.append("train must remain classified ACCESSED")
        if splits["train"].get("labels_or_outcomes_inspected") is not True:
            errors.append("train supervised-label exposure must remain recorded")
        if splits["validation"].get("historical_access") != "ACCESSED":
            errors.append("validation must remain classified ACCESSED")
        if splits["validation"].get("labels_or_outcomes_inspected") is not True:
            errors.append("validation outcome exposure must remain recorded")
        if splits["test"].get("historical_access") != "NO_RETAINED_PROJECT_EVIDENCE_OF_DOWNLOAD_OR_EVALUATION":
            errors.append("test access claim must remain bounded to retained project evidence")
        if splits["test"].get("labels_or_outcomes_inspected") is not False:
            errors.append("test labels/outcomes must not be marked inspected without new evidence")

    development = audit.get("successor_development_surface")
    if not isinstance(development, dict):
        errors.append("successor_development_surface must be an object")
    else:
        if development.get("dataset") != "AI2 ARC-Challenge" or development.get("source_split") != "train":
            errors.append("successor development surface must remain ARC-Challenge train")
        if development.get("allowed") is not True:
            errors.append("development surface must be marked allowed")
        if development.get("confirmatory") is not False:
            errors.append("development surface must not be marked confirmatory")
        if development.get("requires_new_internal_split_manifest") is not True:
            errors.append("development surface must require a new internal split manifest")

    confirmatory = audit.get("successor_confirmatory_surface")
    if not isinstance(confirmatory, dict):
        errors.append("successor_confirmatory_surface must be an object")
    else:
        if confirmatory.get("status") != "UNRESOLVED_HARD_BLOCKER":
            errors.append("confirmatory status must remain UNRESOLVED_HARD_BLOCKER")
        if confirmatory.get("dataset") is not None:
            errors.append("confirmatory dataset must remain null until separately frozen")
        if confirmatory.get("old_arc_test_allowed") is not False:
            errors.append("old ARC test must remain prohibited for successor use")
        if confirmatory.get("one_shot_rule_required") is not True:
            errors.append("confirmatory surface must require a one-shot rule")
        if confirmatory.get("independent_preoutcome_review_required") is not True:
            errors.append("confirmatory surface must require independent pre-outcome review")

    disposition = audit.get("blocker_disposition")
    if disposition != {
        "DATA_FRESHNESS_AUDIT": "READY_FOR_INDEPENDENT_REVIEW",
        "CONFIRMATORY_DATASET": "UNRESOLVED",
    }:
        errors.append("blocker_disposition drift")

    evidence = audit.get("evidence")
    if not isinstance(evidence, list) or not evidence:
        errors.append("evidence must be a non-empty path list")
    else:
        for rel in evidence:
            if not isinstance(rel, str) or not rel or rel.startswith("/") or ".." in Path(rel).parts:
                errors.append(f"invalid evidence path: {rel!r}")
                continue
            if not (root / rel).is_file():
                errors.append(f"missing evidence file: {rel}")

    manifest_path = root / "data/manifests/arc_challenge.json"
    if manifest_path.is_file() and isinstance(splits, dict):
        manifest = _load_json(manifest_path)
        files = manifest.get("files", {})
        for name, expected in EXPECTED_SPLITS.items():
            entry = files.get(name, {})
            if entry.get("sha256") != expected["sha256"] or entry.get("rows") != expected["source_rows"]:
                errors.append(f"repository ARC manifest drift for {name}")

    v5_path = root / "protocols/arc_challenge_v5_repaired_validation.json"
    if v5_path.is_file():
        v5 = _load_json(v5_path)
        if "must not be downloaded, opened, evaluated, or used for selection" not in v5.get("dataset", {}).get("test_split_policy", ""):
            errors.append("v5 test-lock policy drift")

    successor_path = root / "protocols/arc_successor_v1_draft.json"
    if successor_path.is_file():
        successor = _load_json(successor_path)
        if successor.get("old_locked_test_access") != "PROHIBITED":
            errors.append("successor old_locked_test_access must remain PROHIBITED")
        if successor.get("execution_authorized") is not False:
            errors.append("repository successor draft must remain non-authorized")
        blockers = successor.get("hard_blockers", [])
        if "CONFIRMATORY_DATASET" not in blockers:
            errors.append("repository successor draft must retain CONFIRMATORY_DATASET blocker")

    ledger_path = root / "CLAIM_LEDGER.md"
    if ledger_path.is_file():
        ledger = ledger_path.read_text(encoding="utf-8")
        if "Locked ARC confirmatory test remains unopened" not in ledger:
            errors.append("claim ledger no longer records the locked-test boundary")

    audit_md = root / "DATA_FRESHNESS_AUDIT.md"
    if audit_md.is_file():
        text = audit_md.read_text(encoding="utf-8")
        required_phrases = (
            "DOES NOT AUTHORIZE HELD-OUT EXECUTION",
            "UNRESOLVED HARD BLOCKER",
            "development-only",
            "old ARC test",
        )
        for phrase in required_phrases:
            if phrase not in text:
                errors.append(f"DATA_FRESHNESS_AUDIT.md missing boundary phrase: {phrase}")

    return errors


def main() -> int:
    parser = argparse.ArgumentParser(description="Verify the ARC successor data-freshness audit.")
    parser.add_argument(
        "--audit",
        type=Path,
        default=Path("protocols/data_freshness_audit_v1.json"),
    )
    parser.add_argument(
        "--root",
        type=Path,
        default=Path(__file__).resolve().parents[2],
    )
    args = parser.parse_args()

    audit_path = args.audit if args.audit.is_absolute() else args.root / args.audit
    audit = _load_json(audit_path)
    errors = validate_audit(args.root, audit)
    if errors:
        for error in errors:
            print(f"ERROR: {error}")
        return 1

    print("ARC successor freshness audit: VERIFIED_FAIL_CLOSED")
    print("confirmatory_dataset: UNRESOLVED_HARD_BLOCKER")
    print("execution_authorized: false")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
