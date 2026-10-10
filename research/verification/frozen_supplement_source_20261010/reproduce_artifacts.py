"""Reproduce the source-only supplement audit in a fresh output directory.

Requires the two pinned Git revisions and their source objects locally. Executes
the reviewed package producer, never scientific model code or data downloads.
The deliberately invalid witness archives must never be submitted as supplements.
"""
from __future__ import annotations

import argparse
import hashlib
import importlib.util
import json
import os
import subprocess
import sys
import zipfile
from pathlib import Path

ROOT = Path(__file__).resolve().parents[3]
SPEC = importlib.util.spec_from_file_location(
    "retained_source_verifier", ROOT / "scripts/paper/verify_frozen_supplement.py"
)
verifier = importlib.util.module_from_spec(SPEC)
sys.modules[SPEC.name] = verifier
SPEC.loader.exec_module(verifier)


def save_json(path: Path, value: object) -> None:
    with path.open("x", encoding="utf-8") as stream:
        json.dump(value, stream, indent=2, sort_keys=True, allow_nan=False)
        stream.write("\n")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    args.output.mkdir(parents=True, exist_ok=False)
    os.environ.update(GIT_NO_LAZY_FETCH="1", GIT_NO_REPLACE_OBJECTS="1",
                      GIT_ALLOW_PROTOCOL="", GIT_TERMINAL_PROMPT="0")
    reference = verifier.load_reference(ROOT)
    recipe_source = subprocess.check_output(
        ["git", "--no-replace-objects", "-C", str(ROOT), "show",
         f"{verifier.GENERATOR_REVISION}:{verifier.GENERATOR_PATH}"]
    )
    assert hashlib.sha256(recipe_source).hexdigest() == reference.generator_sha256
    producer = {"__file__": str(ROOT / verifier.GENERATOR_PATH), "__name__": "pinned_producer"}
    exec(compile(recipe_source, verifier.GENERATOR_PATH, "exec"), producer)
    producer["ROOT"] = ROOT
    reports = []
    for suffix in ("a", "b"):
        path = args.output / f"canonical-{suffix}.zip"
        producer["build"](path)
        report = verifier.verify_archive(path, ROOT)
        save_json(args.output / f"canonical-{suffix}-verification.json", report)
        reports.append(report)
    assert reports[0]["archive_sha256"] == reports[1]["archive_sha256"]

    witnesses = []
    for kind in ("model", "expected-results"):
        payloads = dict(reference.payloads)
        if kind == "model":
            member = "src/lam_jepa/model.py"
            before, after = b"use_planner: bool = True", b"use_planner: bool = False"
            assert payloads[member].count(before) == 1
            payloads[member] = payloads[member].replace(before, after)
            change = "use_planner default True -> False"
        else:
            member = "EXPECTED_RESULTS.json"
            result = json.loads(payloads[member])
            result["full_lam_jepa_accuracy_mean"] = 0.99
            payloads[member] = verifier._json_bytes(result)
            change = "declared full_lam_jepa_accuracy_mean -> 0.99"
        payloads[verifier.MANIFEST_PATH] = verifier._json_bytes({"files": {
            name: {"sha256": hashlib.sha256(data).hexdigest(), "bytes": len(data)}
            for name, data in sorted(payloads.items()) if name != verifier.MANIFEST_PATH
        }})
        path = args.output / f"INVALID-coherent-{kind}.zip"
        with zipfile.ZipFile(path, "w") as archive:
            for name, data in sorted(payloads.items()):
                archive.writestr(producer["zip_info"](name), data)
        producer["verify"](path)  # The previous internal-manifest check accepts it.
        try:
            verifier.verify_archive(path, ROOT)
        except verifier.VerificationError as exc:
            error = str(exc)
            assert member in error
        else:
            raise AssertionError("the frozen-source falsifier was not rejected")
        witnesses.append({
            "artifact": path.name, "archive_sha256": hashlib.sha256(path.read_bytes()).hexdigest(),
            "member": member, "change": change, "internal_manifest_verifier": "ACCEPTED",
            "frozen_source_verifier": "REJECTED", "rejection": error,
            "invalid_engineering_witness_only": True,
        })
    receipt = {
        "scientific_revision": verifier.SCIENTIFIC_REVISION,
        "generator_revision": verifier.GENERATOR_REVISION,
        "verifier_source_sha256": reports[0]["verifier_source_sha256"],
        "canonical_archive_sha256": reports[0]["archive_sha256"],
        "two_canonical_builds_byte_identical": True,
        "source_file_count": reports[0]["source_file_count"],
        "member_count": reports[0]["member_count"],
        "witnesses": witnesses,
        "training_runs": 0, "protected_outcome_accesses": 0, "paid_compute": 0,
    }
    save_json(args.output / "audit-receipt.json", receipt)
    print(json.dumps(receipt, indent=2))


if __name__ == "__main__":
    main()
