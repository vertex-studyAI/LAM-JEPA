#!/usr/bin/env python3
"""Fail-closed static verifier for the frozen ARC reproduction handoff.

This script does not run experiments. It checks that the run-ready handoff keeps
required provenance, locked-test, raw-evidence, verifier, and negative-result
boundaries explicit before the packet is treated as reproduction-ready.
"""
from __future__ import annotations

import argparse
from pathlib import Path

REQUIRED_PHRASES = (
    "record source SHA",
    "ARC test",
    "preserve raw outputs",
    "independent verifier",
    "frozen negative/inconclusive",
    "No metric/seed/budget switching",
    "fresh output directory",
)

FORBIDDEN_FALSE_GREEN = (
    "ARC test may be used",
    "discard failed seeds",
    "retune the frozen result",
    "positive result required",
)


def verify(path: Path) -> list[str]:
    text = path.read_text(encoding="utf-8")
    errors: list[str] = []

    for phrase in REQUIRED_PHRASES:
        if phrase.lower() not in text.lower():
            errors.append(f"missing required boundary: {phrase!r}")

    for phrase in FORBIDDEN_FALSE_GREEN:
        if phrase.lower() in text.lower():
            errors.append(f"forbidden false-green wording present: {phrase!r}")

    # The frozen five-seed control command must remain explicit in the handoff.
    if "--seeds 1 2 3 4 5" not in text:
        errors.append("missing frozen control seed set '--seeds 1 2 3 4 5'")

    # The handoff must explicitly assert absence of the protected test parquet.
    if "test ! -e ci-evidence/arc-data/arc-challenge-test.parquet" not in text:
        errors.append("missing executable protected-test absence assertion")

    return errors


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "document",
        nargs="?",
        default="RUN_READY_FREEZE_20260825.md",
        help="run-ready handoff document to verify",
    )
    args = parser.parse_args()

    path = Path(args.document)
    if not path.is_file():
        print(f"FAIL: missing document: {path}")
        return 2

    errors = verify(path)
    if errors:
        print("FAIL: run-ready freeze boundary check failed")
        for error in errors:
            print(f"- {error}")
        return 1

    print(f"PASS: {path} preserves the required frozen-reproduction boundaries")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
