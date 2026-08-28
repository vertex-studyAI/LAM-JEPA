#!/usr/bin/env python3
"""Fail closed when paper-candidate evidence is misrepresented as integrated main evidence.

This verifier is intentionally scientific-result agnostic. It does not inspect, rerun,
or alter experiments. It only checks git ancestry and the release-status language in the
closure document so a verified PR candidate cannot be silently promoted to main-level
verification before integration.
"""

from __future__ import annotations

import argparse
import subprocess
from pathlib import Path


NON_INTEGRATED_MARKERS = (
    "not yet integrated into `main`",
    "main is **not yet verified",
    "verified on pr #101 candidate",
    "integration pending",
)

FALSE_GREEN_MARKERS = (
    "green — internal/non-owner gates closed now",
    "manuscript / format:** green",
    "reproducibility / claims:** green",
)


def git(*args: str) -> subprocess.CompletedProcess[str]:
    return subprocess.run(
        ["git", *args],
        text=True,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        check=False,
    )


def is_ancestor(candidate: str, target: str) -> bool:
    result = git("merge-base", "--is-ancestor", candidate, target)
    if result.returncode not in (0, 1):
        raise RuntimeError(
            f"git merge-base failed ({result.returncode}): {result.stderr.strip()}"
        )
    return result.returncode == 0


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--candidate",
        default="beb58c7de92b77cccb4cfa2f1b6ee7e1212c9c89",
        help="exact paper-candidate commit that has verified paper/CI evidence",
    )
    parser.add_argument(
        "--target",
        default="HEAD",
        help="revision whose integration status is being claimed (default: HEAD)",
    )
    parser.add_argument(
        "--document",
        default="RESEARCH_WAVE_EXECUTION_20260823.md",
        help="closure/status document to check",
    )
    args = parser.parse_args()

    document = Path(args.document)
    if not document.is_file():
        raise SystemExit(f"missing closure document: {document}")

    text = document.read_text(encoding="utf-8").lower()
    integrated = is_ancestor(args.candidate, args.target)

    if integrated:
        print(
            f"PASS: candidate {args.candidate} is an ancestor of {args.target}; "
            "candidate integration is established by git ancestry."
        )
        return 0

    false_green = [marker for marker in FALSE_GREEN_MARKERS if marker in text]
    boundary_markers = [marker for marker in NON_INTEGRATED_MARKERS if marker in text]

    if false_green:
        raise SystemExit(
            "FAIL: candidate is not integrated but closure document contains main-level "
            f"green language: {false_green}"
        )

    if not boundary_markers:
        raise SystemExit(
            "FAIL: candidate is not integrated and closure document lacks an explicit "
            "non-integration boundary."
        )

    print(
        f"PASS: candidate {args.candidate} is not an ancestor of {args.target}; "
        "closure document explicitly preserves the candidate-vs-main boundary."
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
