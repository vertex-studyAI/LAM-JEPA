from __future__ import annotations

import argparse
import json
from pathlib import Path

REQUIRED_STATUS_FRAGMENTS = (
    "ARC SUPERIORITY HYPOTHESIS UNSUPPORTED",
    "RESEARCH_COMPLETE_FALSE",
    "VALID_NEGATIVE_OR_INCONCLUSIVE_VALIDATION",
    "The locked ARC confirmatory test must not be used to rescue the failed validation hypothesis.",
)

REQUIRED_README_FRAGMENTS = (
    "The ARC-v5 validation result remains negative or inconclusive under the frozen gate.",
    "Passing CI does not upgrade it to external generalization, confirmatory-test evidence, novelty, model superiority, or `RESEARCH_COMPLETE`.",
)

FORBIDDEN_CURRENT_STATE_FRAGMENTS = (
    "ARC SUPERIORITY HYPOTHESIS SUPPORTED",
    "RESEARCH_COMPLETE_TRUE",
    "ARC-v5 validation result establishes superiority",
    "confirmatory test rescued the",
)


def verify_claim_boundary(status_text: str, readme_text: str) -> dict[str, object]:
    missing_status = [fragment for fragment in REQUIRED_STATUS_FRAGMENTS if fragment not in status_text]
    missing_readme = [fragment for fragment in REQUIRED_README_FRAGMENTS if fragment not in readme_text]
    forbidden_hits = [
        fragment
        for fragment in FORBIDDEN_CURRENT_STATE_FRAGMENTS
        if fragment in status_text or fragment in readme_text
    ]

    if missing_status or missing_readme or forbidden_hits:
        problems = {
            "missing_status_fragments": missing_status,
            "missing_readme_fragments": missing_readme,
            "forbidden_current_state_fragments": forbidden_hits,
        }
        raise ValueError(json.dumps(problems, indent=2))

    return {
        "verdict": "RESEARCH_CLAIM_BOUNDARY_VERIFIED",
        "scientific_state": "VALID_NEGATIVE_OR_INCONCLUSIVE_VALIDATION",
        "arc_superiority_supported": False,
        "research_complete": False,
        "confirmatory_test_authorized_for_hypothesis_rescue": False,
        "required_status_fragments": len(REQUIRED_STATUS_FRAGMENTS),
        "required_readme_fragments": len(REQUIRED_README_FRAGMENTS),
    }


def main() -> None:
    parser = argparse.ArgumentParser(description="Verify LAM-JEPA's negative/inconclusive ARC claim boundary.")
    parser.add_argument("--status", type=Path, default=Path("RESEARCH_STATUS.md"))
    parser.add_argument("--readme", type=Path, default=Path("README.md"))
    parser.add_argument("--out", type=Path)
    args = parser.parse_args()

    report = verify_claim_boundary(
        args.status.read_text(encoding="utf-8"),
        args.readme.read_text(encoding="utf-8"),
    )
    rendered = json.dumps(report, indent=2) + "\n"
    if args.out:
        args.out.parent.mkdir(parents=True, exist_ok=True)
        args.out.write_text(rendered, encoding="utf-8")
    print(rendered, end="")


if __name__ == "__main__":
    main()
