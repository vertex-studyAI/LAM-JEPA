#!/usr/bin/env python3
"""Verify a metadata-only LAM-JEPA successor confirmatory-dataset candidate.

The verifier intentionally cannot freeze or authorize a candidate. It exists to make
candidate selection reproducible while preserving the successor's CONFIRMATORY_DATASET
blocker until independent review and an exact byte receipt are complete.
"""
from __future__ import annotations

import argparse
import json
import re
from pathlib import Path
from typing import Any, Mapping

HEX40 = re.compile(r"^[0-9a-f]{40}$")
HEX64 = re.compile(r"^[0-9a-f]{64}$")
EXPECTED_STATUS = "CANDIDATE_METADATA_ONLY_NOT_APPROVED_NOT_ACCESSED"
EXPECTED_SUCCESSOR = "lam-arc-contextual-successor-v1-draft"
EXPECTED_SOURCE_REPO = "allenai/OpenBookQA"
EXPECTED_SOURCE_COMMIT = "b51971646e9371a61508d9953fc706645e194a71"
EXPECTED_ARCHIVE_SHA256 = "82368cf05df2e3b309c17d162e10b888b4d768fad6e171e0a041954c8553be46"
EXPECTED_HF_COMMIT = "388097ea7776314e93a529163e0fea805b8a6454"
EXPECTED_SPLIT_COUNTS = {"train": 4957, "validation": 500, "test": 500}
EXPECTED_PARQUET_SHA256 = {
    "train": "98148f8a54e62eb862346a75192d5fb824d6cbb68f2f59aecd793d39ecb5cd8b",
    "validation": "35370b9cfee8c1ff325ccc74adc434d12c47ca0ac3244aa87f3fa77069285206",
    "test": "cd5483e366daa230c1c87bbdc512d8b7229f14f6dd04d19fc8b1a3855aaaa8a3",
}


class CandidateVerificationError(ValueError):
    pass


def _load(path: Path) -> dict[str, Any]:
    payload = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(payload, dict):
        raise CandidateVerificationError(f"{path}: expected JSON object")
    return payload


def _require_hex(value: Any, *, length: int, label: str) -> str:
    pattern = HEX40 if length == 40 else HEX64
    if not isinstance(value, str) or pattern.fullmatch(value) is None:
        raise CandidateVerificationError(f"{label} must be exact lowercase {length}-hex")
    return value


def verify_candidate(candidate: Mapping[str, Any], successor: Mapping[str, Any]) -> dict[str, Any]:
    if candidate.get("schema_version") != 1:
        raise CandidateVerificationError("candidate schema_version must equal 1")
    if candidate.get("status") != EXPECTED_STATUS:
        raise CandidateVerificationError(f"candidate status must equal {EXPECTED_STATUS}")
    if candidate.get("resolves_successor_blocker") is not False:
        raise CandidateVerificationError("metadata-only candidate must not resolve successor blocker")
    if candidate.get("execution_authorized") is not False:
        raise CandidateVerificationError("candidate must keep execution_authorized=false")
    if candidate.get("outcome_access_authorized") is not False:
        raise CandidateVerificationError("candidate must keep outcome_access_authorized=false")

    if successor.get("protocol_id") != EXPECTED_SUCCESSOR:
        raise CandidateVerificationError("unexpected successor protocol identity")
    if successor.get("status") != "DRAFT_NOT_FROZEN":
        raise CandidateVerificationError("successor must remain DRAFT_NOT_FROZEN")
    if successor.get("execution_authorized") is not False:
        raise CandidateVerificationError("successor must remain execution_authorized=false")
    blockers = successor.get("hard_blockers")
    if not isinstance(blockers, list) or "CONFIRMATORY_DATASET" not in blockers:
        raise CandidateVerificationError("successor CONFIRMATORY_DATASET blocker must remain unresolved")

    source = candidate.get("official_source")
    if not isinstance(source, Mapping):
        raise CandidateVerificationError("official_source is required")
    if source.get("repository") != EXPECTED_SOURCE_REPO:
        raise CandidateVerificationError("official source repository drift")
    if _require_hex(source.get("repository_commit"), length=40, label="repository_commit") != EXPECTED_SOURCE_COMMIT:
        raise CandidateVerificationError("official source commit drift")
    if _require_hex(source.get("download_script_git_blob_sha1"), length=40, label="download_script_git_blob_sha1") != "69be4903aa6722f9912becaf109ac2c6b5b7b0f9":
        raise CandidateVerificationError("official download-script blob drift")
    if _require_hex(source.get("archive_sha256_from_huggingface_dataset_metadata"), length=64, label="archive sha256") != EXPECTED_ARCHIVE_SHA256:
        raise CandidateVerificationError("archive SHA-256 drift")
    if source.get("archive_bytes_from_huggingface_dataset_metadata") != 1446098:
        raise CandidateVerificationError("archive byte count drift")

    mirror = candidate.get("secondary_mirror_identity_for_crosscheck_only")
    if not isinstance(mirror, Mapping):
        raise CandidateVerificationError("secondary mirror identity is required")
    if _require_hex(mirror.get("huggingface_conversion_commit"), length=40, label="Hugging Face conversion commit") != EXPECTED_HF_COMMIT:
        raise CandidateVerificationError("Hugging Face conversion commit drift")
    if mirror.get("expected_examples") != EXPECTED_SPLIT_COUNTS:
        raise CandidateVerificationError("OpenBookQA split-count metadata drift")
    if mirror.get("parquet_sha256") != EXPECTED_PARQUET_SHA256:
        raise CandidateVerificationError("OpenBookQA parquet SHA-256 metadata drift")

    access = candidate.get("preoutcome_access_policy")
    if not isinstance(access, Mapping):
        raise CandidateVerificationError("preoutcome_access_policy is required")
    if access.get("test_rows_or_answer_keys_opened_for_this_candidate_review") is not False:
        raise CandidateVerificationError("candidate review must not claim test rows/answers were opened")
    if access.get("model_outcomes_generated") is not False:
        raise CandidateVerificationError("candidate review must not contain model outcomes")

    review = candidate.get("required_independent_review_before_freeze")
    if not isinstance(review, list) or len(review) < 5:
        raise CandidateVerificationError("candidate must retain substantive independent-review gates")
    if not any("license" in str(item).lower() or "terms" in str(item).lower() for item in review):
        raise CandidateVerificationError("candidate must retain licensing/usage-term review")
    if not any("overlap" in str(item).lower() or "contamination" in str(item).lower() for item in review):
        raise CandidateVerificationError("candidate must retain overlap/contamination review")

    return {
        "status": "CONFIRMATORY_DATASET_CANDIDATE_VERIFIED_NOT_APPROVED",
        "candidate_status": candidate["status"],
        "successor_confirmatory_dataset_blocker_remains": True,
        "execution_authorized": False,
        "outcome_access_authorized": False,
        "official_source_commit": EXPECTED_SOURCE_COMMIT,
        "archive_sha256": EXPECTED_ARCHIVE_SHA256,
        "mirror_conversion_commit": EXPECTED_HF_COMMIT,
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--candidate",
        type=Path,
        default=Path("protocols/confirmatory_dataset_candidates/openbookqa_v1_sep2018.json"),
    )
    parser.add_argument(
        "--successor",
        type=Path,
        default=Path("protocols/arc_successor_v1_draft.json"),
    )
    args = parser.parse_args()
    result = verify_candidate(_load(args.candidate), _load(args.successor))
    print(result["status"])
    print(f"ARCHIVE_SHA256={result['archive_sha256']}")
    print("CONFIRMATORY_DATASET_BLOCKER_REMAINS=true")


if __name__ == "__main__":
    main()
