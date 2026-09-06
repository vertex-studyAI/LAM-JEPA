import copy
import json
from pathlib import Path

import pytest

from tools.verify_confirmatory_dataset_candidate import (
    CandidateVerificationError,
    EXPECTED_ARCHIVE_SHA256,
    verify_candidate,
)


CANDIDATE_PATH = Path("protocols/confirmatory_dataset_candidates/openbookqa_v1_sep2018.json")
SUCCESSOR_PATH = Path("protocols/arc_successor_v1_draft.json")


def _load(path: Path):
    return json.loads(path.read_text(encoding="utf-8"))


def test_current_openbookqa_candidate_is_metadata_only_and_successor_remains_blocked():
    result = verify_candidate(_load(CANDIDATE_PATH), _load(SUCCESSOR_PATH))
    assert result["status"] == "CONFIRMATORY_DATASET_CANDIDATE_VERIFIED_NOT_APPROVED"
    assert result["successor_confirmatory_dataset_blocker_remains"] is True
    assert result["execution_authorized"] is False
    assert result["outcome_access_authorized"] is False
    assert result["archive_sha256"] == EXPECTED_ARCHIVE_SHA256


def test_candidate_cannot_self_approve_or_authorize():
    successor = _load(SUCCESSOR_PATH)
    for key in ("resolves_successor_blocker", "execution_authorized", "outcome_access_authorized"):
        candidate = _load(CANDIDATE_PATH)
        candidate[key] = True
        with pytest.raises(CandidateVerificationError):
            verify_candidate(candidate, successor)


def test_candidate_rejects_source_or_split_identity_drift():
    successor = _load(SUCCESSOR_PATH)
    candidate = _load(CANDIDATE_PATH)
    candidate["official_source"]["archive_sha256_from_huggingface_dataset_metadata"] = "0" * 64
    with pytest.raises(CandidateVerificationError, match="archive SHA-256 drift"):
        verify_candidate(candidate, successor)

    candidate = _load(CANDIDATE_PATH)
    candidate["secondary_mirror_identity_for_crosscheck_only"]["expected_examples"]["test"] = 499
    with pytest.raises(CandidateVerificationError, match="split-count metadata drift"):
        verify_candidate(candidate, successor)


def test_candidate_rejects_successor_that_silently_drops_confirmatory_blocker():
    candidate = _load(CANDIDATE_PATH)
    successor = copy.deepcopy(_load(SUCCESSOR_PATH))
    successor["hard_blockers"].remove("CONFIRMATORY_DATASET")
    with pytest.raises(CandidateVerificationError, match="CONFIRMATORY_DATASET blocker"):
        verify_candidate(candidate, successor)


def test_candidate_requires_license_and_overlap_review_gates():
    successor = _load(SUCCESSOR_PATH)
    candidate = _load(CANDIDATE_PATH)
    candidate["required_independent_review_before_freeze"] = [
        item
        for item in candidate["required_independent_review_before_freeze"]
        if "license" not in item.lower() and "terms" not in item.lower()
    ]
    with pytest.raises(CandidateVerificationError, match="licensing/usage-term review"):
        verify_candidate(candidate, successor)

    candidate = _load(CANDIDATE_PATH)
    candidate["required_independent_review_before_freeze"] = [
        item
        for item in candidate["required_independent_review_before_freeze"]
        if "overlap" not in item.lower() and "contamination" not in item.lower()
    ]
    with pytest.raises(CandidateVerificationError, match="overlap/contamination review"):
        verify_candidate(candidate, successor)
