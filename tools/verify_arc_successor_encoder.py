from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
from typing import Any


EXPECTED_ARTIFACT_SHA256 = "8d9ea79aa7ee9777d3aad9044b63f1db5813b6e214d2bc7d559539700f09b516"
EXPECTED_REMAINING_BLOCKERS = {
    "DATA_FRESHNESS_AUDIT",
    "CONFIRMATORY_DATASET",
    "COLLAPSE_THRESHOLDS",
    "PARAMETER_MATCH_TOLERANCE",
    "MAX_COMPUTE_RATIO",
    "EXACT_REPRODUCE_COMMAND",
}
EXPECTED = {
    "artifact_id": "lam-jepa-successor-encoder-v1-20260906",
    "status": "FROZEN_PREOUTCOME_ENCODER_IDENTITY_ONLY",
    "repo_id": "distilbert/distilroberta-base",
    "revision": "fb53ab8802853c8e4fbdbcd0529f21fc6f459b2b",
    "checkpoint_sha256": "2b11ca9cf3d2cbb44cc1a93ad96aedc2894231ae6e33e2d2268c3d7b1ff97663",
    "hidden_size": 768,
    "num_hidden_layers": 6,
    "num_attention_heads": 12,
    "experiment_max_length": 512,
}


def verify(path: Path, successor_path: Path | None = None) -> dict[str, Any]:
    artifact_bytes = path.read_bytes()
    digest = hashlib.sha256(artifact_bytes).hexdigest()
    payload = json.loads(artifact_bytes)
    errors: list[str] = []

    if digest != EXPECTED_ARTIFACT_SHA256:
        errors.append(f"encoder artifact byte identity drift: {digest}")
    if payload.get("artifact_id") != EXPECTED["artifact_id"]:
        errors.append("artifact identity drift")
    if payload.get("status") != EXPECTED["status"]:
        errors.append("status drift")
    if payload.get("execution_authorized") is not False:
        errors.append("encoder freeze cannot self-authorize execution")
    if payload.get("outcome_access_authorized") is not False:
        errors.append("encoder freeze cannot authorize outcome access")

    encoder = payload.get("encoder", {})
    if encoder.get("repo_id") != EXPECTED["repo_id"]:
        errors.append("encoder repository drift")
    if encoder.get("revision") != EXPECTED["revision"]:
        errors.append("encoder revision drift")
    if encoder.get("checkpoint_sha256") != EXPECTED["checkpoint_sha256"]:
        errors.append("checkpoint hash drift")
    for key in ("hidden_size", "num_hidden_layers", "num_attention_heads"):
        if encoder.get(key) != EXPECTED[key]:
            errors.append(f"encoder architecture drift: {key}")

    tokenizer = payload.get("tokenizer", {})
    if tokenizer.get("repo_id") != EXPECTED["repo_id"]:
        errors.append("tokenizer repository drift")
    if tokenizer.get("revision") != EXPECTED["revision"]:
        errors.append("tokenizer revision drift")
    if tokenizer.get("experiment_max_length") != EXPECTED["experiment_max_length"]:
        errors.append("sequence-length drift")
    if tokenizer.get("added_tokens") != []:
        errors.append("treatment-specific added tokens are forbidden")
    if tokenizer.get("withheld_marker_policy") is None:
        errors.append("withheld-marker tokenizer policy missing")

    representation = payload.get("representation", {})
    if representation.get("source_layer") != "final_hidden_state":
        errors.append("representation source-layer drift")
    if representation.get("output_dimension") != EXPECTED["hidden_size"]:
        errors.append("representation dimension drift")
    if representation.get("B0_and_T1_supervised_encoder_identity_must_match") is not True:
        errors.append("B0/T1 encoder identity must remain matched")

    matching = payload.get("matching_rules", {})
    for key in (
        "post_outcome_checkpoint_swap",
        "post_outcome_tokenizer_swap",
        "treatment_specific_added_tokens",
        "treatment_specific_sequence_length",
    ):
        if matching.get(key) is not False:
            errors.append(f"forbidden matching-rule drift: {key}")

    receipts = payload.get("external_source_receipts", {})
    if receipts.get("local_byte_receipts_retained") is not False:
        errors.append("pre-download freeze must not falsely claim local byte receipts")

    if successor_path is not None:
        successor = json.loads(successor_path.read_text(encoding="utf-8"))
        if successor.get("status") != "DRAFT_NOT_FROZEN":
            errors.append("successor must remain draft")
        if successor.get("execution_authorized") is not False:
            errors.append("successor must remain non-authorized")
        blockers = set(successor.get("hard_blockers") or [])
        if blockers != EXPECTED_REMAINING_BLOCKERS:
            errors.append("successor remaining hard-blocker set drift")
        if "ENCODER_FAMILY_AND_REVISION" in blockers:
            errors.append("resolved encoder blocker must not remain hard-blocked")
        binding = (successor.get("resolved_blockers") or {}).get("ENCODER_FAMILY_AND_REVISION") or {}
        if binding.get("artifact") != "protocols/arc_successor_v1_encoder.json":
            errors.append("successor encoder artifact binding drift")
        if binding.get("sha256") != digest:
            errors.append("successor encoder SHA-256 binding drift")
        if binding.get("revision") != EXPECTED["revision"]:
            errors.append("successor encoder revision binding drift")
        if binding.get("checkpoint_sha256") != EXPECTED["checkpoint_sha256"]:
            errors.append("successor checkpoint binding drift")
        if binding.get("local_byte_receipts_retained") is not False:
            errors.append("successor must not claim unretained local model bytes")

    if errors:
        raise ValueError("successor encoder freeze invalid: " + "; ".join(errors))
    return payload


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "path",
        nargs="?",
        type=Path,
        default=Path("protocols/arc_successor_v1_encoder.json"),
    )
    parser.add_argument(
        "--successor",
        type=Path,
        default=Path("protocols/arc_successor_v1_draft.json"),
    )
    args = parser.parse_args()
    payload = verify(args.path, args.successor)
    print(
        json.dumps(
            {
                "status": "PASS_PREOUTCOME_ENCODER_FREEZE",
                "artifact_sha256": EXPECTED_ARTIFACT_SHA256,
                "repo_id": payload["encoder"]["repo_id"],
                "revision": payload["encoder"]["revision"],
                "checkpoint_sha256": payload["encoder"]["checkpoint_sha256"],
                "execution_authorized": False,
                "local_byte_receipts_retained": False,
                "remaining_hard_blockers": sorted(EXPECTED_REMAINING_BLOCKERS),
            },
            sort_keys=True,
        )
    )


if __name__ == "__main__":
    main()
