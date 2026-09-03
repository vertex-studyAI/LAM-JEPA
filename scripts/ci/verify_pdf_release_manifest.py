#!/usr/bin/env python3
"""Fail closed on LAM-JEPA PDF provenance or scientific-boundary drift."""

from __future__ import annotations

import hashlib
import json
import math
import subprocess
from pathlib import Path


ROOT = Path(__file__).resolve().parents[2]
MANIFEST_PATH = ROOT / "paper" / "RELEASE_MANIFEST.json"


def sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def require(condition: bool, message: str) -> None:
    if not condition:
        raise SystemExit(message)


def pdf_properties(path: Path) -> dict[str, str]:
    result = subprocess.run(
        ["pdfinfo", str(path)], check=True, capture_output=True, text=True
    )
    properties: dict[str, str] = {}
    for line in result.stdout.splitlines():
        if ":" in line:
            key, value = line.split(":", 1)
            properties[key.strip()] = value.strip()
    return properties


def verify() -> None:
    manifest = json.loads(MANIFEST_PATH.read_text())
    require(manifest["schema_version"] == 1, "unsupported release-manifest schema")
    require(manifest["project"] == "LAM-JEPA", "wrong project identity")
    require(
        manifest["scientific_status"] == "VALID_NEGATIVE_OR_INCONCLUSIVE_VALIDATION",
        "scientific verdict drifted",
    )
    require(
        manifest["submission_status"] == "NO_GO_NEW_SUBMISSION_DEADLINE_PASSED",
        "expired submission target was reopened",
    )

    for relative, expected in manifest["source_files"].items():
        path = ROOT / relative
        require(path.is_file(), f"missing bound source file: {relative}")
        require(path.stat().st_size == expected["bytes"], f"byte drift: {relative}")
        require(sha256(path) == expected["sha256"], f"digest drift: {relative}")

    boundaries = manifest["claim_boundaries"]
    for key in (
        "matched_supervised_comparator_outperformed",
        "planner_benefit_supported",
        "ema_target_benefit_supported",
        "repaired_quantization_benefit_supported",
        "rescue_tuning_authorized",
        "research_complete",
        "preprint_ready",
    ):
        require(boundaries[key] is False, f"prohibited positive boundary: {key}")
    require(
        boundaries["confirmatory_arc_test"] == "locked/unopened",
        "confirmatory ARC test boundary drifted",
    )

    evidence = manifest["workflow_evidence"]
    require(evidence["permanent_archive"] is False, "temporary artifact mislabeled")
    require(evidence["artifact_id"] == 9783283835, "workflow artifact identity drifted")

    bibliography = manifest["bibliography_audit"]
    require(bibliography["entries"] == 8, "bibliography entry count drifted")
    require(
        bibliography["primary_records_verified"] == bibliography["entries"],
        "bibliography verification incomplete",
    )
    require(bibliography["complete"] is True, "bibliography audit not complete")
    for key in (
        "cited_methods_reproduced",
        "cited_methods_budget_matched",
        "cited_methods_executed_as_comparators",
    ):
        require(bibliography[key] is False, f"inflated bibliography claim: {key}")

    audit = manifest["sentence_claim_audit"]
    require(audit["complete"] is True, "sentence claim audit incomplete")
    require(audit["claim_classes"] == 16, "sentence claim class count drifted")
    require(audit["quantitative_claims_reconciled"] is True, "quantitative claims unreconciled")
    for key in (
        "superiority_claim_supported",
        "significance_claim_supported",
        "mechanism_benefit_claim_supported",
        "external_reproduction_claim_supported",
        "novelty_claim_supported",
        "preprint_ready",
    ):
        require(audit[key] is False, f"inflated sentence-audit claim: {key}")

    summary = json.loads((ROOT / "experiments" / "reproducibility-wave-20260812.json").read_text())
    metrics = summary["canonical_metrics"]
    recomputed = {
        "full_minus_matched": metrics["full_lam_jepa_accuracy"]["mean"]
        - metrics["matched_supervised_accuracy"]["mean"],
        "full_minus_no_planner": metrics["full_lam_jepa_accuracy"]["mean"]
        - metrics["no_planner_accuracy"]["mean"],
        "full_minus_no_target": metrics["full_lam_jepa_accuracy"]["mean"]
        - metrics["no_target_accuracy"]["mean"],
        "bounded_pretrained_delta": metrics["bounded_pretrained_characterization"]["lam_jepa_accuracy"]
        - metrics["bounded_pretrained_characterization"]["deberta_accuracy"],
    }
    expected = audit["independent_recomputation"]
    for key, value in recomputed.items():
        require(math.isclose(value, expected[key], abs_tol=1e-12), f"metric recomputation drift: {key}")

    manuscript = (ROOT / "paper" / "main.tex").read_text()
    audit_text = (ROOT / "paper" / "SENTENCE_CLAIM_AUDIT.md").read_text()
    for required in (
        "did not outperform",
        "confirmatory test remains locked",
        "Independent external reproduction remains pending",
    ):
        require(required in manuscript, f"required manuscript boundary missing: {required}")
    for prohibited in ("preprint_ready=true", "test was evaluated", "statistically significant superiority"):
        require(prohibited not in manuscript.lower(), f"prohibited manuscript claim: {prohibited}")
    require("PASS — evidence-bounded negative/inconclusive manuscript" in audit_text, "audit verdict drifted")

    for document in manifest["pdfs"].values():
        path = ROOT / document["path"]
        if not path.exists():
            continue
        require(path.stat().st_size == document["bytes"], f"PDF byte drift: {path}")
        require(sha256(path) == document["sha256"], f"PDF digest drift: {path}")
        properties = pdf_properties(path)
        require(int(properties["Pages"]) == document["pages"], f"page drift: {path}")
        require(properties["Encrypted"].lower() == "no", f"encrypted PDF: {path}")
        require(properties["JavaScript"].lower() == "no", f"JavaScript PDF: {path}")

    print("PDF_RELEASE_MANIFEST_VERIFIED")


if __name__ == "__main__":
    verify()
