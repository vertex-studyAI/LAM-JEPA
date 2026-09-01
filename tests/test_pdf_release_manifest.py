import json
import subprocess
import sys
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]


def test_pdf_release_manifest_verifier_passes():
    result = subprocess.run(
        [sys.executable, "scripts/ci/verify_pdf_release_manifest.py"],
        cwd=ROOT,
        check=True,
        capture_output=True,
        text=True,
    )
    assert "PDF_RELEASE_MANIFEST_VERIFIED" in result.stdout


def test_pdf_release_manifest_fails_closed_on_readiness_and_archive_status():
    manifest = json.loads((ROOT / "paper/RELEASE_MANIFEST.json").read_text())
    assert manifest["claim_boundaries"]["preprint_ready"] is False
    assert manifest["workflow_evidence"]["permanent_archive"] is False
    assert manifest["pdfs"]["historical_icdm"]["new_submission_authorized"] is False
    assert manifest["claim_boundaries"]["confirmatory_arc_test"] == "locked/unopened"


def test_bibliography_audit_is_complete_but_not_reproduction_evidence():
    manifest = json.loads((ROOT / "paper/RELEASE_MANIFEST.json").read_text())
    audit = manifest["bibliography_audit"]
    assert audit["entries"] == 8
    assert audit["primary_records_verified"] == 8
    assert audit["complete"] is True
    assert audit["cited_methods_reproduced"] is False
    assert audit["cited_methods_budget_matched"] is False
    assert audit["cited_methods_executed_as_comparators"] is False

    ledger = (ROOT / "paper/BIBLIOGRAPHY_AUDIT.md").read_text()
    for key in (
        "assran2023ijepa",
        "vandenOord2017vqvae",
        "ye2024lapa",
        "garrido2026latentaction",
        "masip2026ffjepa",
        "clark2018arc",
        "he2023debertav3",
        "pineau2021reproducibility",
    ):
        assert f"`{key}`" in ledger
    assert "not reproduced or run as a comparator" in ledger
    assert "confirmatory ARC test remains locked and unopened" in ledger


def test_sentence_claim_audit_recomputes_metrics_and_fails_closed():
    manifest = json.loads((ROOT / "paper/RELEASE_MANIFEST.json").read_text())
    audit = manifest["sentence_claim_audit"]
    assert audit["complete"] is True
    assert audit["claim_classes"] == 16
    assert audit["quantitative_claims_reconciled"] is True
    assert audit["superiority_claim_supported"] is False
    assert audit["significance_claim_supported"] is False
    assert audit["mechanism_benefit_claim_supported"] is False
    assert audit["external_reproduction_claim_supported"] is False
    assert audit["novelty_claim_supported"] is False
    assert audit["preprint_ready"] is False

    metrics = json.loads((ROOT / "experiments/reproducibility-wave-20260812.json").read_text())["canonical_metrics"]
    recomputed = audit["independent_recomputation"]
    assert abs((metrics["full_lam_jepa_accuracy"]["mean"] - metrics["matched_supervised_accuracy"]["mean"]) - recomputed["full_minus_matched"]) < 1e-12
    assert abs((metrics["full_lam_jepa_accuracy"]["mean"] - metrics["no_planner_accuracy"]["mean"]) - recomputed["full_minus_no_planner"]) < 1e-12
    assert abs((metrics["full_lam_jepa_accuracy"]["mean"] - metrics["no_target_accuracy"]["mean"]) - recomputed["full_minus_no_target"]) < 1e-12
    bounded = metrics["bounded_pretrained_characterization"]
    assert abs((bounded["lam_jepa_accuracy"] - bounded["deberta_accuracy"]) - recomputed["bounded_pretrained_delta"]) < 1e-12

    ledger = (ROOT / "paper/SENTENCE_CLAIM_AUDIT.md").read_text()
    assert "PASS — evidence-bounded negative/inconclusive manuscript" in ledger
    assert "No manuscript sentence, scientific value, protocol field, seed, threshold, or conclusion was" in ledger
