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
