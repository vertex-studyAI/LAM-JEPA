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
