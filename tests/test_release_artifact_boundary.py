import hashlib
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]


def test_legacy_pdf_is_exactly_identified_and_quarantined():
    pdf = ROOT / "paper.pdf"
    assert hashlib.sha256(pdf.read_bytes()).hexdigest() == (
        "213e36e5065544757d4a3e4d8da7aab4ee9018280e7617e994671db3199d85ef"
    )
    status = (ROOT / "PAPER_SOURCE_STATUS.md").read_text()
    assert "FORBIDDEN FOR SUBMISSION OR CURRENT SCIENTIFIC CITATION" in status
    assert "MANUSCRIPT_DRAFT_NEGATIVE_ARC.md" in status


def test_expired_icdm_target_fails_closed_without_invented_extension():
    packet = (ROOT / "paper/ICDM_TEEN_SUBMISSION_PACKET_20260829.md").read_text()
    closure = (ROOT / "paper/ICDM_TEEN_DEADLINE_CLOSURE_20260901.md").read_text()
    assert "NO-GO FOR NEW ICDM 2026 TEEN SUBMISSION" in packet
    assert "NO_GO_NEW_SUBMISSION_DEADLINE_PASSED" in closure
    assert "No repository evidence establishes an extension" in closure
    assert "Do not upload this packet to ICDM 2026" in packet


def test_deadline_closure_preserves_negative_and_locked_test_boundaries():
    closure = (ROOT / "paper/ICDM_TEEN_DEADLINE_CLOSURE_20260901.md").read_text()
    assert "did not outperform" in closure
    assert "did not establish" in closure
    assert "locked and unopened" in closure
    assert "may not be accessed to rescue" in closure
    assert "No ARC superiority" in closure
