from __future__ import annotations

import json
import sys
import unittest
from pathlib import Path
from tempfile import TemporaryDirectory


ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "tools"))

from verify_arc_successor_encoder import EXPECTED, EXPECTED_REMAINING_BLOCKERS, verify  # noqa: E402


ARTIFACT = ROOT / "protocols" / "arc_successor_v1_encoder.json"
SUCCESSOR = ROOT / "protocols" / "arc_successor_v1_draft.json"


class ArcSuccessorEncoderFreezeTests(unittest.TestCase):
    def _verify_mutation(self, payload: dict, pattern: str) -> None:
        with TemporaryDirectory() as tmp:
            path = Path(tmp) / "encoder.json"
            path.write_text(json.dumps(payload), encoding="utf-8")
            with self.assertRaisesRegex(ValueError, pattern):
                verify(path)

    def test_repository_encoder_freeze_verifies_without_authorizing_execution(self) -> None:
        payload = verify(ARTIFACT, SUCCESSOR)
        self.assertFalse(payload["execution_authorized"])
        self.assertFalse(payload["outcome_access_authorized"])
        self.assertFalse(payload["external_source_receipts"]["local_byte_receipts_retained"])
        self.assertEqual(payload["encoder"]["revision"], EXPECTED["revision"])
        self.assertEqual(payload["tokenizer"]["revision"], EXPECTED["revision"])
        successor = json.loads(SUCCESSOR.read_text(encoding="utf-8"))
        self.assertEqual(set(successor["hard_blockers"]), EXPECTED_REMAINING_BLOCKERS)
        self.assertNotIn("ENCODER_FAMILY_AND_REVISION", successor["hard_blockers"])

    def test_checkpoint_or_tokenizer_revision_drift_fails_closed(self) -> None:
        payload = json.loads(ARTIFACT.read_text(encoding="utf-8"))
        payload["encoder"]["revision"] = "main"
        self._verify_mutation(payload, "encoder revision drift")

        payload = json.loads(ARTIFACT.read_text(encoding="utf-8"))
        payload["tokenizer"]["revision"] = "main"
        self._verify_mutation(payload, "tokenizer revision drift")

        payload = json.loads(ARTIFACT.read_text(encoding="utf-8"))
        payload["encoder"]["checkpoint_sha256"] = "0" * 64
        self._verify_mutation(payload, "checkpoint hash drift")

    def test_treatment_specific_token_or_length_change_fails_closed(self) -> None:
        payload = json.loads(ARTIFACT.read_text(encoding="utf-8"))
        payload["tokenizer"]["added_tokens"] = ["[WITHHELD_SPAN]"]
        self._verify_mutation(payload, "added tokens")

        payload = json.loads(ARTIFACT.read_text(encoding="utf-8"))
        payload["matching_rules"]["treatment_specific_sequence_length"] = True
        self._verify_mutation(payload, "treatment_specific_sequence_length")

    def test_encoder_freeze_cannot_self_authorize_or_fake_download_receipts(self) -> None:
        payload = json.loads(ARTIFACT.read_text(encoding="utf-8"))
        payload["execution_authorized"] = True
        self._verify_mutation(payload, "self-authorize")

        payload = json.loads(ARTIFACT.read_text(encoding="utf-8"))
        payload["external_source_receipts"]["local_byte_receipts_retained"] = True
        self._verify_mutation(payload, "must not falsely claim local byte receipts")

    def test_B0_and_T1_encoder_identity_must_remain_matched(self) -> None:
        payload = json.loads(ARTIFACT.read_text(encoding="utf-8"))
        payload["representation"]["B0_and_T1_supervised_encoder_identity_must_match"] = False
        self._verify_mutation(payload, "B0/T1 encoder identity")

    def test_known_contamination_and_remaining_blocker_boundary_is_explicit(self) -> None:
        payload = verify(ARTIFACT)
        limitations = "\n".join(payload["known_limitations"])
        self.assertIn("OpenWebText", limitations)
        self.assertIn("contamination", limitations.lower())
        self.assertIn("local SHA-256", limitations)
        self.assertIn("trainability", limitations)
        self.assertIn("classifier-head", limitations)
        self.assertIn("collapse thresholds", limitations)
        self.assertIn("parameter-match tolerance", limitations)
        self.assertIn("compute ratio", limitations)


if __name__ == "__main__":
    unittest.main()
