from __future__ import annotations

import unittest

from scripts.ci.verify_research_claim_boundary import (
    REQUIRED_README_FRAGMENTS,
    REQUIRED_STATUS_FRAGMENTS,
    verify_claim_boundary,
)


class ResearchClaimBoundaryTests(unittest.TestCase):
    def valid_status(self) -> str:
        return "\n".join(REQUIRED_STATUS_FRAGMENTS)

    def valid_readme(self) -> str:
        return "\n".join(REQUIRED_README_FRAGMENTS)

    def test_current_negative_boundary_is_accepted(self) -> None:
        report = verify_claim_boundary(self.valid_status(), self.valid_readme())
        self.assertEqual(report["verdict"], "RESEARCH_CLAIM_BOUNDARY_VERIFIED")
        self.assertIs(report["arc_superiority_supported"], False)
        self.assertIs(report["research_complete"], False)
        self.assertIs(report["confirmatory_test_authorized_for_hypothesis_rescue"], False)

    def test_missing_negative_verdict_fails_closed(self) -> None:
        status = self.valid_status().replace("VALID_NEGATIVE_OR_INCONCLUSIVE_VALIDATION", "")
        with self.assertRaisesRegex(ValueError, "missing_status_fragments"):
            verify_claim_boundary(status, self.valid_readme())

    def test_readme_cannot_drop_ci_claim_boundary(self) -> None:
        readme = self.valid_readme().replace(REQUIRED_README_FRAGMENTS[1], "")
        with self.assertRaisesRegex(ValueError, "missing_readme_fragments"):
            verify_claim_boundary(self.valid_status(), readme)

    def test_positive_superiority_reclassification_is_rejected(self) -> None:
        status = self.valid_status() + "\nARC SUPERIORITY HYPOTHESIS SUPPORTED\n"
        with self.assertRaisesRegex(ValueError, "forbidden_current_state_fragments"):
            verify_claim_boundary(status, self.valid_readme())


if __name__ == "__main__":
    unittest.main()
