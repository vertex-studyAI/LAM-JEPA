from __future__ import annotations

from pathlib import Path
import re
import unittest

ROOT = Path(__file__).resolve().parents[1]

WORKFLOWS = (
    ".github/workflows/confirmatory-dataset-freezer.yml",
    ".github/workflows/arc-successor-freshness-audit.yml",
    ".github/workflows/arc-successor-authorization-ci.yml",
    ".github/workflows/arc-successor-development-split-freezer.yml",
)

CHECKOUT_SHA = "3d3c42e5aac5ba805825da76410c181273ba90b1"
SETUP_PYTHON_SHA = "5fda3b95a4ea91299a34e894583c3862153e4b97"


class SuccessorWorkflowSupplyChainTests(unittest.TestCase):
    def test_control_workflows_pin_runner_and_actions(self) -> None:
        for relative in WORKFLOWS:
            source = (ROOT / relative).read_text(encoding="utf-8")
            self.assertIn("runs-on: ubuntu-24.04", source, relative)
            self.assertNotIn("runs-on: ubuntu-latest", source, relative)
            self.assertIn(f"actions/checkout@{CHECKOUT_SHA}", source, relative)
            self.assertIn(f"actions/setup-python@{SETUP_PYTHON_SHA}", source, relative)
            self.assertNotRegex(source, r"actions/(?:checkout|setup-python)@v\d+")
            self.assertIn("persist-credentials: false", source, relative)
            self.assertRegex(
                source,
                re.compile(r"ref:\s*\$\{\{ github\.event\.pull_request\.head\.sha \|\| github\.sha \}\}"),
                relative,
            )


if __name__ == "__main__":
    unittest.main()
