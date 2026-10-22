"""Counterexamples for the supplement gate; fixtures are not research results."""
import importlib.util
import json
from pathlib import Path
import stat
import tempfile
import unittest
from unittest.mock import patch
import warnings
import zipfile

SOURCE = Path(__file__).resolve().parents[1] / "scripts/paper/build_aistats2027_anonymous_bundle.py"
SPEC = importlib.util.spec_from_file_location("anonymous_bundle", SOURCE)
bundle = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(bundle)


class AnonymousBundleIntegrityTests(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.addCleanup(self.tmp.cleanup)
        self.archive = Path(self.tmp.name) / "bundle.zip"
        with patch.object(bundle, "frozen_python_paths", return_value=[
            "src/lam_jepa/model.py", "src/lam_jepa/benchmarking/arc_challenge.py"
        ]), patch.object(bundle, "git_bytes", return_value=b"# anonymous source fixture\n"):
            bundle.build(self.archive)
        with zipfile.ZipFile(self.archive) as z:
            self.payloads = {name: z.read(name) for name in z.namelist()}

    def rewrite(self, payloads=None, duplicate=None, info_override=None):
        entries = list((payloads or self.payloads).items())
        if duplicate is not None:
            entries.append((duplicate, self.payloads[duplicate]))
        with warnings.catch_warnings(), zipfile.ZipFile(self.archive, "w") as z:
            warnings.simplefilter("ignore", UserWarning)
            for name, data in sorted(entries):
                info = info_override if info_override and info_override.filename == name else bundle.zip_info(name)
                z.writestr(info, data)

    def assert_rejected(self, message):
        with self.assertRaisesRegex(SystemExit, message):
            bundle.verify(self.archive)

    def test_clean_generated_archive_passes(self):
        bundle.verify(self.archive)

    def test_personal_and_institutional_emails_are_detected(self):
        for address in ["researcher@gmail.com", "analyst@university.example", "name+tag@lab.ac.uk"]:
            with self.subTest(address=address), self.assertRaisesRegex(SystemExit, "email address"):
                bundle.scan_text("source.py", address.encode())

    def test_unmanifested_member_is_rejected(self):
        self.payloads["extra.txt"] = b"anonymous but unbound content"
        self.rewrite()
        self.assert_rejected("cover every payload")

    def test_duplicate_member_is_rejected(self):
        self.rewrite(duplicate="README.md")
        self.assert_rejected("duplicate member")

    def test_manifest_omission_is_rejected(self):
        manifest = json.loads(self.payloads["BUNDLE_MANIFEST.json"])
        del manifest["files"]["README.md"]
        self.payloads["BUNDLE_MANIFEST.json"] = json.dumps(manifest).encode()
        self.rewrite()
        self.assert_rejected("cover every payload")

    def test_modified_payload_is_rejected(self):
        self.payloads["README.md"] += b"changed content"
        self.rewrite()
        self.assert_rejected("manifest mismatch")

    def test_unsafe_paths_are_rejected_before_manifest_validation(self):
        for name in ["../outside.txt", "/absolute.txt", "a/../b.txt", "a//b.txt", "a\\b.txt", "C:note.txt"]:
            with self.subTest(name=name):
                self.rewrite({**self.payloads, name: b"fixture"})
                self.assert_rejected("unsafe bundle member")

    def test_filename_identity_is_detected(self):
        self.rewrite({**self.payloads, "author@gmail.com.txt": b"fixture"})
        self.assert_rejected("email address")

    def test_symlink_member_is_rejected(self):
        info = bundle.zip_info("README.md")
        info.create_system = 3
        info.external_attr = (stat.S_IFLNK | 0o777) << 16
        self.rewrite(info_override=info)
        self.assert_rejected("regular file")

    def test_binary_payload_cannot_bypass_scan(self):
        self.payloads["README.md"] = b"\xff\x00researcher@gmail.com"
        self.rewrite()
        self.assert_rejected("non-UTF-8")

    def test_boolean_manifest_size_is_invalid(self):
        manifest = json.loads(self.payloads["BUNDLE_MANIFEST.json"])
        manifest["files"]["README.md"]["bytes"] = True
        self.payloads["BUNDLE_MANIFEST.json"] = json.dumps(manifest).encode()
        self.rewrite()
        self.assert_rejected("invalid manifest record")

    def test_missing_required_source_is_rejected(self):
        del self.payloads["src/lam_jepa/model.py"]
        self.rewrite()
        self.assert_rejected("missing required files")


if __name__ == "__main__":
    unittest.main()
