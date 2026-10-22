"""External-reference acceptance/rejection tests; no scientific data or models."""
from __future__ import annotations

import hashlib
import importlib.util
import json
import stat
import subprocess
import sys
import zipfile
import zlib
from pathlib import Path

import pytest

SCRIPT = Path(__file__).resolve().parents[1] / "scripts/paper/verify_frozen_supplement.py"
SPEC = importlib.util.spec_from_file_location("frozen_supplement_verifier", SCRIPT)
verifier = importlib.util.module_from_spec(SPEC)
# dataclasses resolves annotations through the registered module.
sys.modules[SPEC.name] = verifier
SPEC.loader.exec_module(verifier)


def canonical_json(value):
    return (json.dumps(value, indent=2, sort_keys=True) + "\n").encode()


def refresh_manifest(payloads):
    payloads["BUNDLE_MANIFEST.json"] = canonical_json({"files": {
        name: {"sha256": hashlib.sha256(data).hexdigest(), "bytes": len(data)}
        for name, data in payloads.items() if name != "BUNDLE_MANIFEST.json"
    }})


def write_archive(path, payloads, *, kind=None, compression=zipfile.ZIP_DEFLATED):
    with zipfile.ZipFile(path, "w") as archive:
        for name, data in sorted(payloads.items()):
            info = zipfile.ZipInfo(name, date_time=(1980, 1, 1, 0, 0, 0))
            info.compress_type = compression
            info.external_attr = (kind if kind is not None else 0o644) << 16
            archive.writestr(info, data)


@pytest.fixture(scope="module")
def pinned_repo(tmp_path_factory):
    repo = tmp_path_factory.mktemp("frozen-source-git")

    def git(*args):
        return subprocess.check_output(
            ["git", "-C", str(repo), "-c", "user.name=Engineering fixture",
             "-c", "user.email=fixture@example.invalid", *args], stderr=subprocess.DEVNULL
        ).decode().strip()

    git("init", "-q")
    scientific_payloads = {
        "pyproject.toml": b"[project]\nname = 'fixture'\n",
        "protocols/fixture.json": b'{"protected_test_opened": false}\n',
        "src/lam_jepa/model.py": b"VALUE = 3\n",
        "src/lam_jepa/operators.py": b"def twice(x):\n    return 2 * x\n",
    }
    for name, data in scientific_payloads.items():
        target = repo / name
        target.parent.mkdir(parents=True, exist_ok=True)
        target.write_bytes(data)
    ignored = repo / "src/lam_jepa/__pycache__/excluded.py"
    ignored.parent.mkdir()
    ignored.write_text("excluded = True\n")
    git("add", ".")
    git("commit", "-qm", "Frozen artificial scientific source")
    scientific_revision = git("rev-parse", "HEAD")
    readme = "# Fixed artificial reproduction instructions\n"
    expected = {"effect": -0.125, "locked_test_evaluated": False}
    producer = repo / verifier.GENERATOR_PATH
    producer.parent.mkdir(parents=True)
    producer.write_text(
        f"SCIENTIFIC_REVISION = {scientific_revision!r}\n"
        "FIXED_PATHS = ('pyproject.toml', 'protocols/fixture.json')\n"
        f"README = {readme!r}\nEXPECTED = {expected!r}\n"
        "raise AssertionError('the verifier must never execute the producer')\n"
    )
    git("add", ".")
    git("commit", "-qm", "Pinned artificial package literals")
    generator_revision = git("rev-parse", "HEAD")
    payloads = {**scientific_payloads, "README.md": readme.encode(),
                "EXPECTED_RESULTS.json": canonical_json(expected)}
    refresh_manifest(payloads)
    return repo, scientific_revision, generator_revision, payloads, git


@pytest.fixture
def artifact(pinned_repo, tmp_path, monkeypatch):
    repo, scientific, generator, payloads, _ = pinned_repo
    monkeypatch.setattr(verifier, "SCIENTIFIC_REVISION", scientific)
    monkeypatch.setattr(verifier, "GENERATOR_REVISION", generator)
    path = tmp_path / "supplement.zip"
    write_archive(path, payloads)
    return repo, path, dict(payloads)


def test_matches_two_independent_git_references(artifact, pinned_repo):
    repo, path, payloads = artifact
    result = verifier.verify_archive(path, repo)
    assert result["status"] == "FROZEN_SUPPLEMENT_SOURCE_VERIFIED"
    assert result["source_file_count"] == 4
    assert result["member_count"] == 7
    assert result["archive_sha256"] == hashlib.sha256(path.read_bytes()).hexdigest()
    assert result["scientific_revision"] == pinned_repo[1]
    assert result["generator_revision"] == pinned_repo[2]
    assert result["scientific_evaluation_performed"] is False
    assert result["members"]["src/lam_jepa/model.py"]["scientific_git_blob"] == pinned_repo[4](
        "rev-parse", f"{pinned_repo[1]}:src/lam_jepa/model.py"
    )
    assert result["members"]["README.md"]["scientific_git_blob"] is None
    assert set(result["members"]) == set(payloads)


@pytest.mark.parametrize("name", [
    "src/lam_jepa/model.py", "protocols/fixture.json", "EXPECTED_RESULTS.json", "README.md",
])
def test_coherent_rewrite_cannot_authenticate_itself(artifact, name):
    repo, path, payloads = artifact
    # Equal length defeats a size-only check. Recomputed hashes make the internal
    # manifest perfectly consistent, while the external scientific binding fails.
    original = payloads[name]
    payloads[name] = bytes([original[0] ^ 1]) + original[1:]
    refresh_manifest(payloads)
    for member, record in json.loads(payloads["BUNDLE_MANIFEST.json"])["files"].items():
        assert record["sha256"] == hashlib.sha256(payloads[member]).hexdigest()
        assert record["bytes"] == len(payloads[member])
    write_archive(path, payloads)
    with pytest.raises(verifier.VerificationError, match="frozen-source content mismatch") as error:
        verifier.verify_archive(path, repo)
    assert name in str(error.value)


@pytest.mark.parametrize("change", ["missing", "extra", "alias", "manifest"])
def test_requires_complete_canonical_membership_and_manifest(artifact, change):
    repo, path, payloads = artifact
    if change == "missing":
        del payloads["src/lam_jepa/operators.py"]
    elif change == "extra":
        payloads["src/lam_jepa/added.py"] = b"added = True\n"
    elif change == "alias":
        payloads["src/lam_jepa/../model.py"] = payloads.pop("src/lam_jepa/model.py")
    else:
        payloads["BUNDLE_MANIFEST.json"] += b" "
    if change != "manifest":
        refresh_manifest(payloads)
    write_archive(path, payloads)
    with pytest.raises(verifier.VerificationError):
        verifier.verify_archive(path, repo)


def test_duplicate_member_rejected(artifact):
    repo, path, payloads = artifact
    with zipfile.ZipFile(path, "a") as archive:
        with pytest.warns(UserWarning, match="Duplicate name"):
            archive.writestr("README.md", payloads["README.md"])
    with pytest.raises(verifier.VerificationError, match="duplicate"):
        verifier.verify_archive(path, repo)


@pytest.mark.parametrize("kind", [stat.S_IFLNK | 0o644, stat.S_IFDIR | 0o644])
def test_nonregular_member_rejected_even_with_identical_bytes(artifact, kind):
    repo, path, payloads = artifact
    write_archive(path, payloads, kind=kind)
    with pytest.raises(verifier.VerificationError, match="regular file"):
        verifier.verify_archive(path, repo)


@pytest.mark.parametrize("compression", [zipfile.ZIP_STORED, zipfile.ZIP_DEFLATED])
def test_container_compression_is_not_scientific_payload(artifact, compression):
    repo, path, payloads = artifact
    write_archive(path, payloads, compression=compression)
    assert verifier.verify_archive(path, repo)["status"].endswith("VERIFIED")


def test_unsupported_compression_rejected(artifact):
    repo, path, payloads = artifact
    write_archive(path, payloads, compression=zipfile.ZIP_BZIP2)
    with pytest.raises(verifier.VerificationError, match="unsupported compressed"):
        verifier.verify_archive(path, repo)


@pytest.mark.parametrize("damage", ["bad_zip", "oversized", "truncated"])
def test_corrupt_or_unbounded_archive_rejected(artifact, damage):
    repo, path, _ = artifact
    data = path.read_bytes()
    if damage == "bad_zip":
        path.write_bytes(b"not a ZIP")
    elif damage == "truncated":
        path.write_bytes(data[:-30])
    else:
        path.write_bytes(b"x" * (5 * 1024 * 1024))
    with pytest.raises(verifier.VerificationError):
        verifier.verify_archive(path, repo)


def test_working_tree_edits_do_not_replace_the_frozen_reference(artifact):
    repo, path, _ = artifact
    model = repo / "src/lam_jepa/model.py"
    original = model.read_bytes()
    try:
        model.write_text("VALUE = 999\n")
        assert verifier.verify_archive(path, repo)["status"].endswith("VERIFIED")
    finally:
        model.write_bytes(original)


def test_git_replacement_objects_do_not_replace_frozen_blobs(artifact, pinned_repo):
    repo, path, _ = artifact
    git = pinned_repo[4]
    old_blob = git("rev-parse", f"{pinned_repo[1]}:src/lam_jepa/model.py")
    new_blob = git("rev-parse", f"{pinned_repo[1]}:src/lam_jepa/operators.py")
    git("replace", old_blob, new_blob)
    try:
        assert verifier.verify_archive(path, repo)["status"].endswith("VERIFIED")
    finally:
        git("replace", "-d", old_blob)


@pytest.mark.parametrize("kind", ["blob", "tree"])
def test_corrupt_git_object_cannot_authenticate_a_coherently_changed_archive(artifact, pinned_repo, kind):
    repo, path, payloads = artifact
    git = pinned_repo[4]
    model_oid = git("rev-parse", f"{pinned_repo[1]}:src/lam_jepa/model.py")
    if kind == "blob":
        object_oid = model_oid
    else:
        object_oid = git("rev-parse", f"{pinned_repo[1]}:src/lam_jepa")
    object_path = repo / ".git/objects" / object_oid[:2] / object_oid[2:]
    original = object_path.read_bytes()
    original_mode = stat.S_IMODE(object_path.stat().st_mode)
    decoded = zlib.decompress(original)
    if kind == "blob":
        altered = decoded.replace(b"VALUE = 3", b"VALUE = 9")
        payloads["src/lam_jepa/model.py"] = b"VALUE = 9\n"
    else:
        operator_oid = git("rev-parse", f"{pinned_repo[1]}:src/lam_jepa/operators.py")
        altered = decoded.replace(bytes.fromhex(model_oid), bytes.fromhex(operator_oid))
        payloads["src/lam_jepa/model.py"] = payloads["src/lam_jepa/operators.py"]
    assert altered != decoded
    refresh_manifest(payloads)
    write_archive(path, payloads)
    try:
        object_path.chmod(0o600)  # Only this disposable test repository is altered.
        object_path.write_bytes(zlib.compress(altered))
        with pytest.raises(verifier.VerificationError, match="corrupt pinned Git object"):
            verifier.verify_archive(path, repo)
    finally:
        object_path.write_bytes(original)
        object_path.chmod(original_mode)


def test_missing_reference_fails_without_automatic_fetch(artifact, monkeypatch):
    repo, path, _ = artifact
    monkeypatch.setattr(verifier, "GENERATOR_REVISION", "0" * 40)
    with pytest.raises(verifier.VerificationError, match="no automatic fetch"):
        verifier.verify_archive(path, repo)


def test_git_subprocess_disables_network_and_replacements(artifact, monkeypatch):
    repo, _, _ = artifact
    calls = []
    run = subprocess.run

    def observe(*args, **kwargs):
        calls.append((args, kwargs))
        return run(*args, **kwargs)

    monkeypatch.setattr(verifier.subprocess, "run", observe)
    verifier.load_reference(repo)
    assert calls
    for args, kwargs in calls:
        assert "--no-replace-objects" in args[0]
        assert kwargs["env"]["GIT_NO_LAZY_FETCH"] == "1"
        assert kwargs["env"]["GIT_ALLOW_PROTOCOL"] == ""
        assert kwargs["timeout"] == 30


def test_cli_preserves_success_attempt_and_refuses_report_overwrite(artifact, tmp_path, capsys):
    repo, path, _ = artifact
    report = tmp_path / "attempt.jsonl"
    args = [str(path), "--repository", str(repo), "--report", str(report)]
    assert verifier.main(args) == 0
    records = [json.loads(line) for line in report.read_text().splitlines()]
    assert [record["status"] for record in records] == ["STARTED", "FROZEN_SUPPLEMENT_SOURCE_VERIFIED"]
    saved = report.read_bytes()
    assert verifier.main(args) == 2
    assert report.read_bytes() == saved
    capsys.readouterr()


def test_cli_retains_failure_and_archive_without_success_claim(artifact, tmp_path, capsys):
    repo, path, payloads = artifact
    payloads["README.md"] += b"tampered\n"
    refresh_manifest(payloads)
    write_archive(path, payloads)
    original = path.read_bytes()
    report = tmp_path / "failed-attempt.jsonl"
    assert verifier.main([str(path), "--repository", str(repo), "--report", str(report)]) == 1
    records = [json.loads(line) for line in report.read_text().splitlines()]
    assert [record["status"] for record in records] == ["STARTED", "FROZEN_SUPPLEMENT_REJECTED"]
    assert path.read_bytes() == original
    assert "error" in records[-1]
    capsys.readouterr()


def test_cli_cannot_overwrite_archive_with_receipt(artifact, capsys):
    repo, path, _ = artifact
    original = path.read_bytes()
    assert verifier.main([str(path), "--repository", str(repo), "--report", str(path)]) == 2
    assert path.read_bytes() == original
    capsys.readouterr()
