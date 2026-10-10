"""Verify supplement payloads against pinned Git inputs, without executing them.

Run the separate anonymous-bundle check as well. Source identity does not certify
scientific validity, anonymity, authorship, or submission permission.
"""
from __future__ import annotations

import argparse
import ast
import hashlib
import io
import json
import os
import re
import stat
import subprocess
import sys
import zipfile
import zlib
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import BinaryIO, TextIO

ROOT = Path(__file__).resolve().parents[2]
SCIENTIFIC_REVISION = "760aa7f9a73a177d5ff4ba7eb470f7e68ace63cb"
GENERATOR_REVISION = "328d33f5d2e49a72ebdaf042cec318798c606563"
GENERATOR_PATH = "scripts/paper/build_aistats2027_anonymous_bundle.py"
MANIFEST_PATH = "BUNDLE_MANIFEST.json"
VERIFIER_VERSION = 1


class VerificationError(ValueError):
    """The reference is unavailable or the artifact does not match it."""


def _json_bytes(value: object) -> bytes:
    return (json.dumps(value, indent=2, sort_keys=True, allow_nan=False) + "\n").encode("utf-8")


def _sha256(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def _git(repo: Path, *args: str, input_bytes: bytes | None = None) -> bytes:
    # Verify local, original Git objects. Never consult replacement objects or
    # let a partial clone perform an implicit network fetch during verification.
    env = dict(os.environ, GIT_NO_LAZY_FETCH="1", GIT_NO_REPLACE_OBJECTS="1",
               GIT_TERMINAL_PROMPT="0", GIT_ALLOW_PROTOCOL="")
    try:
        result = subprocess.run(
            ["git", "--no-replace-objects", "-C", str(repo), *args],
            input=input_bytes, stdout=subprocess.PIPE, stderr=subprocess.PIPE,
            env=env, timeout=30, check=False,
        )
    except (OSError, subprocess.TimeoutExpired) as exc:
        raise VerificationError("cannot read local frozen Git reference") from exc
    if result.returncode:
        raise VerificationError(
            "frozen Git objects unavailable; obtain both pinned revisions and their "
            "source blobs explicitly before verification (no automatic fetch)"
        )
    return result.stdout


def _recipe_literals(source: bytes) -> dict[str, object]:
    wanted = {"SCIENTIFIC_REVISION", "FIXED_PATHS", "README", "EXPECTED"}
    values: dict[str, object] = {}
    try:
        module = ast.parse(source.decode("utf-8"))
        for node in module.body:
            if not isinstance(node, ast.Assign):
                continue
            for target in node.targets:
                if isinstance(target, ast.Name) and target.id in wanted:
                    if target.id in values:
                        raise VerificationError(f"duplicate pinned recipe literal: {target.id}")
                    values[target.id] = ast.literal_eval(node.value)
    except (SyntaxError, UnicodeError, ValueError, TypeError) as exc:
        raise VerificationError("pinned package recipe literals are invalid") from exc
    if set(values) != wanted:
        raise VerificationError("pinned package recipe lacks required literals")
    if values["SCIENTIFIC_REVISION"] != SCIENTIFIC_REVISION:
        raise VerificationError("pinned producer targets a different scientific revision")
    paths = values["FIXED_PATHS"]
    if (not isinstance(paths, tuple) or not paths
            or any(not isinstance(path, str) for path in paths)
            or len(set(paths)) != len(paths)):
        raise VerificationError("pinned package recipe has invalid fixed paths")
    if not isinstance(values["README"], str) or not isinstance(values["EXPECTED"], dict):
        raise VerificationError("pinned package documents must be literal text and mapping")
    return values


def _objects(repo: Path, object_ids: list[str]) -> dict[str, tuple[str, bytes]]:
    if any(re.fullmatch(r"[0-9a-f]{40}", oid) is None for oid in object_ids):
        raise VerificationError("invalid pinned Git object identity")
    response = _git(repo, "cat-file", "--batch",
                    input_bytes=("\n".join(object_ids) + "\n").encode("ascii"))
    stream = io.BytesIO(response)
    result: dict[str, tuple[str, bytes]] = {}
    for oid in object_ids:
        try:
            header = stream.readline().decode("ascii").strip().split()
        except UnicodeError as exc:
            raise VerificationError(f"cannot decode pinned Git object: {oid}") from exc
        if len(header) != 3 or header[0] != oid or header[1] not in {"commit", "tree", "blob"}:
            raise VerificationError(f"missing or invalid pinned Git object: {oid}; no automatic fetch")
        try:
            size = int(header[2])
        except ValueError as exc:
            raise VerificationError(f"cannot decode pinned Git object size: {oid}") from exc
        if size < 0:
            raise VerificationError("negative Git object size")
        data = stream.read(size)
        if len(data) != size or stream.read(1) != b"\n":
            raise VerificationError("truncated pinned Git object")
        # cat-file reports the requested name even for a corrupt loose object.
        # Rehash every commit, intermediate tree and blob before trusting it.
        object_header = f"{header[1]} {size}\0".encode("ascii")
        if hashlib.sha1(object_header + data).hexdigest() != oid:
            raise VerificationError(f"corrupt pinned Git object: {oid}")
        result[oid] = (header[1], data)
    if stream.read(1):
        raise VerificationError("unexpected trailing Git object response")
    return result


def _typed_object(objects: dict[str, tuple[str, bytes]], oid: str, kind: str) -> bytes:
    actual_kind, data = objects[oid]
    if actual_kind != kind:
        raise VerificationError(f"pinned Git object must be a {kind}: {oid}")
    return data


def _commit_tree(objects: dict[str, tuple[str, bytes]], revision: str) -> str:
    data = _typed_object(objects, revision, "commit")
    first = data.split(b"\n", 1)[0]
    if re.fullmatch(rb"tree [0-9a-f]{40}", first) is None:
        raise VerificationError("pinned commit lacks its canonical tree identity")
    return first[5:].decode("ascii")


def _tree_members(data: bytes) -> dict[str, tuple[str, str]]:
    members: dict[str, tuple[str, str]] = {}
    offset = 0
    try:
        while offset < len(data):
            separator = data.index(b" ", offset)
            mode = data[offset:separator].decode("ascii")
            end = data.index(b"\0", separator + 1)
            name = data[separator + 1:end].decode("utf-8")
            oid = data[end + 1:end + 21]
            if (len(oid) != 20 or name in members or name in {"", ".", ".."}
                    or "/" in name or mode not in {"40000", "100644", "100755", "120000", "160000"}):
                raise VerificationError("invalid pinned Git tree entry")
            members[name] = (mode, oid.hex())
            offset = end + 21
    except (ValueError, UnicodeError) as exc:
        raise VerificationError("cannot decode pinned Git tree") from exc
    return members


def _source_tree(repo: Path, root_oid: str) -> dict[str, tuple[str, str]]:
    entries: dict[str, tuple[str, str]] = {}
    pending = [("", root_oid)]
    while pending:
        objects = _objects(repo, sorted({oid for _, oid in pending}))
        children = []
        for prefix, oid in pending:
            for name, (mode, child_oid) in _tree_members(_typed_object(objects, oid, "tree")).items():
                path = prefix + name
                if mode == "40000":
                    children.append((path + "/", child_oid))
                elif mode != "160000":
                    entries[path] = (mode, child_oid)
        pending = children
    return entries


def _recipe_blob(repo: Path, root_oid: str) -> bytes:
    oid = root_oid
    parts = GENERATOR_PATH.split("/")
    for index, part in enumerate(parts):
        objects = _objects(repo, [oid])
        members = _tree_members(_typed_object(objects, oid, "tree"))
        if part not in members:
            raise VerificationError("pinned package recipe is absent from producer tree")
        mode, oid = members[part]
        if index < len(parts) - 1 and mode != "40000":
            raise VerificationError("pinned producer path traverses a non-directory")
        if index == len(parts) - 1 and mode not in {"100644", "100755"}:
            raise VerificationError("pinned package recipe is not a regular file")
    return _typed_object(_objects(repo, [oid]), oid, "blob")


@dataclass(frozen=True)
class FrozenReference:
    payloads: dict[str, bytes]
    source_blobs: dict[str, str]
    generator_sha256: str


def load_reference(repo: Path = ROOT) -> FrozenReference:
    """Read the fixed references; no archive fields influence reference selection."""
    commits = _objects(repo, [SCIENTIFIC_REVISION, GENERATOR_REVISION])
    generator = _recipe_blob(repo, _commit_tree(commits, GENERATOR_REVISION))
    recipe = _recipe_literals(generator)
    entries = _source_tree(repo, _commit_tree(commits, SCIENTIFIC_REVISION))
    python_paths = {
        path for path in entries
        if path.startswith("src/lam_jepa/") and path.endswith(".py")
        and "__pycache__" not in path
    }
    if not python_paths:
        raise VerificationError("pinned scientific package contains no Python sources")
    paths = sorted(set(recipe["FIXED_PATHS"]) | python_paths)
    for path in paths:
        if path not in entries or entries[path][0] not in ("100644", "100755"):
            raise VerificationError(f"pinned source is missing or is not a regular file: {path}")
    source_blobs = {path: entries[path][1] for path in paths}
    blobs = _objects(repo, sorted(set(source_blobs.values())))
    payloads = {path: _typed_object(blobs, source_blobs[path], "blob") for path in paths}
    payloads["README.md"] = recipe["README"].encode("utf-8")
    payloads["EXPECTED_RESULTS.json"] = _json_bytes(recipe["EXPECTED"])
    payloads[MANIFEST_PATH] = _json_bytes({"files": {
        path: {"sha256": _sha256(data), "bytes": len(data)}
        for path, data in sorted(payloads.items())
    }})
    return FrozenReference(payloads, source_blobs, _sha256(generator))


def _snapshot(file: BinaryIO, limit: int) -> bytes:
    data = file.read(limit + 1)
    if len(data) > limit:
        raise VerificationError("archive exceeds the bounded frozen-package container size")
    return data


def verify_archive(archive_path: Path, repo: Path = ROOT) -> dict[str, object]:
    """Return provenance only after every member matches the external reference."""
    reference = load_reference(repo)
    # A bounded snapshot ties the reported hash to the exact bytes inspected,
    # even if another process replaces the path during verification.
    limit = 2 * sum(map(len, reference.payloads.values())) + 4 * 1024 * 1024
    try:
        with archive_path.open("rb") as file:
            snapshot = _snapshot(file, limit)
        with zipfile.ZipFile(io.BytesIO(snapshot)) as archive:
            infos = archive.infolist()
            names = [info.filename for info in infos]
            if len(set(names)) != len(names):
                raise VerificationError("duplicate archive member names")
            missing = set(reference.payloads) - set(names)
            extra = set(names) - set(reference.payloads)
            if missing or extra:
                raise VerificationError(
                    f"frozen package membership differs: missing={sorted(missing)}, extra={sorted(extra)}"
                )
            # Inspect scientific payloads first so a coherently rewritten
            # manifest cannot obscure the particular changed source member.
            for info in sorted(infos, key=lambda item: item.filename == MANIFEST_PATH):
                name = info.filename
                expected = reference.payloads[name]
                kind = stat.S_IFMT(info.external_attr >> 16)
                if (info.orig_filename != name or info.is_dir()
                        or info.external_attr & 0x10 or kind not in (0, stat.S_IFREG)):
                    raise VerificationError(f"archive member is not a regular file: {name}")
                if info.flag_bits & 1 or info.compress_type not in (zipfile.ZIP_STORED, zipfile.ZIP_DEFLATED):
                    raise VerificationError(f"encrypted or unsupported compressed archive member: {name}")
                if info.file_size != len(expected):
                    raise VerificationError(f"frozen-source size mismatch: {name}")
                with archive.open(info) as member:
                    actual = member.read(len(expected) + 1)
                if actual != expected:
                    raise VerificationError(f"frozen-source content mismatch: {name}")
    except (OSError, zipfile.BadZipFile, RuntimeError, NotImplementedError, EOFError, zlib.error) as exc:
        raise VerificationError(f"cannot read frozen supplement: {exc}") from exc
    return {
        "status": "FROZEN_SUPPLEMENT_SOURCE_VERIFIED",
        "verifier_version": VERIFIER_VERSION,
        "verifier_source_sha256": _sha256(Path(__file__).read_bytes()),
        "archive_sha256": _sha256(snapshot),
        "archive_bytes": len(snapshot),
        "scientific_revision": SCIENTIFIC_REVISION,
        "generator_revision": GENERATOR_REVISION,
        "generator_source_sha256": reference.generator_sha256,
        "source_file_count": len(reference.source_blobs),
        "member_count": len(reference.payloads),
        "uncompressed_bytes": sum(map(len, reference.payloads.values())),
        "members": {
            path: {"sha256": _sha256(data), "bytes": len(data),
                   "scientific_git_blob": reference.source_blobs.get(path)}
            for path, data in sorted(reference.payloads.items())
        },
        "scientific_evaluation_performed": False,
        "scope": "Exact uncompressed package identity to the two pinned Git inputs only.",
    }


def _event(stream: TextIO, value: dict[str, object]) -> None:
    record = {"recorded_at_utc": datetime.now(timezone.utc).isoformat(), **value}
    stream.write(json.dumps(record, sort_keys=True, allow_nan=False) + "\n")
    stream.flush()
    os.fsync(stream.fileno())


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("archive", type=Path)
    parser.add_argument("--repository", type=Path, default=ROOT,
                        help="Local checkout containing the pinned Git objects (no network fetch).")
    parser.add_argument("--report", type=Path,
                        help="Fresh JSONL attempt receipt, kept outside the anonymous archive.")
    args = parser.parse_args(argv)
    receipt: TextIO | None = None
    try:
        if args.report:
            # Reserve first: an existing receipt or the input archive is never overwritten.
            receipt = args.report.open("x", encoding="utf-8")
            _event(receipt, {"status": "STARTED", "archive": str(args.archive),
                             "scientific_revision": SCIENTIFIC_REVISION,
                             "generator_revision": GENERATOR_REVISION})
        try:
            result = verify_archive(args.archive, args.repository)
        except VerificationError as exc:
            result = {"status": "FROZEN_SUPPLEMENT_REJECTED", "error": str(exc),
                      "scientific_evaluation_performed": False}
            if receipt:
                _event(receipt, result)
            print(json.dumps(result, indent=2), file=sys.stderr)
            return 1
        if receipt:
            _event(receipt, result)
        print(json.dumps({key: value for key, value in result.items() if key != "members"}, indent=2))
        return 0
    except OSError as exc:
        print(f"cannot create or preserve verification receipt: {exc}", file=sys.stderr)
        return 2
    finally:
        if receipt:
            receipt.close()


if __name__ == "__main__":
    raise SystemExit(main())
