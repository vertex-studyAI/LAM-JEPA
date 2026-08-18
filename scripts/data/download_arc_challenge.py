from __future__ import annotations

import argparse
import hashlib
import json
import time
from http.client import HTTPException
from pathlib import Path
from urllib.request import Request, urlopen


ROOT = Path(__file__).resolve().parents[2]
DEFAULT_MANIFEST = ROOT / "data" / "manifests" / "arc_challenge.json"


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def download(url: str, destination: Path, *, attempts: int = 4, base_delay_seconds: float = 1.0) -> None:
    """Download one immutable split with bounded retries and atomic replacement.

    Network failures must not leave a partial file that can be mistaken for a
    completed download. The manifest checksum remains the final integrity gate.
    """
    if attempts < 1:
        raise ValueError("attempts must be >= 1")

    destination.parent.mkdir(parents=True, exist_ok=True)
    partial = destination.with_name(f"{destination.name}.part")

    for attempt in range(1, attempts + 1):
        partial.unlink(missing_ok=True)
        request = Request(url, headers={"User-Agent": "lam-jepa-reproducibility/0.1"})
        try:
            with urlopen(request, timeout=60) as response, partial.open("wb") as output:
                while True:
                    chunk = response.read(1024 * 1024)
                    if not chunk:
                        break
                    output.write(chunk)
            partial.replace(destination)
            return
        except (OSError, HTTPException):
            partial.unlink(missing_ok=True)
            if attempt == attempts:
                raise
            time.sleep(base_delay_seconds * (2 ** (attempt - 1)))


def main() -> None:
    parser = argparse.ArgumentParser(description="Download checksum-addressed AI2 ARC-Challenge splits.")
    parser.add_argument("--manifest", type=Path, default=DEFAULT_MANIFEST)
    parser.add_argument("--out-dir", type=Path, default=ROOT / "data" / "external" / "arc-challenge")
    parser.add_argument("--splits", nargs="+", choices=["train", "validation", "test"], default=["train", "validation"])
    parser.add_argument("--force", action="store_true")
    args = parser.parse_args()

    manifest = json.loads(args.manifest.read_text(encoding="utf-8"))
    files = manifest.get("files", {})
    evidence: dict[str, dict[str, object]] = {}

    for split in args.splits:
        spec = files.get(split)
        if not isinstance(spec, dict):
            raise SystemExit(f"manifest does not define split: {split}")
        destination = args.out_dir / f"arc-challenge-{split}.parquet"
        if args.force or not destination.exists():
            download(str(spec["url"]), destination)
        actual = sha256_file(destination)
        expected = str(spec["sha256"])
        if actual != expected:
            destination.unlink(missing_ok=True)
            raise SystemExit(f"{split}: checksum mismatch: expected {expected}, got {actual}")
        evidence[split] = {
            "path": str(destination),
            "sha256": actual,
            "expected_rows": int(spec["rows"]),
        }

    print(json.dumps({
        "dataset": manifest["dataset"],
        "license": manifest["license"],
        "splits": evidence,
    }, indent=2))


if __name__ == "__main__":
    main()
