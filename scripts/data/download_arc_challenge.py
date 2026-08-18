from __future__ import annotations

import argparse
import hashlib
import json
import time
from pathlib import Path
from typing import Callable
from urllib.error import URLError
from urllib.request import Request, urlopen


ROOT = Path(__file__).resolve().parents[2]
DEFAULT_MANIFEST = ROOT / "data" / "manifests" / "arc_challenge.json"


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def download(
    url: str,
    destination: Path,
    *,
    attempts: int = 4,
    timeout: float = 60,
    base_delay: float = 2.0,
    opener: Callable[..., object] = urlopen,
    sleep: Callable[[float], None] = time.sleep,
) -> None:
    """Download one pinned artifact with bounded retries for transport failures.

    Retries never change the URL or scientific artifact identity. Data is written
    to a temporary sibling and moved into place only after a complete response,
    so a reset connection cannot leave a partial file that looks usable.
    """
    if attempts < 1:
        raise ValueError("attempts must be >= 1")
    if timeout <= 0:
        raise ValueError("timeout must be > 0")
    if base_delay < 0:
        raise ValueError("base_delay must be >= 0")

    destination.parent.mkdir(parents=True, exist_ok=True)
    partial = destination.with_name(f"{destination.name}.part")
    request = Request(url, headers={"User-Agent": "lam-jepa-reproducibility/0.1"})

    last_error: BaseException | None = None
    for attempt in range(1, attempts + 1):
        partial.unlink(missing_ok=True)
        try:
            with opener(request, timeout=timeout) as response, partial.open("wb") as output:
                while True:
                    chunk = response.read(1024 * 1024)
                    if not chunk:
                        break
                    output.write(chunk)
            partial.replace(destination)
            return
        except (URLError, TimeoutError, ConnectionError, OSError) as error:
            last_error = error
            partial.unlink(missing_ok=True)
            if attempt == attempts:
                break
            delay = base_delay * (2 ** (attempt - 1))
            if delay > 0:
                sleep(delay)

    raise RuntimeError(
        f"download failed after {attempts} attempts for {url}: {last_error}"
    ) from last_error


def main() -> None:
    parser = argparse.ArgumentParser(description="Download checksum-addressed AI2 ARC-Challenge splits.")
    parser.add_argument("--manifest", type=Path, default=DEFAULT_MANIFEST)
    parser.add_argument("--out-dir", type=Path, default=ROOT / "data" / "external" / "arc-challenge")
    parser.add_argument("--splits", nargs="+", choices=["train", "validation", "test"], default=["train", "validation"])
    parser.add_argument("--force", action="store_true")
    parser.add_argument("--download-attempts", type=int, default=4)
    parser.add_argument("--download-timeout", type=float, default=60)
    parser.add_argument("--download-backoff-seconds", type=float, default=2.0)
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
            download(
                str(spec["url"]),
                destination,
                attempts=args.download_attempts,
                timeout=args.download_timeout,
                base_delay=args.download_backoff_seconds,
            )
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
