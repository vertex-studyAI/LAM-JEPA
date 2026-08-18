from __future__ import annotations

import io
import tempfile
import unittest
from pathlib import Path
from urllib.error import URLError

from scripts.data.download_arc_challenge import download


class FlakyOpener:
    def __init__(self, *, failures: int, payload: bytes = b"arc-data") -> None:
        self.failures = failures
        self.payload = payload
        self.calls = 0

    def __call__(self, request, *, timeout: float):
        self.calls += 1
        if self.calls <= self.failures:
            raise URLError("simulated connection reset")
        return io.BytesIO(self.payload)


class ArcDownloadRetryTests(unittest.TestCase):
    def test_transient_failures_retry_same_artifact_and_finish_atomically(self) -> None:
        opener = FlakyOpener(failures=2, payload=b"complete-pinned-artifact")
        delays: list[float] = []

        with tempfile.TemporaryDirectory() as tmp:
            destination = Path(tmp) / "arc-challenge-train.parquet"
            download(
                "https://example.invalid/pinned.parquet",
                destination,
                attempts=4,
                timeout=7,
                base_delay=0.5,
                opener=opener,
                sleep=delays.append,
            )

            self.assertEqual(opener.calls, 3)
            self.assertEqual(delays, [0.5, 1.0])
            self.assertEqual(destination.read_bytes(), b"complete-pinned-artifact")
            self.assertFalse(destination.with_name(f"{destination.name}.part").exists())

    def test_exhausted_retries_fail_without_partial_artifact(self) -> None:
        opener = FlakyOpener(failures=10)

        with tempfile.TemporaryDirectory() as tmp:
            destination = Path(tmp) / "arc-challenge-validation.parquet"
            with self.assertRaisesRegex(RuntimeError, "download failed after 3 attempts"):
                download(
                    "https://example.invalid/pinned.parquet",
                    destination,
                    attempts=3,
                    base_delay=0,
                    opener=opener,
                    sleep=lambda _: None,
                )

            self.assertEqual(opener.calls, 3)
            self.assertFalse(destination.exists())
            self.assertFalse(destination.with_name(f"{destination.name}.part").exists())

    def test_invalid_retry_configuration_fails_closed(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            destination = Path(tmp) / "arc.parquet"
            with self.assertRaisesRegex(ValueError, "attempts must be >= 1"):
                download(
                    "https://example.invalid/pinned.parquet",
                    destination,
                    attempts=0,
                    opener=FlakyOpener(failures=0),
                )


if __name__ == "__main__":
    unittest.main()
