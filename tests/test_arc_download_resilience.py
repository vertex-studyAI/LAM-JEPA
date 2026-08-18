from __future__ import annotations

import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

from scripts.data import download_arc_challenge as downloader


class _Response:
    def __init__(self, chunks: list[bytes], failure: BaseException | None = None) -> None:
        self._chunks = iter(chunks)
        self._failure = failure

    def __enter__(self) -> "_Response":
        return self

    def __exit__(self, exc_type, exc, tb) -> bool:
        return False

    def read(self, _size: int) -> bytes:
        try:
            return next(self._chunks)
        except StopIteration:
            if self._failure is not None:
                failure, self._failure = self._failure, None
                raise failure
            return b""


class ArcDownloadResilienceTests(unittest.TestCase):
    def test_midstream_failure_retries_without_committing_partial_file(self) -> None:
        payload = b"complete-frozen-split"
        broken = _Response([b"partial"], ConnectionResetError("transient reset"))
        healthy = _Response([payload])

        with tempfile.TemporaryDirectory() as tmpdir:
            destination = Path(tmpdir) / "arc-challenge-train.parquet"
            partial = destination.with_name(f"{destination.name}.part")

            with (
                patch.object(downloader, "urlopen", side_effect=[broken, healthy]) as mocked_open,
                patch.object(downloader.time, "sleep") as mocked_sleep,
            ):
                downloader.download(
                    "https://example.invalid/frozen.parquet",
                    destination,
                    attempts=2,
                    base_delay_seconds=0.25,
                )

            self.assertEqual(destination.read_bytes(), payload)
            self.assertFalse(partial.exists())
            self.assertEqual(mocked_open.call_count, 2)
            mocked_sleep.assert_called_once_with(0.25)

    def test_exhausted_retries_leave_no_destination_or_partial_file(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            destination = Path(tmpdir) / "arc-challenge-validation.parquet"
            partial = destination.with_name(f"{destination.name}.part")

            failures = [
                _Response([b"one"], ConnectionResetError("reset one")),
                _Response([b"two"], ConnectionResetError("reset two")),
            ]
            with (
                patch.object(downloader, "urlopen", side_effect=failures),
                patch.object(downloader.time, "sleep"),
            ):
                with self.assertRaises(ConnectionResetError):
                    downloader.download(
                        "https://example.invalid/frozen.parquet",
                        destination,
                        attempts=2,
                        base_delay_seconds=0,
                    )

            self.assertFalse(destination.exists())
            self.assertFalse(partial.exists())

    def test_invalid_attempt_count_fails_before_network_access(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            destination = Path(tmpdir) / "unused.parquet"
            with patch.object(downloader, "urlopen") as mocked_open:
                with self.assertRaisesRegex(ValueError, "attempts must be >= 1"):
                    downloader.download("https://example.invalid/unused", destination, attempts=0)
            mocked_open.assert_not_called()


if __name__ == "__main__":
    unittest.main()
