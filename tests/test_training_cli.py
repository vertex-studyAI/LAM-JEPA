from __future__ import annotations

import os
import subprocess
import sys
import tempfile
import unittest
from pathlib import Path

import torch

ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from lam_jepa.callbacks.checkpointing.load import load_checkpoint
from lam_jepa.model import LAMJEPA, LAMJEPAConfig


class TrainingCliTest(unittest.TestCase):
    def test_readme_command_writes_a_resumable_checkpoint(self) -> None:
        with tempfile.TemporaryDirectory(prefix="lam-jepa-training-") as tmp:
            tmp_path = Path(tmp)
            run_dir = tmp_path / "seed_11"
            export_path = tmp_path / "exports" / "seed_11.pt"
            env = os.environ.copy()
            existing_pythonpath = env.get("PYTHONPATH")
            env["PYTHONPATH"] = os.pathsep.join(
                part for part in (str(SRC), existing_pythonpath) if part
            )

            command = [
                sys.executable,
                "scripts/train/train_single.py",
                "--seed",
                "11",
                "--steps",
                "1",
                "--batch-size",
                "2",
                "--task",
                "parity",
                "--device",
                "cpu",
                "--out-dir",
                str(run_dir),
                "--out",
                str(export_path),
            ]
            completed = subprocess.run(
                command,
                cwd=ROOT,
                env=env,
                capture_output=True,
                text=True,
                timeout=180,
                check=False,
            )
            self.assertEqual(
                completed.returncode,
                0,
                msg=f"stdout:\n{completed.stdout}\nstderr:\n{completed.stderr}",
            )

            final_checkpoint = run_dir / "final.pt"
            self.assertTrue(final_checkpoint.is_file())
            self.assertTrue(export_path.is_file())

            checkpoint = torch.load(final_checkpoint, map_location="cpu", weights_only=False)
            self.assertEqual(checkpoint["step"], 1)
            self.assertIn("model", checkpoint)
            self.assertIn("optimizer", checkpoint)
            self.assertIn("scheduler", checkpoint)
            self.assertIn("rng", checkpoint)
            self.assertIn("config", checkpoint["extra"])
            self.assertIn("train_config", checkpoint["extra"])
            self.assertEqual(checkpoint["extra"]["train_config"]["seed"], 11)
            self.assertEqual(checkpoint["extra"]["train_config"]["checkpoint_dir"], str(run_dir))

            exported = torch.load(export_path, map_location="cpu", weights_only=False)
            self.assertEqual(exported.keys(), checkpoint.keys())
            self.assertEqual(exported["step"], checkpoint["step"])

            cfg = LAMJEPAConfig(**checkpoint["extra"]["config"])
            reloaded_model = LAMJEPA(cfg)
            loaded = load_checkpoint(final_checkpoint, reloaded_model, map_location="cpu")
            self.assertEqual(loaded["step"], 1)


if __name__ == "__main__":
    unittest.main()
