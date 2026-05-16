from __future__ import annotations
from pathlib import Path

from lam_jepa.model import LAMJEPA, LAMJEPAConfig
from lam_jepa.trainers.trainer import Trainer, TrainerConfig


def test_checkpoint_save_and_load(tmp_path: Path):
    cfg = LAMJEPAConfig()
    model = LAMJEPA(cfg)
    tcfg = TrainerConfig(steps=3, batch_size=8, task="parity", device="cpu", checkpoint_dir=str(tmp_path))
    trainer = Trainer(model, cfg, tcfg)
    trainer.fit()
    ckpt = trainer.save("checkpoint.pt")
    assert ckpt.exists()

    model2 = LAMJEPA(cfg)
    trainer2 = Trainer(model2, cfg, tcfg)
    trainer2.load(ckpt)
    assert trainer2.step == trainer.step
