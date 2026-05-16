from __future__ import annotations

from lam_jepa.model import LAMJEPA, LAMJEPAConfig
from lam_jepa.trainers.trainer import Trainer, TrainerConfig


def test_training_loop_runs():
    cfg = LAMJEPAConfig()
    model = LAMJEPA(cfg)
    tcfg = TrainerConfig(steps=4, batch_size=8, task="mixed", device="cpu", eval_every=2, save_every=2)
    trainer = Trainer(model, cfg, tcfg)
    trained = trainer.fit()
    assert trained is not None
    assert len(trainer.history) > 0
