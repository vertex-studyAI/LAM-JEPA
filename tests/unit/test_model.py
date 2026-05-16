from __future__ import annotations
import torch

from lam_jepa.model import LAMJEPA, LAMJEPAConfig
from lam_jepa.data import sample_batch


def test_model_forward_shapes():
    cfg = LAMJEPAConfig()
    model = LAMJEPA(cfg)
    batch = sample_batch("modadd", batch=4, vocab_size=cfg.vocab_size)
    out = model(batch.tokens, numeric_x=batch.numeric_x, steps=0)
    assert out["logits"].shape == (4, cfg.vocab_size)
    assert out["confidence"].shape == (4, 1)
    assert out["verifier"].shape == (4, 1)
    assert out["rubric"].shape == (4, cfg.num_rubric)
    assert len(out["traj"]) >= 1


def test_model_ablation_flags():
    cfg = LAMJEPAConfig(use_quantizer=False, use_memory=False, use_planner=False, use_target=False)
    model = LAMJEPA(cfg)
    batch = sample_batch("parity", batch=2, vocab_size=cfg.vocab_size)
    out = model(batch.tokens, numeric_x=batch.numeric_x, steps=0)
    assert torch.isfinite(out["logits"]).all()
