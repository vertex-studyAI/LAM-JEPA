from __future__ import annotations

from lam_jepa.model import LAMJEPA, LAMJEPAConfig
from lam_jepa.data import sample_batch


def test_large_batch_forward():
    cfg = LAMJEPAConfig()
    model = LAMJEPA(cfg)
    batch = sample_batch("chain", batch=128, vocab_size=cfg.vocab_size)
    out = model(batch.tokens, numeric_x=batch.numeric_x, steps=0)
    assert out["logits"].shape[0] == 128
