from __future__ import annotations
import torch

from lam_jepa.model import LAMJEPA, LAMJEPAConfig
from lam_jepa.data import sample_batch
from lam_jepa.losses import total_loss


def test_total_loss_is_finite():
    cfg = LAMJEPAConfig()
    model = LAMJEPA(cfg)
    batch = sample_batch("algebra", batch=4, vocab_size=cfg.vocab_size)
    out = model(batch.tokens, numeric_x=batch.numeric_x, steps=0)
    loss, stats = total_loss(out, batch.labels, batch.rubric)
    assert torch.isfinite(loss)
    assert "total" in stats
    assert stats["total"] == stats["total"]
