from __future__ import annotations
import torch

from lam_jepa.model import LAMJEPA, LAMJEPAConfig
from lam_jepa.data import sample_batch
from lam_jepa.utils import set_seed


def test_seed_reproducibility():
    cfg = LAMJEPAConfig()
    set_seed(123)
    model1 = LAMJEPA(cfg)
    batch1 = sample_batch("modadd", batch=4, vocab_size=cfg.vocab_size)
    out1 = model1(batch1.tokens, numeric_x=batch1.numeric_x, steps=0)["logits"]

    set_seed(123)
    model2 = LAMJEPA(cfg)
    batch2 = sample_batch("modadd", batch=4, vocab_size=cfg.vocab_size)
    out2 = model2(batch2.tokens, numeric_x=batch2.numeric_x, steps=0)["logits"]

    assert torch.allclose(out1, out2)
