from __future__ import annotations

import torch

from lam_jepa.benchmarking.arc_challenge import LAMARCClassifier
from lam_jepa.benchmarking.arc_v5_repair import (
    ARC_V5_CONTINUOUS_FRACTION,
    ARC_V5_EMA_PSEUDOCOUNT,
    ARC_V5_HARD_FRACTION,
    ARC_V5_REPAIR_ID,
    apply_arc_v5_quantizer_repair,
    arc_v5_repair_spec,
)
from lam_jepa.model import LAMJEPAConfig


def test_arc_v5_repair_is_opt_in_and_state_dict_compatible():
    torch.manual_seed(11)
    model = LAMARCClassifier(LAMJEPAConfig(), num_choices=4)
    keys_before = tuple(model.state_dict().keys())
    quantizer = model.backbone.quantizer

    assert torch.count_nonzero(quantizer.ema_count).item() == 0
    assert not torch.allclose(quantizer.ema_weight, quantizer.codebook.detach())

    returned = apply_arc_v5_quantizer_repair(model)

    assert returned is model
    assert tuple(model.state_dict().keys()) == keys_before
    assert model._arc_v5_quantizer_repair_id == ARC_V5_REPAIR_ID
    assert torch.allclose(
        quantizer.ema_count,
        torch.full_like(quantizer.ema_count, ARC_V5_EMA_PSEUDOCOUNT),
    )
    assert torch.allclose(quantizer.ema_weight, quantizer.codebook.detach())


def test_arc_v5_repair_matches_predeclared_residual_forward_in_eval_mode():
    torch.manual_seed(17)
    model = LAMARCClassifier(LAMJEPAConfig(), num_choices=4)
    apply_arc_v5_quantizer_repair(model)
    quantizer = model.backbone.quantizer
    quantizer.eval()

    z = torch.randn(7, quantizer.dim)
    with torch.no_grad():
        flat = z.view(-1, quantizer.dim)
        dist = (
            flat.pow(2).sum(1, keepdim=True)
            - 2 * flat @ quantizer.codebook.t()
            + quantizer.codebook.pow(2).sum(1)
        )
        indices = dist.argmin(dim=1)
        hard = quantizer.codebook[indices].view_as(z)
        expected = z + ARC_V5_HARD_FRACTION * (hard - z)
        repaired, _, observed_indices = quantizer(z)

    assert torch.equal(observed_indices, indices)
    assert torch.allclose(repaired, expected, atol=1e-7, rtol=1e-6)


def test_arc_v5_repair_spec_is_fail_closed_for_validation():
    spec = arc_v5_repair_spec()
    assert spec["repair_id"] == ARC_V5_REPAIR_ID
    assert spec["hard_fraction"] == ARC_V5_HARD_FRACTION
    assert spec["continuous_fraction"] == ARC_V5_CONTINUOUS_FRACTION
    assert spec["legacy_default_behavior_changed"] is False
    assert spec["validation_authorized"] is False
    assert spec["confirmatory_test_authorized"] is False
