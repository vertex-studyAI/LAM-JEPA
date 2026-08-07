from __future__ import annotations

from typing import Any

import torch

from lam_jepa.model import LAMJEPAConfig

ARC_V5_REPAIR_ID = "arc-v5-stable-ema-residual-0.03125"
ARC_V5_HARD_FRACTION = 0.03125
ARC_V5_CONTINUOUS_FRACTION = 1.0 - ARC_V5_HARD_FRACTION
ARC_V5_EMA_PSEUDOCOUNT = 1.0


def apply_arc_v5_quantizer_repair(model: Any) -> Any:
    """Apply the predeclared ARC v5 quantizer repair without changing legacy defaults.

    The repair is intentionally opt-in so older checkpoints and frozen protocol
    artifacts keep their original semantics. It combines only the two mechanisms
    supported by the train-only falsification sequence:

    1. coherent EMA initialization (unit pseudocounts and weights synchronized to
       the live codebook), preventing the proven first-update codebook explosion;
    2. the smallest predeclared near-continuous residual condition that passed the
       frozen two-seed overfit gate: 3.125% hard-code displacement / 96.875%
       continuous projected latent.

    The underlying quantizer parameter/buffer names are unchanged, so model state
    dict keys remain stable. The residual behavior is installed as a forward hook
    and therefore must be reapplied by the explicit ARC v5 builder after loading a
    checkpoint.
    """

    if getattr(model, "_arc_v5_quantizer_repair_id", None) == ARC_V5_REPAIR_ID:
        return model

    backbone = getattr(model, "backbone", None)
    if backbone is None or not getattr(backbone.cfg, "use_quantizer", False):
        raise ValueError("ARC v5 quantizer repair requires an enabled LAM quantizer")

    quantizer = backbone.quantizer
    with torch.no_grad():
        quantizer.ema_count.fill_(ARC_V5_EMA_PSEUDOCOUNT)
        quantizer.ema_weight.copy_(quantizer.codebook.detach())

    def _residual_hook(module, inputs, output):
        if len(inputs) != 1:
            raise RuntimeError("unexpected EMAQuantizer input signature")
        z = inputs[0]
        hard_z_q, quant_loss, indices = output
        mixed_z_q = z + ARC_V5_HARD_FRACTION * (hard_z_q - z)
        return mixed_z_q, quant_loss, indices

    handle = quantizer.register_forward_hook(_residual_hook)
    model._arc_v5_quantizer_repair_handle = handle
    model._arc_v5_quantizer_repair_id = ARC_V5_REPAIR_ID
    model._arc_v5_quantizer_hard_fraction = ARC_V5_HARD_FRACTION
    model._arc_v5_quantizer_continuous_fraction = ARC_V5_CONTINUOUS_FRACTION
    return model


def build_arc_v5_repaired_classifier(
    cfg: LAMJEPAConfig | None = None,
    *,
    num_choices: int = 4,
):
    """Build the versioned repaired ARC classifier while preserving legacy defaults."""

    from lam_jepa.benchmarking.arc_challenge import LAMARCClassifier

    model = LAMARCClassifier(cfg or LAMJEPAConfig(), num_choices=num_choices)
    return apply_arc_v5_quantizer_repair(model)


def arc_v5_repair_spec() -> dict[str, object]:
    return {
        "repair_id": ARC_V5_REPAIR_ID,
        "ema_pseudocount": ARC_V5_EMA_PSEUDOCOUNT,
        "ema_weight_initialization": "copy_live_codebook",
        "hard_fraction": ARC_V5_HARD_FRACTION,
        "continuous_fraction": ARC_V5_CONTINUOUS_FRACTION,
        "legacy_default_behavior_changed": False,
        "validation_authorized": False,
        "confirmatory_test_authorized": False,
    }
