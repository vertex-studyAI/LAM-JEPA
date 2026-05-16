from __future__ import annotations
import numpy as np
import torch


def expected_calibration_error(confidence: torch.Tensor, correct: torch.Tensor, n_bins: int = 10) -> float:
    conf = confidence.detach().float().cpu().view(-1).numpy()
    corr = correct.detach().float().cpu().view(-1).numpy()
    bins = np.linspace(0.0, 1.0, n_bins + 1)
    ece = 0.0
    for i in range(n_bins):
        lo, hi = bins[i], bins[i + 1]
        mask = (conf >= lo) & (conf <= hi if i == n_bins - 1 else conf < hi)
        if mask.any():
            acc = corr[mask].mean()
            avg_conf = conf[mask].mean()
            ece += mask.mean() * abs(acc - avg_conf)
    return float(ece)


def reliability_bins(confidence: torch.Tensor, correct: torch.Tensor, n_bins: int = 10):
    conf = confidence.detach().float().cpu().view(-1).numpy()
    corr = correct.detach().float().cpu().view(-1).numpy()
    bins = np.linspace(0.0, 1.0, n_bins + 1)
    out = []
    for i in range(n_bins):
        lo, hi = bins[i], bins[i + 1]
        mask = (conf >= lo) & (conf <= hi if i == n_bins - 1 else conf < hi)
        if mask.any():
            out.append({
                "bin": i,
                "count": int(mask.sum()),
                "confidence": float(conf[mask].mean()),
                "accuracy": float(corr[mask].mean()),
            })
    return out
