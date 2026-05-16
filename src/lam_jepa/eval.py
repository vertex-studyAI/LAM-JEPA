from __future__ import annotations
from typing import Dict, Iterable, Tuple, Optional
import torch
import torch.nn.functional as F


@torch.no_grad()
def evaluate(model, loader, device="cpu"):
    model.eval()
    correct = 0
    total = 0
    confs = []
    preds_all = []
    labels_all = []
    for batch in loader:
        tokens = batch["tokens"].to(device)
        numeric_x = batch["numeric_x"].to(device)
        labels = batch["labels"].to(device)
        rubric = batch["rubric"].to(device)
        outputs = model(tokens, numeric_x=numeric_x)
        pred = outputs["logits"].argmax(dim=-1)
        correct += (pred == labels).sum().item()
        total += labels.numel()
        confs.append(outputs["confidence"].detach().cpu())
        preds_all.append(pred.detach().cpu())
        labels_all.append(labels.detach().cpu())
    conf = torch.cat(confs).mean().item() if confs else 0.0
    return {"accuracy": correct / max(total, 1), "confidence": conf, "preds": torch.cat(preds_all) if preds_all else torch.tensor([]), "labels": torch.cat(labels_all) if labels_all else torch.tensor([])}
