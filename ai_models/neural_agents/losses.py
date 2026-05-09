from __future__ import annotations

import torch
from torch import Tensor
import torch.nn.functional as F


def neural_agent_loss(outputs: dict[str, Tensor], batch: dict[str, Tensor]) -> tuple[Tensor, dict[str, Tensor]]:
    safety = F.binary_cross_entropy_with_logits(outputs["safety_logits"], batch["safety_targets"])
    risk = F.cross_entropy(outputs["risk_logits"], batch["risk_class"])
    debug = F.cross_entropy(outputs["debug_logits"], batch["debug_class"])
    firmware = F.cross_entropy(outputs["firmware_logits"], batch["firmware_class"])
    loss = safety * 1.2 + risk + debug + firmware
    return loss, {"safety": safety, "risk": risk, "debug": debug, "firmware": firmware}


def multilabel_f1(logits: Tensor, targets: Tensor, threshold: float = 0.5) -> float:
    preds = (logits.sigmoid() >= threshold).float()
    tp = (preds * targets).sum()
    fp = (preds * (1 - targets)).sum()
    fn = ((1 - preds) * targets).sum()
    return float((2 * tp / (2 * tp + fp + fn + 1e-8)).detach().cpu())


def accuracy(logits: Tensor, targets: Tensor) -> float:
    return float((logits.argmax(dim=-1) == targets).float().mean().detach().cpu())

