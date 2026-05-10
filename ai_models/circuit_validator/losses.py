from __future__ import annotations

import torch
from torch import Tensor
import torch.nn.functional as F


def circuit_loss(outputs: dict[str, Tensor], batch: dict[str, Tensor]) -> tuple[Tensor, dict[str, Tensor]]:
    issues = F.binary_cross_entropy_with_logits(outputs["issue_logits"], batch["issue_targets"])
    risk = F.cross_entropy(outputs["risk_logits"], batch["risk_class"])
    repair = F.cross_entropy(outputs["repair_logits"], batch["repair_class"])
    board = F.cross_entropy(outputs["board_logits"], batch["board_class"])
    loss = issues * 1.4 + risk + repair + board * 0.4
    return loss, {"issues": issues, "risk": risk, "repair": repair, "board": board}


def multilabel_f1(logits: Tensor, targets: Tensor, threshold: float = 0.5) -> float:
    preds = (logits.sigmoid() >= threshold).float()
    tp = (preds * targets).sum()
    fp = (preds * (1 - targets)).sum()
    fn = ((1 - preds) * targets).sum()
    return float((2 * tp / (2 * tp + fp + fn + 1e-8)).detach().cpu())


def accuracy(logits: Tensor, targets: Tensor) -> float:
    return float((logits.argmax(dim=-1) == targets).float().mean().detach().cpu())

