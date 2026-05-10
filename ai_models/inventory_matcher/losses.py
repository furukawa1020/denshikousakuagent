from __future__ import annotations

import torch
from torch import Tensor
import torch.nn.functional as F


def inventory_loss(outputs: dict[str, Tensor], batch: dict[str, Tensor]) -> tuple[Tensor, dict[str, Tensor]]:
    project = F.cross_entropy(outputs["project_logits"], batch["project_class"])
    tier = F.cross_entropy(outputs["tier_logits"], batch["tier_class"])
    missing = F.binary_cross_entropy_with_logits(outputs["missing_component_logits"], batch["missing_targets"])
    completion = F.mse_loss(outputs["completion_logits"].sigmoid(), batch["completion_targets"])
    budget = F.mse_loss(outputs["budget_fit_logits"].sigmoid(), batch["budget_targets"])
    loss = project + tier * 0.7 + missing * 1.15 + completion * 1.8 + budget * 1.2
    return loss, {"project": project, "tier": tier, "missing": missing, "completion": completion, "budget": budget}


def accuracy(logits: Tensor, targets: Tensor) -> float:
    return float((logits.argmax(dim=-1) == targets).float().mean().detach().cpu())


def multilabel_f1(logits: Tensor, targets: Tensor, threshold: float = 0.45) -> float:
    preds = (logits.sigmoid() >= threshold).float()
    tp = (preds * targets).sum()
    fp = (preds * (1 - targets)).sum()
    fn = ((1 - preds) * targets).sum()
    return float((2 * tp / (2 * tp + fp + fn + 1e-8)).detach().cpu())


def mean_abs_error(logits: Tensor, targets: Tensor) -> float:
    return float((logits.sigmoid() - targets).abs().mean().detach().cpu())

