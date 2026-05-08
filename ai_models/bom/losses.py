from __future__ import annotations

import torch
from torch import Tensor
import torch.nn.functional as F


def quantile_loss(predicted: Tensor, target: Tensor, quantiles: tuple[float, float, float] = (0.1, 0.5, 0.9)) -> Tensor:
    losses = []
    for index, quantile in enumerate(quantiles):
        error = target[:, index] - predicted[:, index]
        losses.append(torch.maximum((quantile - 1) * error, quantile * error).unsqueeze(-1))
    return torch.cat(losses, dim=-1).mean()


def bom_loss(outputs: dict[str, Tensor], batch: dict[str, Tensor]) -> tuple[Tensor, dict[str, Tensor]]:
    component_loss = F.binary_cross_entropy_with_logits(outputs["component_logits"], batch["component_targets"])
    priority_loss = F.mse_loss(outputs["priority_logits"].sigmoid(), batch["priority_targets"])
    cost_loss = quantile_loss(outputs["cost_quantiles"], batch["cost_quantiles"])
    risk_loss = F.cross_entropy(outputs["risk_logits"], batch["risk_class"])
    total = component_loss + 0.45 * priority_loss + 2.5 * cost_loss + 0.55 * risk_loss
    return total, {
        "component": component_loss,
        "priority": priority_loss,
        "cost": cost_loss,
        "risk": risk_loss,
    }


@torch.no_grad()
def component_f1(logits: Tensor, targets: Tensor, threshold: float = 0.45) -> float:
    predicted = logits.sigmoid().ge(threshold)
    truth = targets.bool()
    true_positive = (predicted & truth).sum().float()
    precision = true_positive / predicted.sum().float().clamp(min=1)
    recall = true_positive / truth.sum().float().clamp(min=1)
    f1 = 2 * precision * recall / (precision + recall).clamp(min=1e-6)
    return float(f1.detach().cpu())


@torch.no_grad()
def risk_accuracy(logits: Tensor, targets: Tensor) -> float:
    return float((logits.argmax(dim=-1) == targets).float().mean().detach().cpu())


@torch.no_grad()
def mean_absolute_cost_error(predicted: Tensor, target: Tensor, cost_scale: float) -> float:
    return float((predicted - target).abs().mean().mul(cost_scale).detach().cpu())
