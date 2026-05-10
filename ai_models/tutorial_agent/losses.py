from __future__ import annotations

import torch
from torch import Tensor
import torch.nn.functional as F


def tutorial_loss(outputs: dict[str, Tensor], batch: dict[str, Tensor]) -> tuple[Tensor, dict[str, Tensor]]:
    project = F.cross_entropy(outputs["project_logits"], batch["project_class"])
    stage = F.cross_entropy(outputs["stage_logits"], batch["stage_class"])
    action = F.cross_entropy(outputs["action_logits"], batch["action_class"])
    question = F.cross_entropy(outputs["question_logits"], batch["question_class"])
    checkpoint = F.cross_entropy(outputs["checkpoint_logits"], batch["checkpoint_class"])
    route = F.cross_entropy(outputs["route_logits"], batch["route_class"])
    concepts = F.binary_cross_entropy_with_logits(outputs["concept_logits"], batch["concept_targets"])
    autonomy = F.mse_loss(outputs["autonomy_logits"].sigmoid(), batch["autonomy_target"])
    loss = project * 0.7 + stage * 1.4 + action * 1.2 + question + checkpoint + route * 0.8 + concepts * 1.1 + autonomy * 0.8
    return loss, {
        "project": project,
        "stage": stage,
        "action": action,
        "question": question,
        "checkpoint": checkpoint,
        "route": route,
        "concepts": concepts,
        "autonomy": autonomy,
    }


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
