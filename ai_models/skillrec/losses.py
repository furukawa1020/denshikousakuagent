from __future__ import annotations

import torch
from torch import Tensor
import torch.nn.functional as F


def skillrec_loss(outputs: dict[str, Tensor], batch: dict[str, Tensor]) -> tuple[Tensor, dict[str, Tensor]]:
    skill_loss = F.mse_loss(outputs["skill_vector"], batch["target_skill_vector"])
    confidence_target = 1.0 - (outputs["skill_vector"].detach() - batch["target_skill_vector"]).abs().clamp(0, 1)
    confidence_loss = F.mse_loss(outputs["confidence"], confidence_target)
    next_skill_loss = F.cross_entropy(outputs["next_skill_logits"], batch["next_skill"])
    project_loss = F.cross_entropy(outputs["project_scores"], batch["target_project"])
    completion_loss = F.mse_loss(outputs["completion_probability"], project_one_hot(batch["target_project"], outputs["completion_probability"].size(1)))
    total = skill_loss * 2.2 + confidence_loss * 0.35 + next_skill_loss * 0.45 + project_loss * 0.9 + completion_loss * 0.25
    return total, {
        "skill": skill_loss,
        "confidence": confidence_loss,
        "next_skill": next_skill_loss,
        "project": project_loss,
        "completion": completion_loss,
    }


def project_one_hot(indices: Tensor, count: int) -> Tensor:
    return F.one_hot(indices, num_classes=count).float()


@torch.no_grad()
def skill_mae(predicted: Tensor, target: Tensor) -> float:
    return float((predicted - target).abs().mean().detach().cpu())


@torch.no_grad()
def class_accuracy(logits: Tensor, target: Tensor) -> float:
    return float((logits.argmax(dim=-1) == target).float().mean().detach().cpu())
