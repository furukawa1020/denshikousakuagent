from __future__ import annotations

import torch
from torch import Tensor
import torch.nn.functional as F


def symmetric_contrastive_loss(logits: Tensor) -> Tensor:
    labels = torch.arange(logits.size(0), device=logits.device)
    query_to_project = F.cross_entropy(logits, labels)
    project_to_query = F.cross_entropy(logits.T, labels)
    return (query_to_project + project_to_query) * 0.5


def profile_regression_loss(predicted: Tensor, targets: Tensor) -> Tensor:
    return F.mse_loss(predicted, targets)


@torch.no_grad()
def retrieval_accuracy(logits: Tensor, top_k: int = 1) -> float:
    labels = torch.arange(logits.size(0), device=logits.device)
    _, indices = logits.topk(k=min(top_k, logits.size(1)), dim=1)
    return (indices == labels.unsqueeze(1)).any(dim=1).float().mean().item()
