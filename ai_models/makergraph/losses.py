from __future__ import annotations

import torch
from torch import Tensor
import torch.nn.functional as F


def symmetric_contrastive_loss(logits: Tensor) -> Tensor:
    labels = torch.arange(logits.size(0), device=logits.device)
    query_to_project = F.cross_entropy(logits, labels)
    project_to_query = F.cross_entropy(logits.T, labels)
    return (query_to_project + project_to_query) * 0.5


def multi_positive_contrastive_loss(logits: Tensor, labels: list[str]) -> Tensor:
    """Contrastive loss where all examples with the same project id are positive.

    Synthetic maker-intent datasets naturally contain many phrasings for the
    same project. Treating only the diagonal pair as positive makes equivalent
    project descriptions false negatives. This loss fixes that.
    """

    positive_mask = make_positive_mask(labels, logits.device)
    query_to_project = masked_log_softmax_loss(logits, positive_mask)
    project_to_query = masked_log_softmax_loss(logits.T, positive_mask.T)
    return (query_to_project + project_to_query) * 0.5


def make_positive_mask(labels: list[str], device: torch.device) -> Tensor:
    label_tensor = torch.arange(len(labels), device=device)
    mask = torch.zeros(len(labels), len(labels), dtype=torch.bool, device=device)
    for row, label in enumerate(labels):
        for col, other in enumerate(labels):
            if label == other:
                mask[row, col] = True
    return mask


def masked_log_softmax_loss(logits: Tensor, positive_mask: Tensor) -> Tensor:
    log_probs = F.log_softmax(logits, dim=1)
    masked_log_probs = log_probs.masked_fill(~positive_mask, -1e4)
    return -torch.logsumexp(masked_log_probs, dim=1).mean()


def profile_regression_loss(predicted: Tensor, targets: Tensor) -> Tensor:
    return F.mse_loss(predicted, targets)


@torch.no_grad()
def retrieval_accuracy(logits: Tensor, top_k: int = 1) -> float:
    labels = torch.arange(logits.size(0), device=logits.device)
    _, indices = logits.topk(k=min(top_k, logits.size(1)), dim=1)
    return (indices == labels.unsqueeze(1)).any(dim=1).float().mean().item()


@torch.no_grad()
def labeled_retrieval_accuracy(logits: Tensor, labels: list[str], top_k: int = 1) -> float:
    _, indices = logits.topk(k=min(top_k, logits.size(1)), dim=1)
    correct = []
    for row, row_indices in enumerate(indices.detach().cpu().tolist()):
        correct.append(any(labels[col] == labels[row] for col in row_indices))
    return float(sum(correct) / max(1, len(correct)))
