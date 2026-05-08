from __future__ import annotations

import math
from typing import Any

import torch
from torch import Tensor, nn

from .config import SkillRecConfig


class SkillRecTransformer(nn.Module):
    """DKT-style Transformer for maker skill state and next-project ranking."""

    def __init__(self, config: SkillRecConfig) -> None:
        super().__init__()
        self.config = config
        self.skill_embedding = nn.Embedding(config.skill_count + 1, config.d_model, padding_idx=0)
        self.event_embedding = nn.Embedding(config.event_count + 1, config.d_model, padding_idx=0)
        self.project_embedding = nn.Embedding(config.project_count + 1, config.d_model, padding_idx=0)
        self.outcome_embedding = nn.Embedding(3, config.d_model, padding_idx=0)
        self.position_embedding = nn.Embedding(config.max_events, config.d_model)
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=config.d_model,
            nhead=config.n_heads,
            dim_feedforward=config.dim_feedforward,
            dropout=config.dropout,
            activation="gelu",
            batch_first=True,
            norm_first=True,
        )
        self.encoder = nn.TransformerEncoder(encoder_layer, config.n_layers)
        self.norm = nn.LayerNorm(config.d_model)
        self.pool = AttentionPool(config.d_model)
        self.skill_head = nn.Sequential(nn.Linear(config.d_model, config.d_model), nn.GELU(), nn.Linear(config.d_model, config.skill_count), nn.Sigmoid())
        self.confidence_head = nn.Sequential(nn.Linear(config.d_model, config.skill_count), nn.Sigmoid())
        self.next_skill_head = nn.Linear(config.d_model, config.skill_count)
        self.project_score_head = nn.Linear(config.d_model, config.project_count)
        self.completion_head = nn.Linear(config.d_model, config.project_count)
        self.learning_gain_head = nn.Linear(config.d_model, config.project_count)

    def forward(self, skill_ids: Tensor, event_ids: Tensor, project_ids: Tensor, outcomes: Tensor, attention_mask: Tensor) -> dict[str, Tensor]:
        batch_size, seq_len = skill_ids.shape
        positions = torch.arange(seq_len, device=skill_ids.device).unsqueeze(0).expand(batch_size, seq_len)
        hidden = (
            self.skill_embedding(skill_ids)
            + self.event_embedding(event_ids)
            + self.project_embedding(project_ids)
            + self.outcome_embedding(outcomes)
            + self.position_embedding(positions)
        ) * math.sqrt(self.config.d_model)
        hidden = self.encoder(hidden, src_key_padding_mask=~attention_mask)
        pooled = self.norm(self.pool(hidden, attention_mask))
        return {
            "skill_vector": self.skill_head(pooled),
            "confidence": self.confidence_head(pooled),
            "next_skill_logits": self.next_skill_head(pooled),
            "project_scores": self.project_score_head(pooled),
            "completion_probability": self.completion_head(pooled).sigmoid(),
            "learning_gain": self.learning_gain_head(pooled).sigmoid(),
        }


class AttentionPool(nn.Module):
    def __init__(self, d_model: int) -> None:
        super().__init__()
        self.score = nn.Sequential(nn.Linear(d_model, d_model), nn.Tanh(), nn.Linear(d_model, 1))

    def forward(self, hidden: Tensor, attention_mask: Tensor) -> Tensor:
        scores = self.score(hidden).squeeze(-1).masked_fill(~attention_mask, -1e4)
        weights = scores.softmax(dim=-1).unsqueeze(-1)
        return (hidden * weights).sum(dim=1)


def count_parameters(model: nn.Module) -> int:
    return sum(parameter.numel() for parameter in model.parameters() if parameter.requires_grad)


def load_checkpoint(checkpoint: str, map_location: str | torch.device = "cpu") -> tuple[SkillRecTransformer, SkillRecConfig, dict[str, Any]]:
    payload = torch.load(checkpoint, map_location=map_location)
    config = SkillRecConfig.from_json(payload["config"])
    model = SkillRecTransformer(config)
    model.load_state_dict(payload["model_state"])
    model.eval()
    return model, config, payload
