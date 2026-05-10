from __future__ import annotations

import math
from typing import Any

import torch
from torch import Tensor, nn

from .config import InventoryMatcherConfig


class InventoryProjectMatcher(nn.Module):
    """Transformer that maps inventory and constraints to feasible maker projects."""

    def __init__(self, config: InventoryMatcherConfig) -> None:
        super().__init__()
        self.config = config
        self.token_embedding = nn.Embedding(config.vocab_size, config.d_model, padding_idx=config.pad_token_id)
        self.position_embedding = nn.Embedding(config.max_length, config.d_model)
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
        self.pool = AttentionPool(config.d_model)
        self.norm = nn.LayerNorm(config.d_model)
        self.project_head = nn.Linear(config.d_model, config.project_count)
        self.completion_head = nn.Linear(config.d_model, config.project_count)
        self.budget_fit_head = nn.Linear(config.d_model, config.project_count)
        self.missing_component_head = nn.Linear(config.d_model, config.component_count)
        self.tier_head = nn.Linear(config.d_model, config.tier_count)

    def forward(self, input_ids: Tensor, attention_mask: Tensor) -> dict[str, Tensor]:
        batch_size, seq_len = input_ids.shape
        positions = torch.arange(seq_len, device=input_ids.device).unsqueeze(0).expand(batch_size, seq_len)
        hidden = (self.token_embedding(input_ids) + self.position_embedding(positions)) * math.sqrt(self.config.d_model)
        hidden = self.encoder(hidden, src_key_padding_mask=~attention_mask)
        pooled = self.norm(self.pool(hidden, attention_mask))
        return {
            "project_logits": self.project_head(pooled),
            "completion_logits": self.completion_head(pooled),
            "budget_fit_logits": self.budget_fit_head(pooled),
            "missing_component_logits": self.missing_component_head(pooled),
            "tier_logits": self.tier_head(pooled),
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


def load_checkpoint(checkpoint: str, map_location: str | torch.device = "cpu") -> tuple[InventoryProjectMatcher, InventoryMatcherConfig, dict[str, Any]]:
    payload = torch.load(checkpoint, map_location=map_location)
    config = InventoryMatcherConfig.from_json(payload["config"])
    model = InventoryProjectMatcher(config)
    model.load_state_dict(payload["model_state"])
    model.eval()
    return model, config, payload

