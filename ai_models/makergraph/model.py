from __future__ import annotations

import math
from typing import Any

import torch
from torch import Tensor, nn
import torch.nn.functional as F

from .config import MakerIntentConfig


class AttentionPooler(nn.Module):
    def __init__(self, d_model: int) -> None:
        super().__init__()
        self.score = nn.Sequential(
            nn.Linear(d_model, d_model),
            nn.Tanh(),
            nn.Linear(d_model, 1),
        )

    def forward(self, hidden: Tensor, attention_mask: Tensor) -> Tensor:
        scores = self.score(hidden).squeeze(-1)
        scores = scores.masked_fill(~attention_mask, -1e4)
        weights = torch.softmax(scores, dim=-1).unsqueeze(-1)
        return torch.sum(hidden * weights, dim=1)


class MakerIntentTransformer(nn.Module):
    """Transformer encoder for beginner maker intent.

    It maps both user intent text and project/component/graph documents into a
    shared normalized embedding. The auxiliary profile heads make the embedding
    useful for difficulty, budget, and novelty-aware recommendation.
    """

    def __init__(self, config: MakerIntentConfig) -> None:
        super().__init__()
        self.config = config
        self.token_embedding = nn.Embedding(config.vocab_size, config.d_model, padding_idx=config.pad_token_id)
        self.position_embedding = nn.Embedding(config.max_length, config.d_model)
        self.type_embedding = nn.Embedding(config.type_vocab_size, config.d_model)
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=config.d_model,
            nhead=config.n_heads,
            dim_feedforward=config.dim_feedforward,
            dropout=config.dropout,
            activation="gelu",
            batch_first=True,
            norm_first=True,
        )
        self.encoder = nn.TransformerEncoder(encoder_layer, num_layers=config.n_layers)
        self.final_norm = nn.LayerNorm(config.d_model)
        self.pooler = AttentionPooler(config.d_model)
        self.projection = nn.Sequential(
            nn.Linear(config.d_model, config.d_model),
            nn.GELU(),
            nn.LayerNorm(config.d_model),
            nn.Linear(config.d_model, config.embedding_dim),
        )
        self.profile_head = nn.Sequential(
            nn.Linear(config.d_model, config.d_model // 2),
            nn.GELU(),
            nn.Dropout(config.dropout),
            nn.Linear(config.d_model // 2, 3),
            nn.Sigmoid(),
        )
        self._reset_parameters()

    def _reset_parameters(self) -> None:
        nn.init.normal_(self.token_embedding.weight, mean=0.0, std=0.02)
        nn.init.normal_(self.position_embedding.weight, mean=0.0, std=0.02)
        nn.init.normal_(self.type_embedding.weight, mean=0.0, std=0.02)
        if self.token_embedding.padding_idx is not None:
            with torch.no_grad():
                self.token_embedding.weight[self.token_embedding.padding_idx].fill_(0)

    def forward(
        self,
        input_ids: Tensor,
        attention_mask: Tensor,
        token_type_ids: Tensor | None = None,
    ) -> dict[str, Tensor]:
        batch_size, seq_len = input_ids.shape
        if token_type_ids is None:
            token_type_ids = torch.zeros_like(input_ids)
        positions = torch.arange(seq_len, device=input_ids.device).unsqueeze(0).expand(batch_size, seq_len)

        hidden = (
            self.token_embedding(input_ids)
            + self.position_embedding(positions)
            + self.type_embedding(token_type_ids.clamp(min=0, max=self.config.type_vocab_size - 1))
        )
        hidden = hidden * math.sqrt(self.config.d_model)
        hidden = self.encoder(hidden, src_key_padding_mask=~attention_mask)
        hidden = self.final_norm(hidden)
        pooled = self.pooler(hidden, attention_mask)
        embedding = F.normalize(self.projection(pooled), dim=-1)
        profile = self.profile_head(pooled)
        return {"embedding": embedding, "profile": profile, "hidden": hidden, "pooled": pooled}


class MakerGraphDualEncoder(nn.Module):
    """Two-tower contrastive model for intent-to-project retrieval."""

    def __init__(self, config: MakerIntentConfig) -> None:
        super().__init__()
        self.encoder = MakerIntentTransformer(config)
        self.log_temperature = nn.Parameter(torch.tensor(math.log(config.temperature)))

    def encode(self, input_ids: Tensor, attention_mask: Tensor, tower_type: int) -> dict[str, Tensor]:
        token_type_ids = torch.full_like(input_ids, tower_type)
        return self.encoder(input_ids, attention_mask, token_type_ids)

    def forward(
        self,
        query_ids: Tensor,
        query_mask: Tensor,
        project_ids: Tensor,
        project_mask: Tensor,
    ) -> dict[str, Tensor]:
        query = self.encode(query_ids, query_mask, tower_type=1)
        project = self.encode(project_ids, project_mask, tower_type=2)
        temperature = self.log_temperature.exp().clamp(min=0.01, max=0.5)
        logits = query["embedding"] @ project["embedding"].T / temperature
        return {
            "logits": logits,
            "query_embedding": query["embedding"],
            "project_embedding": project["embedding"],
            "query_profile": query["profile"],
            "project_profile": project["profile"],
            "temperature": temperature,
        }


class ProjectGraphSequenceTransformer(nn.Module):
    """Transformer decoder for graph-token generation.

    This is the deep-learning version of Project Graph Generator. It receives
    intent tokens as source and learns to generate graph tokens such as
    input:distance_sensor, processing:esp32, output:led, safety:gnd_common.
    """

    def __init__(self, config: MakerIntentConfig) -> None:
        super().__init__()
        self.config = config
        self.source_embedding = nn.Embedding(config.vocab_size, config.d_model, padding_idx=config.pad_token_id)
        self.target_embedding = nn.Embedding(config.vocab_size, config.d_model, padding_idx=config.pad_token_id)
        self.position_embedding = nn.Embedding(config.max_length, config.d_model)
        self.transformer = nn.Transformer(
            d_model=config.d_model,
            nhead=config.n_heads,
            num_encoder_layers=max(1, config.n_layers // 2),
            num_decoder_layers=max(1, config.n_layers // 2),
            dim_feedforward=config.dim_feedforward,
            dropout=config.dropout,
            activation="gelu",
            batch_first=True,
            norm_first=True,
        )
        self.output = nn.Linear(config.d_model, config.vocab_size)

    def forward(self, source_ids: Tensor, source_mask: Tensor, target_ids: Tensor, target_mask: Tensor) -> Tensor:
        source = self._embed(self.source_embedding, source_ids)
        target = self._embed(self.target_embedding, target_ids)
        causal_mask = nn.Transformer.generate_square_subsequent_mask(target_ids.size(1), device=target_ids.device)
        hidden = self.transformer(
            source,
            target,
            tgt_mask=causal_mask,
            src_key_padding_mask=~source_mask,
            tgt_key_padding_mask=~target_mask,
        )
        return self.output(hidden)

    def _embed(self, embedding: nn.Embedding, ids: Tensor) -> Tensor:
        batch_size, seq_len = ids.shape
        positions = torch.arange(seq_len, device=ids.device).unsqueeze(0).expand(batch_size, seq_len)
        return (embedding(ids) + self.position_embedding(positions)) * math.sqrt(self.config.d_model)


def count_parameters(model: nn.Module) -> int:
    return sum(parameter.numel() for parameter in model.parameters() if parameter.requires_grad)


def load_checkpoint(checkpoint: str, map_location: str | torch.device = "cpu") -> tuple[MakerGraphDualEncoder, MakerIntentConfig, dict[str, Any]]:
    payload = torch.load(checkpoint, map_location=map_location)
    config = MakerIntentConfig.from_json(payload["config"])
    model = MakerGraphDualEncoder(config)
    model.load_state_dict(payload["model_state"])
    model.eval()
    return model, config, payload
