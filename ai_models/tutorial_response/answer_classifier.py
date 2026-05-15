from __future__ import annotations

import json
import math
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any

import torch
from torch import Tensor, nn


ANSWER_KIND_LABELS = [
    "confirmed",
    "negative",
    "unknown",
    "photo_request",
    "problem_report",
    "inventory_report",
    "board_report",
]


@dataclass
class TutorialAnswerClassifierConfig:
    vocab_size: int
    max_source_length: int = 320
    d_model: int = 160
    n_heads: int = 5
    n_layers: int = 3
    dim_feedforward: int = 640
    dropout: float = 0.1
    pad_token_id: int = 0

    def to_json(self) -> dict[str, Any]:
        return asdict(self)

    @classmethod
    def from_json(cls, data: dict[str, Any]) -> "TutorialAnswerClassifierConfig":
        return cls(**data)

    def save(self, path: str | Path) -> None:
        Path(path).write_text(json.dumps(self.to_json(), ensure_ascii=False, indent=2), encoding="utf-8")


class TutorialAnswerClassifier(nn.Module):
    def __init__(self, config: TutorialAnswerClassifierConfig, num_labels: int = len(ANSWER_KIND_LABELS)) -> None:
        super().__init__()
        self.config = config
        self.token_embedding = nn.Embedding(config.vocab_size, config.d_model, padding_idx=config.pad_token_id)
        self.position = nn.Embedding(config.max_source_length, config.d_model)
        layer = nn.TransformerEncoderLayer(
            d_model=config.d_model,
            nhead=config.n_heads,
            dim_feedforward=config.dim_feedforward,
            dropout=config.dropout,
            activation="gelu",
            batch_first=True,
            norm_first=True,
        )
        self.encoder = nn.TransformerEncoder(layer, num_layers=config.n_layers)
        self.norm = nn.LayerNorm(config.d_model)
        self.classifier = nn.Linear(config.d_model, num_labels)

    def forward(self, source_ids: Tensor, source_mask: Tensor) -> Tensor:
        batch_size, source_len = source_ids.shape
        positions = torch.arange(source_len, device=source_ids.device).unsqueeze(0).expand(batch_size, source_len)
        hidden = (self.token_embedding(source_ids) + self.position(positions)) * math.sqrt(self.config.d_model)
        encoded = self.encoder(hidden, src_key_padding_mask=~source_mask)
        mask = source_mask.unsqueeze(-1).float()
        pooled = (encoded * mask).sum(dim=1) / mask.sum(dim=1).clamp_min(1.0)
        return self.classifier(self.norm(pooled))


def count_parameters(model: nn.Module) -> int:
    return sum(parameter.numel() for parameter in model.parameters() if parameter.requires_grad)


def load_checkpoint(checkpoint: str | Path, map_location: str | torch.device = "cpu") -> tuple[TutorialAnswerClassifier, TutorialAnswerClassifierConfig, dict[str, Any]]:
    payload = torch.load(checkpoint, map_location=map_location)
    config = TutorialAnswerClassifierConfig.from_json(payload["config"])
    model = TutorialAnswerClassifier(config, num_labels=len(payload.get("labels") or ANSWER_KIND_LABELS))
    model.load_state_dict(payload["model_state"])
    model.eval()
    return model, config, payload
