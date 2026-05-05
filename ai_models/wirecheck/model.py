from __future__ import annotations

import math
from typing import Any

import torch
from torch import Tensor, nn

from .config import WireCheckConfig


class PatchEmbedding(nn.Module):
    def __init__(self, config: WireCheckConfig) -> None:
        super().__init__()
        self.proj = nn.Conv2d(
            config.in_channels,
            config.d_model,
            kernel_size=config.patch_size,
            stride=config.patch_size,
        )
        self.norm = nn.LayerNorm(config.d_model)

    def forward(self, images: Tensor) -> Tensor:
        patches = self.proj(images).flatten(2).transpose(1, 2)
        return self.norm(patches)


class WireCheckNet(nn.Module):
    """CNN patch embedding + Transformer decoder for wiring graph recovery.

    Outputs:
    - part slots: class logits and normalized bbox cx,cy,w,h
    - wire slots: class logits and normalized endpoints x1,y1,x2,y2
    - risk logits: safety classification for beginner guidance
    """

    def __init__(self, config: WireCheckConfig) -> None:
        super().__init__()
        self.config = config
        self.patch_embedding = PatchEmbedding(config)
        self.position_embedding = nn.Parameter(torch.zeros(1, config.num_patches, config.d_model))
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=config.d_model,
            nhead=config.n_heads,
            dim_feedforward=config.dim_feedforward,
            dropout=config.dropout,
            activation="gelu",
            batch_first=True,
            norm_first=True,
        )
        self.encoder = nn.TransformerEncoder(encoder_layer, config.encoder_layers)

        total_queries = config.num_part_queries + config.num_wire_queries
        self.query_embedding = nn.Parameter(torch.randn(1, total_queries, config.d_model) * 0.02)
        decoder_layer = nn.TransformerDecoderLayer(
            d_model=config.d_model,
            nhead=config.n_heads,
            dim_feedforward=config.dim_feedforward,
            dropout=config.dropout,
            activation="gelu",
            batch_first=True,
            norm_first=True,
        )
        self.decoder = nn.TransformerDecoder(decoder_layer, config.decoder_layers)
        self.output_norm = nn.LayerNorm(config.d_model)

        self.part_class_head = nn.Linear(config.d_model, config.num_part_classes)
        self.part_box_head = mlp(config.d_model, config.d_model, 4, layers=3)
        self.wire_class_head = nn.Linear(config.d_model, config.num_wire_classes)
        self.wire_endpoint_head = mlp(config.d_model, config.d_model, 4, layers=3)
        self.risk_head = nn.Sequential(
            nn.LayerNorm(config.d_model),
            nn.Linear(config.d_model, config.d_model),
            nn.GELU(),
            nn.Dropout(config.dropout),
            nn.Linear(config.d_model, config.num_risk_classes),
        )
        self._reset_parameters()

    def _reset_parameters(self) -> None:
        nn.init.trunc_normal_(self.position_embedding, std=0.02)
        nn.init.trunc_normal_(self.query_embedding, std=0.02)

    def forward(self, images: Tensor) -> dict[str, Tensor]:
        memory = self.patch_embedding(images) + self.position_embedding
        memory = self.encoder(memory)
        queries = self.query_embedding.expand(images.size(0), -1, -1)
        decoded = self.decoder(queries, memory)
        decoded = self.output_norm(decoded)

        part_hidden = decoded[:, : self.config.num_part_queries]
        wire_hidden = decoded[:, self.config.num_part_queries :]
        pooled = memory.mean(dim=1)
        return {
            "part_logits": self.part_class_head(part_hidden),
            "part_boxes": self.part_box_head(part_hidden).sigmoid(),
            "wire_logits": self.wire_class_head(wire_hidden),
            "wire_endpoints": self.wire_endpoint_head(wire_hidden).sigmoid(),
            "risk_logits": self.risk_head(pooled),
        }


def mlp(input_dim: int, hidden_dim: int, output_dim: int, layers: int) -> nn.Sequential:
    modules: list[nn.Module] = []
    current = input_dim
    for _ in range(layers - 1):
        modules.extend([nn.Linear(current, hidden_dim), nn.GELU()])
        current = hidden_dim
    modules.append(nn.Linear(current, output_dim))
    return nn.Sequential(*modules)


def count_parameters(model: nn.Module) -> int:
    return sum(parameter.numel() for parameter in model.parameters() if parameter.requires_grad)


def load_checkpoint(checkpoint: str, map_location: str | torch.device = "cpu") -> tuple[WireCheckNet, WireCheckConfig, dict[str, Any]]:
    payload = torch.load(checkpoint, map_location=map_location)
    config = WireCheckConfig.from_json(payload["config"])
    model = WireCheckNet(config)
    model.load_state_dict(payload["model_state"])
    model.eval()
    return model, config, payload


@torch.no_grad()
def decode_outputs(outputs: dict[str, Tensor], config: WireCheckConfig, score_threshold: float = 0.35) -> dict[str, Any]:
    part_probs = outputs["part_logits"].softmax(dim=-1)[0]
    wire_probs = outputs["wire_logits"].softmax(dim=-1)[0]
    risk_probs = outputs["risk_logits"].softmax(dim=-1)[0]
    parts = []
    for index, probs in enumerate(part_probs):
        score, class_id = probs.max(dim=-1)
        class_name = config.part_classes[int(class_id)]
        if class_name == "none" or float(score) < score_threshold:
            continue
        parts.append({
            "slot": index,
            "class": class_name,
            "score": round(float(score), 4),
            "bbox": [round(float(value), 4) for value in outputs["part_boxes"][0, index].detach().cpu()],
        })

    wires = []
    for index, probs in enumerate(wire_probs):
        score, class_id = probs.max(dim=-1)
        class_name = config.wire_classes[int(class_id)]
        if class_name == "none" or float(score) < score_threshold:
            continue
        wires.append({
            "slot": index,
            "class": class_name,
            "score": round(float(score), 4),
            "endpoints": [round(float(value), 4) for value in outputs["wire_endpoints"][0, index].detach().cpu()],
        })

    risk_score, risk_id = risk_probs.max(dim=-1)
    return {
        "parts": parts,
        "wires": wires,
        "risk": {
            "class": config.risk_classes[int(risk_id)],
            "score": round(float(risk_score), 4),
        },
    }
