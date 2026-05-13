from __future__ import annotations

import math
from typing import Any

import torch
from torch import Tensor, nn

from .config import TutorialResponseConfig


class TutorialResponseTransformer(nn.Module):
    """Encoder-decoder Transformer that generates tutorial replies from free answers."""

    def __init__(self, config: TutorialResponseConfig) -> None:
        super().__init__()
        self.config = config
        self.token_embedding = nn.Embedding(config.vocab_size, config.d_model, padding_idx=config.pad_token_id)
        self.source_position = nn.Embedding(config.max_source_length, config.d_model)
        self.target_position = nn.Embedding(config.max_target_length, config.d_model)
        self.transformer = nn.Transformer(
            d_model=config.d_model,
            nhead=config.n_heads,
            num_encoder_layers=config.n_encoder_layers,
            num_decoder_layers=config.n_decoder_layers,
            dim_feedforward=config.dim_feedforward,
            dropout=config.dropout,
            activation="gelu",
            batch_first=True,
            norm_first=True,
        )
        self.norm = nn.LayerNorm(config.d_model)
        self.output_head = nn.Linear(config.d_model, config.vocab_size)

    def forward(
        self,
        source_ids: Tensor,
        source_mask: Tensor,
        decoder_input_ids: Tensor,
        decoder_mask: Tensor | None = None,
    ) -> Tensor:
        batch_size, source_len = source_ids.shape
        target_len = decoder_input_ids.shape[1]
        source_pos = torch.arange(source_len, device=source_ids.device).unsqueeze(0).expand(batch_size, source_len)
        target_pos = torch.arange(target_len, device=source_ids.device).unsqueeze(0).expand(batch_size, target_len)
        source = (self.token_embedding(source_ids) + self.source_position(source_pos)) * math.sqrt(self.config.d_model)
        target = (self.token_embedding(decoder_input_ids) + self.target_position(target_pos)) * math.sqrt(self.config.d_model)
        causal_mask = nn.Transformer.generate_square_subsequent_mask(target_len, device=source_ids.device)
        hidden = self.transformer(
            source,
            target,
            tgt_mask=causal_mask,
            src_key_padding_mask=~source_mask,
            tgt_key_padding_mask=~decoder_mask if decoder_mask is not None else decoder_input_ids.eq(self.config.pad_token_id),
            memory_key_padding_mask=~source_mask,
        )
        return self.output_head(self.norm(hidden))

    @torch.no_grad()
    def generate(self, source_ids: Tensor, source_mask: Tensor, max_new_tokens: int = 140) -> Tensor:
        batch_size = source_ids.shape[0]
        generated = torch.full((batch_size, 1), self.config.bos_token_id, dtype=torch.long, device=source_ids.device)
        for _ in range(max_new_tokens):
            decoder_mask = ~generated.eq(self.config.pad_token_id)
            logits = self.forward(source_ids, source_mask, generated, decoder_mask)
            next_token = logits[:, -1].argmax(dim=-1, keepdim=True)
            generated = torch.cat([generated, next_token], dim=1)
            if bool((next_token == self.config.eos_token_id).all()):
                break
        return generated


def count_parameters(model: nn.Module) -> int:
    return sum(parameter.numel() for parameter in model.parameters() if parameter.requires_grad)


def load_checkpoint(checkpoint: str, map_location: str | torch.device = "cpu") -> tuple[TutorialResponseTransformer, TutorialResponseConfig, dict[str, Any]]:
    payload = torch.load(checkpoint, map_location=map_location)
    config = TutorialResponseConfig.from_json(payload["config"])
    model = TutorialResponseTransformer(config)
    model.load_state_dict(payload["model_state"])
    model.eval()
    return model, config, payload

