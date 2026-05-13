from __future__ import annotations

import json
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any


@dataclass
class TutorialResponseConfig:
    vocab_size: int
    max_source_length: int = 320
    max_target_length: int = 160
    d_model: int = 192
    n_heads: int = 6
    n_encoder_layers: int = 3
    n_decoder_layers: int = 3
    dim_feedforward: int = 768
    dropout: float = 0.1
    pad_token_id: int = 0
    bos_token_id: int = 2
    eos_token_id: int = 3

    def to_json(self) -> dict[str, Any]:
        return asdict(self)

    @classmethod
    def from_json(cls, data: dict[str, Any]) -> "TutorialResponseConfig":
        return cls(**data)

    def save(self, path: str | Path) -> None:
        Path(path).write_text(json.dumps(self.to_json(), ensure_ascii=False, indent=2), encoding="utf-8")

    @classmethod
    def load(cls, path: str | Path) -> "TutorialResponseConfig":
        return cls.from_json(json.loads(Path(path).read_text(encoding="utf-8")))

