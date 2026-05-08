from __future__ import annotations

import json
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any


@dataclass
class BOMEstimatorConfig:
    vocab_size: int
    component_count: int
    risk_count: int
    max_length: int = 256
    d_model: int = 192
    n_heads: int = 6
    n_layers: int = 4
    dim_feedforward: int = 768
    dropout: float = 0.1
    pad_token_id: int = 0
    cost_scale: float = 20000.0

    def to_json(self) -> dict[str, Any]:
        return asdict(self)

    @classmethod
    def from_json(cls, data: dict[str, Any]) -> "BOMEstimatorConfig":
        return cls(**data)

    def save(self, path: str | Path) -> None:
        Path(path).write_text(json.dumps(self.to_json(), ensure_ascii=False, indent=2), encoding="utf-8")

    @classmethod
    def load(cls, path: str | Path) -> "BOMEstimatorConfig":
        return cls.from_json(json.loads(Path(path).read_text(encoding="utf-8")))
