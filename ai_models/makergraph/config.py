from __future__ import annotations

import json
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any


@dataclass
class MakerIntentConfig:
    vocab_size: int
    max_length: int = 256
    d_model: int = 256
    n_heads: int = 8
    n_layers: int = 4
    dim_feedforward: int = 1024
    dropout: float = 0.1
    embedding_dim: int = 128
    type_vocab_size: int = 4
    pad_token_id: int = 0
    temperature: float = 0.07

    def to_json(self) -> dict[str, Any]:
        return asdict(self)

    @classmethod
    def from_json(cls, data: dict[str, Any]) -> "MakerIntentConfig":
        return cls(**data)

    def save(self, path: str | Path) -> None:
        Path(path).write_text(json.dumps(self.to_json(), ensure_ascii=False, indent=2), encoding="utf-8")

    @classmethod
    def load(cls, path: str | Path) -> "MakerIntentConfig":
        return cls.from_json(json.loads(Path(path).read_text(encoding="utf-8")))


DEFAULT_DOMAIN_TERMS = [
    "Lチカ",
    "電子工作",
    "Arduino",
    "ESP32",
    "M5Stack",
    "M5StickC",
    "Raspberry Pi Pico",
    "micro:bit",
    "CircuitPython",
    "LED",
    "NeoPixel",
    "GND",
    "GPIO",
    "PWM",
    "I2C",
    "UART",
    "SPI",
    "USB",
    "ブレッドボード",
    "ジャンパ線",
    "抵抗",
    "光センサー",
    "距離センサー",
    "温湿度センサー",
    "土壌水分センサー",
    "ブザー",
    "サーボモーター",
    "リチウムイオン",
    "AC100V",
    "電源",
    "回路",
    "配線",
    "コード",
    "デバッグ",
    "失敗ログ",
    "作品グラフ",
    "回路グラフ",
    "最小構成",
    "標準構成",
    "拡張構成",
    "かわいい",
    "便利",
    "研究っぽい",
    "部屋に置きたい",
    "友達に見せたい",
]
