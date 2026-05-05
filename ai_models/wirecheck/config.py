from __future__ import annotations

import json
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any


PART_CLASSES = [
    "none",
    "breadboard",
    "esp32",
    "arduino_uno",
    "led",
    "resistor",
    "buzzer",
    "distance_sensor",
    "light_sensor",
    "soil_sensor",
    "usb_cable",
]


WIRE_CLASSES = [
    "none",
    "gnd",
    "vcc",
    "signal",
]


RISK_CLASSES = [
    "safe",
    "retake_needed",
    "missing_gnd",
    "missing_resistor",
    "voltage_mismatch",
    "unsafe_power",
]


@dataclass
class WireCheckConfig:
    image_size: int = 224
    patch_size: int = 16
    in_channels: int = 3
    d_model: int = 192
    n_heads: int = 6
    encoder_layers: int = 4
    decoder_layers: int = 3
    dim_feedforward: int = 768
    dropout: float = 0.1
    num_part_queries: int = 12
    num_wire_queries: int = 16
    part_classes: list[str] = field(default_factory=lambda: PART_CLASSES[:])
    wire_classes: list[str] = field(default_factory=lambda: WIRE_CLASSES[:])
    risk_classes: list[str] = field(default_factory=lambda: RISK_CLASSES[:])

    @property
    def num_part_classes(self) -> int:
        return len(self.part_classes)

    @property
    def num_wire_classes(self) -> int:
        return len(self.wire_classes)

    @property
    def num_risk_classes(self) -> int:
        return len(self.risk_classes)

    @property
    def num_patches(self) -> int:
        return (self.image_size // self.patch_size) ** 2

    def to_json(self) -> dict[str, Any]:
        return asdict(self)

    @classmethod
    def from_json(cls, data: dict[str, Any]) -> "WireCheckConfig":
        return cls(**data)

    def save(self, path: str | Path) -> None:
        Path(path).write_text(json.dumps(self.to_json(), ensure_ascii=False, indent=2), encoding="utf-8")

    @classmethod
    def load(cls, path: str | Path) -> "WireCheckConfig":
        return cls.from_json(json.loads(Path(path).read_text(encoding="utf-8")))
