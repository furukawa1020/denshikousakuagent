from __future__ import annotations

import random
from dataclasses import dataclass
from typing import Any

import torch
from torch.utils.data import Dataset

from .config import WireCheckConfig


@dataclass
class SyntheticCircuit:
    name: str
    risk: str
    parts: list[tuple[str, tuple[float, float, float, float]]]
    wires: list[tuple[str, tuple[float, float, float, float]]]


class SyntheticWiringDataset(Dataset):
    """Procedural wiring images for bootstrapping WireCheckNet.

    This is not a replacement for annotated real photos. It gives the model a
    CUDA-trainable starting point and validates the object/wire graph pipeline.
    """

    def __init__(self, size: int, config: WireCheckConfig, seed: int = 42) -> None:
        self.size = size
        self.config = config
        self.seed = seed

    def __len__(self) -> int:
        return self.size

    def __getitem__(self, index: int) -> dict[str, Any]:
        rng = random.Random(self.seed + index)
        circuit = make_circuit(rng)
        image = render_circuit(circuit, self.config.image_size, rng)
        labels = encode_labels(circuit, self.config)
        labels["image"] = image
        labels["circuit_name"] = circuit.name
        return labels


def make_circuit(rng: random.Random) -> SyntheticCircuit:
    board_type = rng.choice(["esp32", "arduino_uno"])
    sensor = rng.choice(["distance_sensor", "light_sensor", "soil_sensor"])
    include_buzzer = rng.random() < 0.55
    risk = rng.choices(
        ["safe", "missing_gnd", "missing_resistor", "voltage_mismatch", "retake_needed"],
        weights=[0.58, 0.12, 0.14, 0.08, 0.08],
        k=1,
    )[0]

    board = jitter_box((0.12, 0.32, 0.22, 0.32), rng, 0.025)
    breadboard = jitter_box((0.42, 0.2, 0.46, 0.55), rng, 0.025)
    led = jitter_box((0.7, 0.58, 0.07, 0.1), rng, 0.02)
    resistor = jitter_box((0.58, 0.58, 0.12, 0.04), rng, 0.02)
    sensor_box = jitter_box((0.68, 0.26, 0.16, 0.12), rng, 0.02)
    parts = [
        ("breadboard", breadboard),
        (board_type, board),
        ("led", led),
        ("resistor", resistor),
        (sensor, sensor_box),
        ("usb_cable", jitter_box((0.05, 0.12, 0.1, 0.12), rng, 0.02)),
    ]
    if include_buzzer:
        parts.append(("buzzer", jitter_box((0.5, 0.75, 0.1, 0.1), rng, 0.02)))

    wires = [
        ("vcc", endpoints(board, breadboard, 0.25, 0.18)),
        ("signal", endpoints(board, sensor_box, 0.75, 0.5)),
        ("signal", endpoints(board, led, 0.65, 0.5)),
    ]
    if risk != "missing_gnd":
        wires.append(("gnd", endpoints(board, breadboard, 0.25, 0.82)))
    if include_buzzer:
        wires.append(("signal", endpoints(board, parts[-1][1], 0.65, 0.5)))
    if risk == "missing_resistor":
        parts = [part for part in parts if part[0] != "resistor"]
    if risk == "voltage_mismatch":
        wires.append(("vcc", endpoints(board, sensor_box, 0.92, 0.18)))

    return SyntheticCircuit(name=f"{board_type}_{sensor}", risk=risk, parts=parts, wires=wires)


def render_circuit(circuit: SyntheticCircuit, image_size: int, rng: random.Random) -> torch.Tensor:
    image = torch.full((3, image_size, image_size), 0.88, dtype=torch.float32)
    add_noise(image, rng, strength=0.035)
    colors = {
        "breadboard": (0.94, 0.94, 0.9),
        "esp32": (0.08, 0.24, 0.34),
        "arduino_uno": (0.0, 0.46, 0.52),
        "led": (0.95, 0.74, 0.12),
        "resistor": (0.72, 0.45, 0.2),
        "buzzer": (0.12, 0.12, 0.12),
        "distance_sensor": (0.25, 0.3, 0.72),
        "light_sensor": (0.32, 0.5, 0.25),
        "soil_sensor": (0.18, 0.42, 0.24),
        "usb_cable": (0.18, 0.18, 0.18),
        "gnd": (0.02, 0.02, 0.02),
        "vcc": (0.88, 0.1, 0.06),
        "signal": (0.08, 0.32, 0.78),
    }
    for wire_class, points in circuit.wires:
        draw_line(image, points, colors[wire_class], width=3)
    for part_class, box in circuit.parts:
        draw_rect(image, box, colors[part_class], fill=True)
        draw_rect(image, box, (0.05, 0.06, 0.06), fill=False)
    if circuit.risk == "retake_needed":
        image = image * 0.55 + 0.2
    return image.clamp(0, 1)


def encode_labels(circuit: SyntheticCircuit, config: WireCheckConfig) -> dict[str, Any]:
    part_classes = torch.zeros(config.num_part_queries, dtype=torch.long)
    part_boxes = torch.zeros(config.num_part_queries, 4, dtype=torch.float32)
    for index, (part_class, box) in enumerate(circuit.parts[: config.num_part_queries]):
        part_classes[index] = config.part_classes.index(part_class)
        part_boxes[index] = torch.tensor(box, dtype=torch.float32)

    wire_classes = torch.zeros(config.num_wire_queries, dtype=torch.long)
    wire_endpoints = torch.zeros(config.num_wire_queries, 4, dtype=torch.float32)
    for index, (wire_class, endpoints_value) in enumerate(circuit.wires[: config.num_wire_queries]):
        wire_classes[index] = config.wire_classes.index(wire_class)
        wire_endpoints[index] = torch.tensor(endpoints_value, dtype=torch.float32)

    risk_class = torch.tensor(config.risk_classes.index(circuit.risk), dtype=torch.long)
    return {
        "part_classes": part_classes,
        "part_boxes": part_boxes,
        "wire_classes": wire_classes,
        "wire_endpoints": wire_endpoints,
        "risk_class": risk_class,
    }


def draw_rect(image: torch.Tensor, box: tuple[float, float, float, float], color: tuple[float, float, float], fill: bool) -> None:
    _, height, width = image.shape
    cx, cy, bw, bh = box
    x1 = max(0, int((cx - bw / 2) * width))
    x2 = min(width - 1, int((cx + bw / 2) * width))
    y1 = max(0, int((cy - bh / 2) * height))
    y2 = min(height - 1, int((cy + bh / 2) * height))
    color_tensor = torch.tensor(color, dtype=image.dtype).view(3, 1, 1)
    if fill:
        image[:, y1:y2, x1:x2] = color_tensor
    else:
        image[:, y1:y1 + 2, x1:x2] = color_tensor
        image[:, y2 - 2:y2, x1:x2] = color_tensor
        image[:, y1:y2, x1:x1 + 2] = color_tensor
        image[:, y1:y2, x2 - 2:x2] = color_tensor


def draw_line(image: torch.Tensor, endpoints_value: tuple[float, float, float, float], color: tuple[float, float, float], width: int = 2) -> None:
    _, height, image_width = image.shape
    x1, y1, x2, y2 = endpoints_value
    steps = max(8, int(max(abs(x2 - x1), abs(y2 - y1)) * image_width))
    color_tensor = torch.tensor(color, dtype=image.dtype).view(3, 1, 1)
    for step in range(steps + 1):
        ratio = step / steps
        x = int((x1 * (1 - ratio) + x2 * ratio) * image_width)
        y = int((y1 * (1 - ratio) + y2 * ratio) * height)
        xs = slice(max(0, x - width), min(image_width, x + width + 1))
        ys = slice(max(0, y - width), min(height, y + width + 1))
        image[:, ys, xs] = color_tensor


def endpoints(a: tuple[float, float, float, float], b: tuple[float, float, float, float], ax: float, by: float) -> tuple[float, float, float, float]:
    acx, acy, aw, ah = a
    bcx, bcy, bw, bh = b
    return (
        acx + (ax - 0.5) * aw,
        acy,
        bcx,
        bcy + (by - 0.5) * bh,
    )


def jitter_box(box: tuple[float, float, float, float], rng: random.Random, amount: float) -> tuple[float, float, float, float]:
    cx, cy, width, height = box
    return (
        clamp(cx + rng.uniform(-amount, amount)),
        clamp(cy + rng.uniform(-amount, amount)),
        clamp(width + rng.uniform(-amount, amount), 0.04, 0.7),
        clamp(height + rng.uniform(-amount, amount), 0.035, 0.7),
    )


def clamp(value: float, minimum: float = 0.03, maximum: float = 0.97) -> float:
    return min(maximum, max(minimum, value))


def add_noise(image: torch.Tensor, rng: random.Random, strength: float) -> None:
    generator = torch.Generator().manual_seed(rng.randint(0, 2**31 - 1))
    image += torch.randn(image.shape, generator=generator) * strength
