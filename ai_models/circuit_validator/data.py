from __future__ import annotations

import json
import random
from pathlib import Path
from typing import Any

import torch
from torch.utils.data import Dataset

from ai_models.makergraph.tokenizer import MakerTokenizer

from .taxonomy import BOARD_CLASSES, CIRCUIT_ISSUES, REPAIR_ACTIONS, RISK_CLASSES


BOARD_PROFILES = {
    "arduino_uno": {"logic": "5V", "pins": ["D2", "D3", "D4", "D5", "D6", "D7", "A0"]},
    "esp32": {"logic": "3.3V", "pins": ["GPIO5", "GPIO18", "GPIO21", "GPIO25", "GPIO26", "GPIO34"]},
    "m5stack": {"logic": "3.3V", "pins": ["GPIO2", "GPIO26", "GPIO35", "GPIO36"]},
    "pico": {"logic": "3.3V", "pins": ["GP14", "GP15", "GP16", "GP26"]},
}

ISSUE_TEXT = {
    "ok": "LED has 220ohm resistor, sensor VCC matches board logic, all modules share GND, unique GPIO pins",
    "gnd_not_shared": "sensor signal connected but sensor GND is missing, module ground not connected to board ground",
    "led_missing_resistor": "LED anode connected directly to GPIO, no 220ohm resistor, current limit missing",
    "voltage_mismatch": "5V sensor signal connected to ESP32 3.3V GPIO without level shifter",
    "pin_conflict": "LED and buzzer both assigned to same GPIO pin, duplicate pin mapping conflict",
    "motor_direct_gpio": "DC motor connected directly to GPIO pin without transistor or motor driver",
    "over_current": "multiple LEDs and buzzer draw too much current from one GPIO or board 3V3 pin",
    "floating_input": "button or sensor input has no pullup and no pulldown, input can float",
    "reversed_polarity": "LED cathode and anode reversed, electrolytic capacitor polarity reversed",
}


def generate_records(samples: int = 12000, seed: int = 61) -> list[dict[str, Any]]:
    rng = random.Random(seed)
    records: list[dict[str, Any]] = []
    issue_weights = [
        ("ok", 0.25),
        ("gnd_not_shared", 0.13),
        ("led_missing_resistor", 0.13),
        ("voltage_mismatch", 0.11),
        ("pin_conflict", 0.10),
        ("motor_direct_gpio", 0.10),
        ("over_current", 0.07),
        ("floating_input", 0.06),
        ("reversed_polarity", 0.05),
    ]
    for _ in range(samples):
        board = rng.choice(BOARD_CLASSES)
        issue = weighted_choice(issue_weights, rng)
        issues = [issue]
        if issue != "ok" and rng.random() < 0.22:
            extra = rng.choice([item for item in CIRCUIT_ISSUES if item not in {"ok", issue}])
            issues.append(extra)
        text = build_circuit_text(board, issues, rng)
        records.append({
            "text": text,
            "board": board,
            "issues": issues,
            "risk": risk_for(issues),
            "repair": repair_for(issues[0]),
        })
    rng.shuffle(records)
    return records


def build_circuit_text(board: str, issues: list[str], rng: random.Random) -> str:
    profile = BOARD_PROFILES[board]
    pins = profile["pins"]
    led_pin = rng.choice(pins)
    sensor_pin = rng.choice(pins)
    buzzer_pin = rng.choice(pins)
    if "pin_conflict" not in issues and sensor_pin == led_pin:
        sensor_pin = pins[(pins.index(led_pin) + 1) % len(pins)]
    if "pin_conflict" in issues:
        buzzer_pin = led_pin
    rows = [
        f"board: {board} logic:{profile['logic']}",
        f"connection: LED anode -> {led_pin}; LED cathode -> GND; resistor: {'missing' if 'led_missing_resistor' in issues else '220ohm'}",
        f"connection: distance_or_soil_sensor VCC -> {'5V' if 'voltage_mismatch' in issues else profile['logic']}; GND -> {'none' if 'gnd_not_shared' in issues else 'GND'}; SIG -> {sensor_pin}",
        f"connection: buzzer plus -> {buzzer_pin}; buzzer minus -> GND",
    ]
    if "motor_direct_gpio" in issues:
        rows.append(f"connection: DC motor plus -> {rng.choice(pins)}; motor minus -> GND; driver: none")
    if "over_current" in issues:
        rows.append("load: 4 LEDs plus buzzer from one GPIO, estimated current 90mA")
    if "floating_input" in issues:
        rows.append(f"connection: push button -> {rng.choice(pins)}; pullup: none; pulldown: none")
    if "reversed_polarity" in issues:
        rows.append("orientation: LED cathode on signal side, capacitor plus to GND")
    rows.append("observed: " + " | ".join(ISSUE_TEXT[issue] for issue in issues))
    return "\n".join(rows)


def risk_for(issues: list[str]) -> str:
    if "motor_direct_gpio" in issues:
        return "blocked"
    if any(issue in issues for issue in ["over_current", "voltage_mismatch"]):
        return "high"
    if any(issue in issues for issue in ["gnd_not_shared", "led_missing_resistor", "pin_conflict", "floating_input", "reversed_polarity"]):
        return "medium"
    return "low"


def repair_for(issue: str) -> str:
    return {
        "ok": "none",
        "gnd_not_shared": "connect_common_gnd",
        "led_missing_resistor": "add_led_resistor",
        "voltage_mismatch": "use_level_shifter_or_3v3_part",
        "pin_conflict": "move_to_free_gpio",
        "motor_direct_gpio": "add_motor_driver",
        "over_current": "separate_power_supply",
        "floating_input": "add_pullup_or_pulldown",
        "reversed_polarity": "flip_polarity",
    }[issue]


def weighted_choice(items: list[tuple[str, float]], rng: random.Random) -> str:
    total = sum(weight for _item, weight in items)
    cursor = rng.random() * total
    for item, weight in items:
        cursor -= weight
        if cursor <= 0:
            return item
    return items[-1][0]


def collect_texts(records: list[dict[str, Any]]) -> list[str]:
    return [str(record["text"]) for record in records]


class CircuitDataset(Dataset):
    def __init__(self, records: list[dict[str, Any]], tokenizer: MakerTokenizer, max_length: int) -> None:
        self.records = records
        self.tokenizer = tokenizer
        self.max_length = max_length

    def __len__(self) -> int:
        return len(self.records)

    def __getitem__(self, index: int) -> dict[str, torch.Tensor]:
        record = self.records[index]
        ids, mask = self.tokenizer.encode(str(record["text"]), self.max_length, "<graph>")
        issue_targets = torch.zeros(len(CIRCUIT_ISSUES), dtype=torch.float32)
        for issue in record["issues"]:
            issue_targets[CIRCUIT_ISSUES.index(issue)] = 1.0
        return {
            "input_ids": torch.tensor(ids, dtype=torch.long),
            "attention_mask": torch.tensor(mask, dtype=torch.bool),
            "issue_targets": issue_targets,
            "risk_class": torch.tensor(RISK_CLASSES.index(record["risk"]), dtype=torch.long),
            "repair_class": torch.tensor(REPAIR_ACTIONS.index(record["repair"]), dtype=torch.long),
            "board_class": torch.tensor(BOARD_CLASSES.index(record["board"]), dtype=torch.long),
        }


def collate_circuit_batch(items: list[dict[str, torch.Tensor]]) -> dict[str, torch.Tensor]:
    return {key: torch.stack([item[key] for item in items]) for key in items[0].keys()}


def write_jsonl(path: str | Path, records: list[dict[str, Any]]) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as handle:
        for record in records:
            handle.write(json.dumps(record, ensure_ascii=False) + "\n")


def load_jsonl(path: str | Path) -> list[dict[str, Any]]:
    with Path(path).open("r", encoding="utf-8") as handle:
        return [json.loads(line) for line in handle if line.strip()]

