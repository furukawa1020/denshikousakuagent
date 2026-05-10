from __future__ import annotations

import json
import random
from pathlib import Path
from typing import Any

import torch
from torch.utils.data import Dataset

from ai_models.makergraph.tokenizer import MakerTokenizer

from .taxonomy import BOARD_CLASSES, FIRMWARE_CLASSES, FIRMWARE_VARIANTS, PIN_PROFILES, PROJECTS, SAFETY_LABELS, DEBUG_CAUSES, index_or_default


PROJECT_DESCRIPTORS = {
    "light_charm": [
        "dark room led charm photoresistor A0 led D5 resistor usb breadboard",
        "kurakunaru to hikaru omamori led hikari sensor",
        "cute light accessory learns threshold and led polarity",
    ],
    "desk_pet": [
        "desk pet distance sensor buzzer led esp32 cute proximity reaction",
        "chikazuku to naku tsukue pet ultrasonic sensor buzzer",
        "small character reacts when a hand approaches",
    ],
    "plant_ping": [
        "plant watering notifier soil moisture sensor led buzzer usb",
        "mizuyari tsuuchi soil sensor dry plant reminder",
        "checks plant dryness and makes a small alert",
    ],
    "posture_guard": [
        "posture alert device distance sensor desk focus led buzzer",
        "shisei chuui distance sensor too close to desk",
        "focus helper warns when face is too near",
    ],
    "temp_face": [
        "temperature face display dht oled i2c m5stack room monitor",
        "ondo de hyoujou ga kawaru mini character screen",
        "environment monitor with face and temperature sensor",
    ],
}

DEBUG_PATTERNS = {
    "usb_port_or_driver": [
        "cannot upload no com port board not found usb cable charge only",
        "arduino ide says port unavailable upload timeout",
    ],
    "wrong_pin_mapping": [
        "code uses D5 but jumper is on D6 pin mapping mismatch",
        "serial starts but output part never reacts after changing gpio",
    ],
    "missing_gnd": [
        "sensor value fixed and gnd may not be connected common ground missing",
        "module has vcc and signal but no ground wire",
    ],
    "led_polarity_or_resistor": [
        "led does not light resistor missing anode cathode may be reversed",
        "led hot no resistor or wrong direction",
        "led does not light and resistor may be missing",
    ],
    "sensor_power_or_signal": [
        "sensor value does not change vcc signal threshold wiring suspicious",
        "analog value stuck at zero from soil or light sensor",
    ],
    "library_missing": [
        "compile error no such file library not found include missing",
        "dht oled library missing board package error",
    ],
    "brownout_power": [
        "device resets when motor or buzzer starts brownout power drops",
        "esp32 restarts after output turns on usb power weak",
    ],
    "buzzer_pin_or_polarity": [
        "buzzer does not sound polarity or signal pin may be wrong",
        "tone code runs but no sound from buzzer",
    ],
    "unknown": [
        "something does not work and no log has been checked",
        "it stopped suddenly no symptom detail",
    ],
}

SAFETY_PATTERNS = {
    "ok_low_voltage": ["usb 5v low voltage breadboard resistor common gnd beginner safe"],
    "missing_resistor_led": ["led directly to gpio no resistor current limit missing"],
    "gnd_not_shared": ["sensor signal connected but gnd not shared"],
    "voltage_mismatch": ["5v sensor signal into esp32 3.3v gpio level mismatch"],
    "motor_direct": ["motor connected directly to gpio without driver transistor"],
    "ac_mains": ["ac100v mains wall outlet relay household power direct"],
    "high_voltage": ["high voltage boost converter neon tube electric shock risk"],
    "lipo_charge": ["lipo 18650 self made charging circuit battery charger"],
    "water_power": ["water pump wet area near power plant watering leak"],
    "thermal_load": ["heater nichrome hot plate heat control fire risk"],
}


def generate_records(records_per_project: int = 1000, seed: int = 47) -> list[dict[str, Any]]:
    rng = random.Random(seed)
    records: list[dict[str, Any]] = []
    debug_causes = DEBUG_CAUSES
            safety_labels = SAFETY_LABELS
    for project in PROJECTS:
        for _ in range(records_per_project):
            board_label = rng.choice(BOARD_CLASSES)
            pin_profile = choose_pin_profile(board_label, rng)
            firmware_variant = choose_firmware_variant(project, rng)
            debug = rng.choice(debug_causes)
            base = rng.choice(PROJECT_DESCRIPTORS[project])
            debug_text = rng.choice(DEBUG_PATTERNS[debug])
            if rng.random() < 0.28:
                safety = ["ok_low_voltage"]
                safety_text = ""
            else:
                safety = choose_safety(rng)
                safety_text = " ".join(SAFETY_PATTERNS[label][0] for label in safety)
            budget = rng.choice(["budget 1000 yen", "budget 3000 yen", "budget 5000 yen", "budget 10000 yen"])
            board = board_label.replace("_", " ")
            text = " | ".join([
                f"projectId {project}",
                f"firmwareVariant {firmware_variant}",
                f"boardClass {board_label}",
                f"pinProfile {pin_profile}",
                base,
                board,
                budget,
                debug_text,
                safety_text,
            ])
            records.append({
                "text": text,
                "project": project,
                "debug": debug,
                "safety": safety,
                "risk": risk_from_safety(safety),
                "firmware": project,
                "firmware_variant": firmware_variant,
                "board": board_label,
                "pin_profile": pin_profile,
            })
        for _ in range(max(80, records_per_project // 5)):
            descriptor = rng.choice(PROJECT_DESCRIPTORS[project])
            board_label = default_board(project)
            pin_profile = choose_pin_profile(board_label, rng)
            firmware_variant = choose_firmware_variant(project, rng)
            for text in [
                f"projectId {project}",
                f"projectId: {project}",
                f"projectId {project} | {descriptor}",
                f"projectId {project} | firmwareVariant {firmware_variant} | boardClass {board_label} | pinProfile {pin_profile}",
            ]:
                records.append({
                    "text": text,
                    "project": project,
                    "debug": "unknown",
                    "safety": ["ok_low_voltage"],
                    "risk": "low",
                    "firmware": project,
                    "firmware_variant": firmware_variant,
                    "board": board_label,
                    "pin_profile": pin_profile,
                })
        hazard_labels = [
            "missing_resistor_led",
            "gnd_not_shared",
            "voltage_mismatch",
            "motor_direct",
            "water_power",
            "thermal_load",
            "ac_mains",
            "high_voltage",
            "lipo_charge",
        ]
        for label in hazard_labels:
            for _ in range(max(20, records_per_project // 16)):
                descriptor = rng.choice(PROJECT_DESCRIPTORS[project])
                hazard = SAFETY_PATTERNS[label][0]
                board_label = default_board(project)
                pin_profile = choose_pin_profile(board_label, rng)
                firmware_variant = choose_firmware_variant(project, rng)
                text_options = [
                    hazard,
                    f"projectId {project} | {hazard}",
                    f"projectId {project} | {descriptor} | firmwareVariant {firmware_variant} | boardClass {board_label} | pinProfile {pin_profile} | {hazard}",
                ]
                for text in text_options:
                    records.append({
                        "text": text,
                        "project": project,
                        "debug": "unknown",
                        "safety": [label],
                        "risk": risk_from_safety([label]),
                        "firmware": project,
                        "firmware_variant": firmware_variant,
                        "board": board_label,
                        "pin_profile": pin_profile,
                    })
        for debug_label in [label for label in debug_causes if label != "unknown"]:
            for _ in range(max(20, records_per_project // 20)):
                descriptor = rng.choice(PROJECT_DESCRIPTORS[project])
                symptom = rng.choice(DEBUG_PATTERNS[debug_label])
                board_label = default_board(project)
                pin_profile = choose_pin_profile(board_label, rng)
                firmware_variant = choose_firmware_variant(project, rng)
                for text in [
                    symptom,
                    f"projectId {project} | {symptom}",
                    f"projectId {project} | {descriptor} | firmwareVariant {firmware_variant} | boardClass {board_label} | pinProfile {pin_profile} | {symptom}",
                ]:
                    records.append({
                        "text": text,
                        "project": project,
                        "debug": debug_label,
                        "safety": ["ok_low_voltage"],
                        "risk": "low",
                        "firmware": project,
                        "firmware_variant": firmware_variant,
                        "board": board_label,
                        "pin_profile": pin_profile,
                    })
    rng.shuffle(records)
    return records


def choose_safety(rng: random.Random) -> list[str]:
    roll = rng.random()
    if roll < 0.55:
        labels = ["ok_low_voltage"]
        if rng.random() < 0.08:
            labels.append("gnd_not_shared")
        return labels
    if roll < 0.72:
        return [rng.choice(["missing_resistor_led", "gnd_not_shared", "voltage_mismatch"])]
    if roll < 0.9:
        return [rng.choice(["motor_direct", "water_power", "thermal_load"])]
    return [rng.choice(["ac_mains", "high_voltage", "lipo_charge"])]


def risk_from_safety(labels: list[str]) -> str:
    if any(label in labels for label in ["ac_mains", "high_voltage", "lipo_charge"]):
        return "blocked"
    if any(label in labels for label in ["motor_direct", "water_power", "thermal_load"]):
        return "high"
    if any(label in labels for label in ["missing_resistor_led", "gnd_not_shared", "voltage_mismatch"]):
        return "medium"
    return "low"


def choose_firmware_variant(project: str, rng: random.Random) -> str:
    variants = {
        "light_charm": ["light_charm_minimal", "light_charm_debug"],
        "desk_pet": ["desk_pet_led", "desk_pet_buzzer"],
        "plant_ping": ["plant_ping_led", "plant_ping_buzzer"],
        "posture_guard": ["posture_guard_led", "posture_guard_buzzer"],
        "temp_face": ["temp_face_serial", "temp_face_i2c_ready"],
    }
    return rng.choice(variants[project])


def default_board(project: str) -> str:
    if project in {"desk_pet", "posture_guard"}:
        return "esp32"
    if project == "temp_face":
        return "m5stack"
    if project == "light_charm":
        return "arduino_uno"
    return "esp32"


def choose_pin_profile(board: str, rng: random.Random) -> str:
    options = {
        "arduino_uno": ["arduino_default"],
        "esp32": ["esp32_default", "esp32_grove_safe"],
        "m5stack": ["m5stack_grove", "esp32_grove_safe"],
        "pico": ["pico_default"],
    }
    return rng.choice(options[board])


def collect_texts(records: list[dict[str, Any]]) -> list[str]:
    return [str(record["text"]) for record in records]


class NeuralAgentDataset(Dataset):
    def __init__(self, records: list[dict[str, Any]], tokenizer: MakerTokenizer, max_length: int) -> None:
        self.records = records
        self.tokenizer = tokenizer
        self.max_length = max_length

    def __len__(self) -> int:
        return len(self.records)

    def __getitem__(self, index: int) -> dict[str, torch.Tensor]:
        record = self.records[index]
        ids, mask = self.tokenizer.encode(str(record["text"]), self.max_length, "<log>")
        safety = torch.zeros(len(SAFETY_LABELS), dtype=torch.float32)
        for label in record["safety"]:
            safety[SAFETY_LABELS.index(label)] = 1.0
        return {
            "input_ids": torch.tensor(ids, dtype=torch.long),
            "attention_mask": torch.tensor(mask, dtype=torch.bool),
            "safety_targets": safety,
            "risk_class": torch.tensor(risk_index(record["risk"]), dtype=torch.long),
            "debug_class": torch.tensor(index_or_default(DEBUG_CAUSES, record["debug"], "unknown"), dtype=torch.long),
            "firmware_class": torch.tensor(index_or_default(FIRMWARE_CLASSES, record["firmware"], "light_charm"), dtype=torch.long),
            "firmware_variant_class": torch.tensor(index_or_default(FIRMWARE_VARIANTS, record["firmware_variant"], "light_charm_minimal"), dtype=torch.long),
            "board_class": torch.tensor(index_or_default(BOARD_CLASSES, record["board"], "esp32"), dtype=torch.long),
            "pin_profile_class": torch.tensor(index_or_default(PIN_PROFILES, record["pin_profile"], "esp32_default"), dtype=torch.long),
        }


def collate_agent_batch(items: list[dict[str, torch.Tensor]]) -> dict[str, torch.Tensor]:
    keys = items[0].keys()
    return {key: torch.stack([item[key] for item in items]) for key in keys}


def risk_index(value: str) -> int:
    from .taxonomy import RISK_CLASSES

    return index_or_default(RISK_CLASSES, value, "medium")


def write_jsonl(path: str | Path, records: list[dict[str, Any]]) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as handle:
        for record in records:
            handle.write(json.dumps(record, ensure_ascii=False) + "\n")


def load_jsonl(path: str | Path) -> list[dict[str, Any]]:
    with Path(path).open("r", encoding="utf-8") as handle:
        return [json.loads(line) for line in handle if line.strip()]
