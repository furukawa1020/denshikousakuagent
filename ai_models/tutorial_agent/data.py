from __future__ import annotations

import json
import random
from pathlib import Path
from typing import Any

import torch
from torch.utils.data import Dataset

from ai_models.makergraph.tokenizer import MakerTokenizer

from .taxonomy import (
    ACTIONS,
    CHECKPOINTS,
    CONCEPTS,
    PROJECT_IDS,
    QUESTIONS,
    ROUTES,
    TUTORIAL_STAGES,
    index_or_default,
)


BOARDS = ["arduino_uno", "esp32", "m5stack", "pico"]
INVENTORIES = [
    "ESP32 LED 220ohm resistor breadboard jumper wires USB cable",
    "Arduino Uno LED resistor breadboard USB cable",
    "M5StickC Grove cable DHT sensor USB cable",
    "LED only",
    "ESP32 soil sensor LED breadboard jumper wires",
    "Arduino Uno light sensor LED resistor jumper wires",
    "Pico LED buzzer breadboard USB cable",
]
USER_TEXTS = [
    "作りたいものはまだ決まっていない。かわいいものがいい。",
    "予算5000円でLチカの次に進みたい。",
    "LEDが光らない。配線が不安。",
    "COMポートが出ない。書き込めない。",
    "センサー値がずっと同じで変わらない。",
    "checkpoint loaded と出たあと止まっている。",
    "部品は少し持っているけど何を買えばいいか分からない。",
    "標準版まで作れたので、次の改造をしたい。",
]


def generate_records(samples: int = 16000, seed: int = 89) -> list[dict[str, Any]]:
    rng = random.Random(seed)
    records: list[dict[str, Any]] = []
    for _ in range(samples):
        project = rng.choice(PROJECT_IDS)
        board = rng.choice(BOARDS)
        budget = rng.choice([0, 1000, 3000, 5000, 10000, 20000])
        inventory = rng.choice(INVENTORIES)
        user_text = rng.choice(USER_TEXTS)
        current_stage = rng.choices(
            TUTORIAL_STAGES,
            weights=[1.2, 1.1, 1.2, 1.1, 1.0, 1.4, 0.9, 0.7, 0.8, 0.6],
        )[0]
        symptom = choose_symptom(current_stage, user_text, rng)
        skill = generate_skill_vector(current_stage, symptom, rng)
        labels = label_record(project, current_stage, symptom, skill, budget, inventory, user_text)
        text = build_input_text({
            "projectId": project,
            "board": board,
            "budget": budget,
            "inventory": inventory,
            "currentStage": current_stage,
            "symptom": symptom,
            "text": user_text,
            "skillVector": skill,
        })
        records.append({"text": text, "project": project, **labels})
    rng.shuffle(records)
    return records


def choose_symptom(current_stage: str, user_text: str, rng: random.Random) -> str:
    raw = user_text.lower()
    if "光らない" in raw:
        return "led_not_lighting"
    if "ポート" in raw or "書き込めない" in raw:
        return "upload_failed"
    if "センサー" in raw:
        return "sensor_static"
    if "checkpoint loaded" in raw:
        return "app_stuck_after_checkpoint"
    if current_stage == "debug_triage":
        return rng.choice(["led_not_lighting", "upload_failed", "sensor_static", "buzzer_silent", "power_reset"])
    return "none"


def generate_skill_vector(current_stage: str, symptom: str, rng: random.Random) -> dict[str, float]:
    base = {
        "led_polarity": rng.uniform(0.25, 0.75),
        "resistor_usage": rng.uniform(0.25, 0.75),
        "gnd_common": rng.uniform(0.18, 0.7),
        "gpio": rng.uniform(0.25, 0.8),
        "analog_input": rng.uniform(0.1, 0.65),
        "firmware_upload": rng.uniform(0.2, 0.75),
        "serial_monitor": rng.uniform(0.15, 0.65),
        "project_decomposition": rng.uniform(0.25, 0.85),
        "enclosure_design": rng.uniform(0.1, 0.55),
    }
    if current_stage in {"standard_build", "enclosure", "extension", "completion_log"}:
        for key in ["led_polarity", "resistor_usage", "gnd_common", "gpio", "firmware_upload"]:
            base[key] = min(0.95, base[key] + rng.uniform(0.18, 0.32))
    if symptom == "led_not_lighting":
        base["led_polarity"] = min(base["led_polarity"], rng.uniform(0.1, 0.38))
        base["resistor_usage"] = min(base["resistor_usage"], rng.uniform(0.1, 0.45))
    if symptom == "sensor_static":
        base["analog_input"] = min(base["analog_input"], rng.uniform(0.08, 0.35))
        base["gnd_common"] = min(base["gnd_common"], rng.uniform(0.12, 0.4))
    if symptom == "upload_failed":
        base["firmware_upload"] = min(base["firmware_upload"], rng.uniform(0.08, 0.34))
    return {key: round(value, 3) for key, value in base.items()}


def label_record(project: str, current_stage: str, symptom: str, skill: dict[str, float], budget: int, inventory: str, user_text: str) -> dict[str, Any]:
    stage = current_stage
    action = "ask_single_question"
    question = "confirm_inventory"
    checkpoint = "not_started"
    route = "minimal"
    autonomy = 0.45
    concepts = ["project_decomposition"]

    if symptom != "none":
        stage = "debug_triage"
        action = "diagnose_one_cause"
        checkpoint = "needs_debug"
        route = "debug_recovery"
        autonomy = 0.32
        concepts = ["gnd_common", "gpio_pin_match", "serial_monitor"]
        if symptom == "led_not_lighting":
            question = "check_led_polarity" if skill.get("led_polarity", 0) < skill.get("resistor_usage", 0) else "check_resistor"
            concepts.extend(["led_polarity", "resistor_usage"])
        elif symptom == "upload_failed":
            question = "check_usb_port"
            concepts.extend(["firmware_upload"])
        elif symptom == "sensor_static":
            question = "check_serial_value"
            concepts.extend(["analog_threshold", "sensor_power", "gnd_common"])
        else:
            question = "check_gnd"
            concepts.extend(["safe_power"])
    elif current_stage == "orient":
        action = "show_three_choices"
        question = "confirm_budget" if not budget else "choose_mood"
        checkpoint = "not_started"
        concepts = ["project_decomposition", "bom_check"]
        autonomy = 0.72
    elif current_stage == "parts_check":
        action = "check_inventory"
        question = "confirm_inventory"
        checkpoint = "idea_selected"
        concepts = ["bom_check", "safe_power"]
        route = "minimal" if budget <= 5000 else "standard"
        autonomy = 0.68
    elif current_stage == "minimal_circuit":
        action = "guide_wiring"
        question = "check_gnd" if skill.get("gnd_common", 0) < 0.45 else "check_resistor"
        checkpoint = "parts_ready"
        concepts = ["breadboard_rows", "led_polarity", "resistor_usage", "gnd_common", "gpio_pin_match"]
        autonomy = 0.63
    elif current_stage == "firmware_upload":
        action = "generate_test_firmware"
        question = "confirm_board"
        checkpoint = "wired_minimal"
        concepts = ["firmware_upload", "gpio_pin_match", "serial_monitor"]
        autonomy = 0.6
    elif current_stage == "observe_serial":
        action = "request_serial_log"
        question = "check_serial_value"
        checkpoint = "code_uploaded"
        concepts = ["serial_monitor", "analog_threshold"]
        autonomy = 0.56
    elif current_stage == "standard_build":
        action = "guide_wiring"
        question = "check_gnd"
        checkpoint = "observed_signal"
        route = "standard"
        concepts = ["safe_power", "sensor_power", "gnd_common"]
        autonomy = 0.58
    elif current_stage == "enclosure":
        action = "ask_single_question"
        question = "choose_next_extension"
        checkpoint = "standard_done"
        route = "standard"
        concepts = ["enclosure_fixing", "safe_power"]
        autonomy = 0.66
    elif current_stage == "extension":
        action = "unlock_next_project"
        question = "choose_next_extension"
        checkpoint = "standard_done"
        route = "extension"
        concepts = ["next_project_link", "analog_threshold", "enclosure_fixing"]
        autonomy = 0.75
    elif current_stage == "completion_log":
        action = "save_build_log"
        question = "choose_next_extension"
        checkpoint = "completed"
        route = "extension"
        concepts = ["next_project_link", "project_decomposition"]
        autonomy = 0.82

    if "LED" in inventory or "led" in inventory.lower():
        concepts.append("led_polarity")
    if "breadboard" in inventory.lower():
        concepts.append("breadboard_rows")
    if project == "plant_ping":
        concepts.append("safe_power")
        concepts.append("analog_threshold")
    if project == "temp_face":
        concepts.append("serial_monitor")
    return {
        "stage": stage,
        "action": action,
        "question": question,
        "checkpoint": checkpoint,
        "route": route,
        "concepts": sorted(set(concepts)),
        "autonomy": round(autonomy, 3),
    }


def build_input_text(payload: dict[str, Any]) -> str:
    skill = payload.get("skillVector") or payload.get("skills") or {}
    if not isinstance(skill, dict):
        skill = {}
    skill_text = ", ".join(f"{key}:{float(value):.2f}" for key, value in sorted(skill.items()) if isinstance(value, int | float))
    parts = [
        f"project_id: {payload.get('projectId') or payload.get('project') or 'unknown'}",
        f"board: {payload.get('board') or payload.get('boardClass') or 'unknown'}",
        f"budget: {payload.get('budget') or payload.get('budgetLimit') or 'unknown'}",
        f"inventory: {payload.get('inventory') or payload.get('owned') or payload.get('components') or 'unknown'}",
        f"current_stage: {payload.get('currentStage') or payload.get('stage') or payload.get('checkpoint') or 'unknown'}",
        f"symptom: {payload.get('symptom') or payload.get('issue') or 'none'}",
        f"user_text: {payload.get('text') or payload.get('message') or ''}",
        f"skill_vector: {skill_text}",
        "task: autonomously choose next beginner electronics tutorial step, question, route, and concepts",
    ]
    return "\n".join(parts)


def collect_texts(records: list[dict[str, Any]]) -> list[str]:
    return [record["text"] for record in records]


class TutorialDataset(Dataset):
    def __init__(self, records: list[dict[str, Any]], tokenizer: MakerTokenizer, max_length: int) -> None:
        self.records = records
        self.tokenizer = tokenizer
        self.max_length = max_length

    def __len__(self) -> int:
        return len(self.records)

    def __getitem__(self, index: int) -> dict[str, torch.Tensor]:
        record = self.records[index]
        ids, mask = self.tokenizer.encode(record["text"], self.max_length, "<tutorial>")
        concepts = torch.zeros(len(CONCEPTS), dtype=torch.float32)
        for concept in record["concepts"]:
            concepts[index_or_default(CONCEPTS, concept, "project_decomposition")] = 1.0
        return {
            "input_ids": torch.tensor(ids, dtype=torch.long),
            "attention_mask": torch.tensor(mask, dtype=torch.bool),
            "project_class": torch.tensor(index_or_default(PROJECT_IDS, record["project"], "desk_pet"), dtype=torch.long),
            "stage_class": torch.tensor(index_or_default(TUTORIAL_STAGES, record["stage"], "orient"), dtype=torch.long),
            "action_class": torch.tensor(index_or_default(ACTIONS, record["action"], "ask_single_question"), dtype=torch.long),
            "question_class": torch.tensor(index_or_default(QUESTIONS, record["question"], "confirm_inventory"), dtype=torch.long),
            "checkpoint_class": torch.tensor(index_or_default(CHECKPOINTS, record["checkpoint"], "not_started"), dtype=torch.long),
            "route_class": torch.tensor(index_or_default(ROUTES, record["route"], "minimal"), dtype=torch.long),
            "concept_targets": concepts,
            "autonomy_target": torch.tensor(float(record["autonomy"]), dtype=torch.float32),
        }


def collate_tutorial_batch(items: list[dict[str, torch.Tensor]]) -> dict[str, torch.Tensor]:
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
