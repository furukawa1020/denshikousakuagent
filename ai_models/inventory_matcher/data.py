from __future__ import annotations

import json
import random
from pathlib import Path
from typing import Any

import torch
from torch.utils.data import Dataset

from ai_models.bom.component_catalog import COMPONENT_BY_ID, COMPONENT_IDS, PROJECT_COMPONENTS, component_cost_quantiles
from ai_models.makergraph.tokenizer import MakerTokenizer

from .taxonomy import MATCH_TIERS, PROJECT_IDS, project_index, tier_index


INTERESTS = ["かわいい", "便利", "植物", "集中", "部屋", "光る", "机上", "研究っぽい"]
BASE_COMPONENTS = ["led_5mm", "resistor_220", "breadboard", "jumper_wires", "usb_cable"]


def generate_records(samples: int = 14000, seed: int = 71) -> list[dict[str, Any]]:
    rng = random.Random(seed)
    records = []
    all_components = COMPONENT_IDS[:]
    for _ in range(samples):
        budget = rng.choice([0, 1000, 3000, 5000, 10000, 20000])
        interest = rng.choice(INTERESTS)
        owned = choose_inventory(all_components, rng)
        if rng.random() < 0.45:
            owned.extend(rng.sample(BASE_COMPONENTS, k=rng.randint(1, len(BASE_COMPONENTS))))
        owned = unique(owned)
        fit = score_projects(owned, budget, interest)
        best_project = max(fit, key=lambda item: item["score"])
        missing = [component_id for component_id in PROJECT_COMPONENTS[best_project["projectId"]] if component_id not in owned]
        text = build_input_text(owned, budget, interest)
        records.append({
            "text": text,
            "owned": owned,
            "budget": budget,
            "interest": interest,
            "project": best_project["projectId"],
            "tier": best_project["tier"],
            "missing": missing,
            "completion_targets": [item["completion"] for item in fit],
            "budget_targets": [item["budgetFit"] for item in fit],
        })
    rng.shuffle(records)
    return records


def choose_inventory(all_components: list[str], rng: random.Random) -> list[str]:
    owned = []
    for component_id in all_components:
        probability = 0.18
        if component_id in BASE_COMPONENTS:
            probability = 0.38
        if component_id in {"esp32_devkit", "arduino_uno"}:
            probability = 0.30
        if component_id in {"m5stickc", "servo_sg90", "oled_display"}:
            probability = 0.12
        if rng.random() < probability:
            owned.append(component_id)
    return owned


def score_projects(owned: list[str], budget: int, interest: str) -> list[dict[str, Any]]:
    owned_set = set(owned)
    scored = []
    for project_id in PROJECT_IDS:
        required = PROJECT_COMPONENTS[project_id]
        missing = [component_id for component_id in required if component_id not in owned_set]
        q10, q50, q90 = component_cost_quantiles(missing)
        reuse = len(set(required) & owned_set) / max(1, len(required))
        budget_fit = 1.0 if budget == 0 and q50 == 0 else max(0.0, min(1.0, 1.15 - q50 / max(800, budget or 800)))
        interest_fit = project_interest_fit(project_id, interest)
        difficulty_penalty = {"light_charm": 0.05, "desk_pet": 0.16, "plant_ping": 0.17, "posture_guard": 0.28, "temp_face": 0.32}[project_id]
        completion = max(0.05, min(0.98, 0.35 + reuse * 0.38 + budget_fit * 0.26 + interest_fit * 0.16 - difficulty_penalty))
        score = completion + interest_fit * 0.18 + reuse * 0.12
        if q50 == 0:
            tier = "ready_now"
        elif budget and q50 <= budget:
            tier = "buy_small"
        elif budget and q50 <= budget * 1.35:
            tier = "budget_stretch"
        else:
            tier = "too_hard"
        scored.append({"projectId": project_id, "score": score, "completion": completion, "budgetFit": budget_fit, "tier": tier})
    return scored


def project_interest_fit(project_id: str, interest: str) -> float:
    table = {
        "desk_pet": {"かわいい": 0.95, "机上": 0.9, "光る": 0.55, "便利": 0.45},
        "light_charm": {"光る": 0.95, "かわいい": 0.78, "部屋": 0.7},
        "temp_face": {"研究っぽい": 0.82, "部屋": 0.74, "かわいい": 0.64},
        "plant_ping": {"植物": 0.97, "便利": 0.78, "研究っぽい": 0.52},
        "posture_guard": {"集中": 0.95, "便利": 0.75, "机上": 0.62},
    }
    return table.get(project_id, {}).get(interest, 0.35)


def build_input_text(owned: list[str], budget: int, interest: str) -> str:
    owned_names = ", ".join(COMPONENT_BY_ID[component_id].name for component_id in owned) if owned else "none"
    owned_ids = ", ".join(owned) if owned else "none"
    return "\n".join([
        f"inventory_ids: {owned_ids}",
        f"inventory_names: {owned_names}",
        f"budget_limit: {budget} yen",
        f"interest: {interest}",
        "task: choose feasible beginner electronics project, predict missing parts and budget fit",
    ])


def collect_texts(records: list[dict[str, Any]]) -> list[str]:
    return [record["text"] for record in records]


class InventoryDataset(Dataset):
    def __init__(self, records: list[dict[str, Any]], tokenizer: MakerTokenizer, max_length: int) -> None:
        self.records = records
        self.tokenizer = tokenizer
        self.max_length = max_length

    def __len__(self) -> int:
        return len(self.records)

    def __getitem__(self, index: int) -> dict[str, torch.Tensor]:
        record = self.records[index]
        ids, mask = self.tokenizer.encode(record["text"], self.max_length, "<component>")
        missing_targets = torch.zeros(len(COMPONENT_IDS), dtype=torch.float32)
        for component_id in record["missing"]:
            missing_targets[COMPONENT_IDS.index(component_id)] = 1.0
        return {
            "input_ids": torch.tensor(ids, dtype=torch.long),
            "attention_mask": torch.tensor(mask, dtype=torch.bool),
            "project_class": torch.tensor(project_index(record["project"]), dtype=torch.long),
            "tier_class": torch.tensor(tier_index(record["tier"]), dtype=torch.long),
            "missing_targets": missing_targets,
            "completion_targets": torch.tensor(record["completion_targets"], dtype=torch.float32),
            "budget_targets": torch.tensor(record["budget_targets"], dtype=torch.float32),
        }


def collate_inventory_batch(items: list[dict[str, torch.Tensor]]) -> dict[str, torch.Tensor]:
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


def unique(values: list[str]) -> list[str]:
    seen = []
    for value in values:
        if value not in seen:
            seen.append(value)
    return seen

