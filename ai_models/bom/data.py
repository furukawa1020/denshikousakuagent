from __future__ import annotations

import json
import random
from pathlib import Path
from typing import Any

import torch
from torch.utils.data import Dataset

from ai_models.makergraph.sample_data import PROJECT_CATALOG, project_to_document
from ai_models.makergraph.tokenizer import MakerTokenizer

from .component_catalog import (
    COMPONENT_BY_ID,
    COMPONENT_IDS,
    PROJECT_COMPONENTS,
    PROJECT_OPTIONALS,
    RISK_CLASSES,
    component_cost_quantiles,
    project_risk,
)


def generate_bom_records(records_per_project: int, seed: int = 45) -> list[dict[str, Any]]:
    rng = random.Random(seed)
    project_by_id = {project["id"]: project for project in PROJECT_CATALOG}
    records: list[dict[str, Any]] = []
    for project_id, required in PROJECT_COMPONENTS.items():
        project = project_by_id[project_id]
        optionals = PROJECT_OPTIONALS.get(project_id, [])
        for _ in range(records_per_project):
            tier = rng.choices(["minimal", "standard", "extended"], weights=[0.45, 0.38, 0.17], k=1)[0]
            component_ids = required[:]
            if tier in {"standard", "extended"}:
                component_ids += [component_id for component_id in optionals if component_id == "case_material"]
                if project_id in {"desk_pet", "plant_ping"} and "buzzer" in optionals:
                    component_ids.append("buzzer")
            if tier == "extended":
                component_ids += optionals
            component_ids = unique(component_ids)
            owned = choose_inventory(component_ids, rng)
            missing = [component_id for component_id in component_ids if component_id not in owned]
            q10, q50, q90 = jitter_quantiles(component_cost_quantiles(missing), rng)
            budget = rng.choice([1000, 3000, 5000, 10000, 20000, None])
            soldering = rng.choice([False, False, True])
            text = build_input_text(project, tier, component_ids, owned, budget, soldering)
            records.append({
                "project_id": project_id,
                "input_text": text,
                "tier": tier,
                "required_components": component_ids,
                "owned_components": owned,
                "missing_components": missing,
                "cost_quantiles": [q10, q50, q90],
                "budget": budget,
                "soldering_allowed": soldering,
                "risk": project_risk(project_id, component_ids),
            })
    rng.shuffle(records)
    return records


def write_bom_jsonl(path: str | Path, records: list[dict[str, Any]]) -> None:
    target = Path(path)
    target.parent.mkdir(parents=True, exist_ok=True)
    with target.open("w", encoding="utf-8") as handle:
        for record in records:
            handle.write(json.dumps(record, ensure_ascii=False) + "\n")


def load_bom_jsonl(path: str | Path) -> list[dict[str, Any]]:
    records = []
    with Path(path).open("r", encoding="utf-8") as handle:
        for line in handle:
            line = line.strip()
            if line:
                records.append(json.loads(line))
    return records


def collect_bom_texts(records: list[dict[str, Any]]) -> list[str]:
    return [record["input_text"] for record in records]


class BOMDataset(Dataset):
    def __init__(self, records: list[dict[str, Any]], tokenizer: MakerTokenizer, max_length: int, cost_scale: float) -> None:
        self.records = records
        self.tokenizer = tokenizer
        self.max_length = max_length
        self.cost_scale = cost_scale

    def __len__(self) -> int:
        return len(self.records)

    def __getitem__(self, index: int) -> dict[str, Any]:
        record = self.records[index]
        input_ids, mask = self.tokenizer.encode(record["input_text"], self.max_length, "<project>")
        component_targets = torch.zeros(len(COMPONENT_IDS), dtype=torch.float32)
        priority_targets = torch.zeros(len(COMPONENT_IDS), dtype=torch.float32)
        missing = record["missing_components"]
        for rank, component_id in enumerate(missing):
            component_targets[COMPONENT_IDS.index(component_id)] = 1.0
            priority_targets[COMPONENT_IDS.index(component_id)] = 1.0 / (rank + 1)
        return {
            "input_ids": torch.tensor(input_ids, dtype=torch.long),
            "attention_mask": torch.tensor(mask, dtype=torch.bool),
            "component_targets": component_targets,
            "priority_targets": priority_targets,
            "cost_quantiles": torch.tensor(record["cost_quantiles"], dtype=torch.float32) / self.cost_scale,
            "risk_class": torch.tensor(RISK_CLASSES.index(record["risk"]), dtype=torch.long),
            "project_id": record["project_id"],
            "input_text": record["input_text"],
        }


def collate_bom_batch(items: list[dict[str, Any]]) -> dict[str, Any]:
    return {
        "input_ids": torch.stack([item["input_ids"] for item in items]),
        "attention_mask": torch.stack([item["attention_mask"] for item in items]),
        "component_targets": torch.stack([item["component_targets"] for item in items]),
        "priority_targets": torch.stack([item["priority_targets"] for item in items]),
        "cost_quantiles": torch.stack([item["cost_quantiles"] for item in items]),
        "risk_class": torch.stack([item["risk_class"] for item in items]),
        "project_id": [item["project_id"] for item in items],
        "input_text": [item["input_text"] for item in items],
    }


def build_input_text(project: dict[str, Any], tier: str, component_ids: list[str], owned: list[str], budget: int | None, soldering: bool) -> str:
    missing_hint = "owned: " + ", ".join(COMPONENT_BY_ID[component_id].name for component_id in owned) if owned else "owned: none"
    component_hint = "components: " + ", ".join(COMPONENT_BY_ID[component_id].name for component_id in component_ids)
    budget_hint = f"budget_limit: {budget} yen" if budget else "budget_limit: unknown"
    soldering_hint = f"soldering_allowed: {str(soldering).lower()}"
    return "\n".join([
        project_to_document(project),
        f"bom_tier: {tier}",
        component_hint,
        missing_hint,
        budget_hint,
        soldering_hint,
        "must_include: USB cable, jumper wires, resistor for LED, breadboard unless owned",
        "safety: AC100V excluded, LiPo charging excluded, motor direct drive excluded",
    ])


def choose_inventory(component_ids: list[str], rng: random.Random) -> list[str]:
    owned = []
    for component_id in component_ids:
        probability = 0.2
        if component_id in {"led_5mm", "resistor_220", "breadboard", "jumper_wires", "usb_cable"}:
            probability = 0.45
        if rng.random() < probability:
            owned.append(component_id)
    return owned


def jitter_quantiles(values: tuple[int, int, int], rng: random.Random) -> tuple[int, int, int]:
    q10, q50, q90 = values
    multiplier = rng.uniform(0.92, 1.12)
    q10 = int(q10 * multiplier)
    q50 = int(max(q10, q50 * multiplier))
    q90 = int(max(q50, q90 * multiplier))
    return q10, q50, q90


def unique(values: list[str]) -> list[str]:
    seen = []
    for value in values:
        if value not in seen:
            seen.append(value)
    return seen
