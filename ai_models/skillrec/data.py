from __future__ import annotations

import json
import random
from pathlib import Path
from typing import Any

import torch
from torch.utils.data import Dataset

from .taxonomy import (
    EVENT_TYPES,
    PROJECT_COMPONENTS,
    PROJECT_COST,
    PROJECT_DIFFICULTY,
    PROJECT_SKILLS,
    PROJECTS,
    SKILLS,
    project_index,
    skill_index,
)


def generate_skill_records(count: int, seed: int = 47) -> list[dict[str, Any]]:
    rng = random.Random(seed)
    records = []
    for user_id in range(count):
        skill_state = [rng.uniform(0.08, 0.42) for _ in SKILLS]
        inventory = set(rng.sample(["led_5mm", "resistor_220", "breadboard", "jumper_wires", "esp32_devkit", "arduino_uno"], k=rng.randint(1, 4)))
        interest = rng.choice(["かわいい", "便利", "研究", "生活改善", "見せたい"])
        events = []
        completed_projects = []
        current_project = choose_start_project(skill_state, interest, rng)
        for _step in range(rng.randint(18, 52)):
            project_skills = PROJECT_SKILLS[current_project]
            weak_skill = min(project_skills, key=lambda s: skill_state[skill_index(s)])
            event_type, outcome = sample_event(skill_state[skill_index(weak_skill)], rng)
            events.append({"skill": weak_skill, "event": event_type, "project": current_project, "outcome": outcome})
            update_skill(skill_state, weak_skill, event_type, outcome, rng)
            if event_type == "completed":
                completed_projects.append(current_project)
                inventory |= PROJECT_COMPONENTS[current_project]
                current_project = choose_next_project(skill_state, inventory, interest, completed_projects, rng)

        next_skill = choose_next_skill(skill_state)
        target_project = choose_next_project(skill_state, inventory, interest, completed_projects, rng)
        records.append({
            "user_id": f"user_{user_id}",
            "events": events,
            "target_skill_vector": [round(value, 4) for value in skill_state],
            "next_skill": next_skill,
            "target_project": target_project,
            "inventory": sorted(inventory),
            "interest": interest,
            "completed_projects": completed_projects,
        })
    return records


def write_skill_jsonl(path: str | Path, records: list[dict[str, Any]]) -> None:
    target = Path(path)
    target.parent.mkdir(parents=True, exist_ok=True)
    with target.open("w", encoding="utf-8") as handle:
        for record in records:
            handle.write(json.dumps(record, ensure_ascii=False) + "\n")


def load_skill_jsonl(path: str | Path) -> list[dict[str, Any]]:
    records = []
    with Path(path).open("r", encoding="utf-8") as handle:
        for line in handle:
            line = line.strip()
            if line:
                records.append(json.loads(line))
    return records


class SkillRecDataset(Dataset):
    def __init__(self, records: list[dict[str, Any]], max_events: int) -> None:
        self.records = records
        self.max_events = max_events

    def __len__(self) -> int:
        return len(self.records)

    def __getitem__(self, index: int) -> dict[str, Any]:
        record = self.records[index]
        events = record["events"][-self.max_events:]
        skill_ids = []
        event_ids = []
        project_ids = []
        outcomes = []
        for event in events:
            skill_ids.append(skill_index(event["skill"]) + 1)
            event_ids.append(EVENT_TYPES.index(event["event"]) + 1)
            project_ids.append(project_index(event["project"]) + 1)
            outcomes.append(int(event["outcome"]) + 1)
        attention_mask = [1] * len(skill_ids)
        while len(skill_ids) < self.max_events:
            skill_ids.append(0)
            event_ids.append(0)
            project_ids.append(0)
            outcomes.append(0)
            attention_mask.append(0)
        return {
            "skill_ids": torch.tensor(skill_ids, dtype=torch.long),
            "event_ids": torch.tensor(event_ids, dtype=torch.long),
            "project_ids": torch.tensor(project_ids, dtype=torch.long),
            "outcomes": torch.tensor(outcomes, dtype=torch.long),
            "attention_mask": torch.tensor(attention_mask, dtype=torch.bool),
            "target_skill_vector": torch.tensor(record["target_skill_vector"], dtype=torch.float32),
            "next_skill": torch.tensor(skill_index(record["next_skill"]), dtype=torch.long),
            "target_project": torch.tensor(project_index(record["target_project"]), dtype=torch.long),
        }


def collate_skill_batch(items: list[dict[str, Any]]) -> dict[str, Any]:
    keys = ["skill_ids", "event_ids", "project_ids", "outcomes", "attention_mask", "target_skill_vector", "next_skill", "target_project"]
    return {key: torch.stack([item[key] for item in items]) for key in keys}


def choose_start_project(skill_state: list[float], interest: str, rng: random.Random) -> str:
    if interest in {"かわいい", "見せたい"}:
        return rng.choice(["light_charm", "desk_pet"])
    if interest == "生活改善":
        return rng.choice(["plant_ping", "posture_guard"])
    if interest == "研究":
        return rng.choice(["temp_face", "plant_ping"])
    return rng.choice(PROJECTS[:3])


def choose_next_project(skill_state: list[float], inventory: set[str], interest: str, completed: list[str], rng: random.Random) -> str:
    scores = {}
    for project in PROJECTS:
        project_skill_values = [skill_state[skill_index(skill)] for skill in PROJECT_SKILLS[project]]
        completion = sum(project_skill_values) / len(project_skill_values)
        reuse = len(inventory & PROJECT_COMPONENTS[project]) / max(1, len(PROJECT_COMPONENTS[project]))
        novelty = 0.2 if project in completed[-2:] else 0.55
        difficulty_penalty = abs(PROJECT_DIFFICULTY[project] / 3 - min(1.0, completion + 0.22))
        interest_bonus = project_interest_bonus(project, interest)
        scores[project] = completion * 0.36 + reuse * 0.22 + novelty * 0.12 + interest_bonus - difficulty_penalty * 0.24 - PROJECT_COST[project] / 50000
    ranked = sorted(PROJECTS, key=lambda project: scores[project], reverse=True)
    if rng.random() < 0.86:
        return ranked[0]
    return rng.choice(ranked[:3])


def project_interest_bonus(project: str, interest: str) -> float:
    if interest == "かわいい" and project in {"desk_pet", "temp_face", "light_charm"}:
        return 0.18
    if interest == "生活改善" and project in {"plant_ping", "posture_guard"}:
        return 0.2
    if interest == "研究" and project in {"temp_face", "plant_ping"}:
        return 0.17
    if interest == "見せたい" and project in {"desk_pet", "temp_face"}:
        return 0.16
    return 0.05


def choose_next_skill(skill_state: list[float]) -> str:
    sorted_indices = sorted(range(len(SKILLS)), key=lambda index: skill_state[index])
    return SKILLS[sorted_indices[0]]


def sample_event(skill_value: float, rng: random.Random) -> tuple[str, int]:
    if skill_value < 0.25:
        choices = [("question", 0), ("hint_viewed", 0), ("wiring_error", 0), ("debug_success", 1)]
        weights = [0.28, 0.24, 0.24, 0.24]
    elif skill_value < 0.55:
        choices = [("question", 0), ("code_edit", 1), ("wiring_check", 1), ("debug_success", 1), ("completed", 1)]
        weights = [0.18, 0.2, 0.2, 0.22, 0.2]
    else:
        choices = [("code_edit", 1), ("wiring_check", 1), ("completed", 1), ("project_started", 1)]
        weights = [0.24, 0.2, 0.36, 0.2]
    return rng.choices(choices, weights=weights, k=1)[0]


def update_skill(skill_state: list[float], skill: str, event_type: str, outcome: int, rng: random.Random) -> None:
    index = skill_index(skill)
    if outcome:
        gain = 0.025 if event_type != "completed" else 0.06
    else:
        gain = -0.015 if event_type in {"wiring_error", "compile_error"} else 0.005
    skill_state[index] = min(1.0, max(0.0, skill_state[index] + gain + rng.uniform(-0.006, 0.012)))
    if event_type == "debug_success":
        debug_index = skill_index("debugging")
        skill_state[debug_index] = min(1.0, skill_state[debug_index] + 0.03)
