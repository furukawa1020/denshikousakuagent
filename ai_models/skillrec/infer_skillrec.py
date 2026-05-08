from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

import torch

from .model import load_checkpoint
from .taxonomy import EVENT_TYPES, PROJECT_LABELS, PROJECTS, RECOMMENDATION_TYPES, SKILL_LABELS, SKILLS, project_index, skill_index


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run Skill State Model inference.")
    parser.add_argument("--model-dir", default="runs/skillrec_transformer")
    parser.add_argument("--events-json", default=None)
    parser.add_argument("--device", default="auto")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    service = SkillRecInference(args.model_dir, args.device)
    events = json.loads(args.events_json) if args.events_json else default_events()
    print(json.dumps(service.predict(events), ensure_ascii=False, indent=2))


class SkillRecInference:
    def __init__(self, model_dir: str | Path, device: str = "auto") -> None:
        self.model_dir = Path(model_dir)
        selected_device = "cuda" if device == "auto" and torch.cuda.is_available() else device
        if selected_device == "auto":
            selected_device = "cpu"
        self.device = torch.device(selected_device)
        self.model, self.config, self.payload = load_checkpoint(str(self.model_dir / "best.pt"), map_location=self.device)
        self.model.to(self.device)

    @torch.no_grad()
    def predict(self, events: list[dict[str, Any]]) -> dict[str, Any]:
        batch = encode_events(events, self.config.max_events, self.device)
        outputs = self.model(batch["skill_ids"], batch["event_ids"], batch["project_ids"], batch["outcomes"], batch["attention_mask"])
        skill_values = outputs["skill_vector"][0].detach().cpu().tolist()
        confidence_values = outputs["confidence"][0].detach().cpu().tolist()
        next_skill_id = int(outputs["next_skill_logits"][0].argmax().detach().cpu())
        project_scores = outputs["project_scores"][0].softmax(dim=-1).detach().cpu()
        completion = outputs["completion_probability"][0].detach().cpu()
        learning_gain = outputs["learning_gain"][0].detach().cpu()
        ranked = project_scores.argsort(descending=True).tolist()
        recommendations = []
        for index, project_id in enumerate(ranked[:3]):
            project_key = PROJECTS[project_id]
            recommendations.append({
                "type": RECOMMENDATION_TYPES[index],
                "projectId": project_key,
                "title": PROJECT_LABELS[project_key],
                "score": round(float(project_scores[project_id]), 4),
                "completionProbability": round(float(completion[project_id]), 4),
                "learningGain": round(float(learning_gain[project_id]), 4),
                "reason": recommendation_reason(index, project_key),
            })
        weaknesses = sorted(
            [{"skill": skill, "label": SKILL_LABELS[skill], "value": round(skill_values[i], 4), "confidence": round(confidence_values[i], 4)} for i, skill in enumerate(SKILLS)],
            key=lambda item: item["value"],
        )[:5]
        return {
            "available": True,
            "device": str(self.device),
            "skillVector": {skill: round(skill_values[i], 4) for i, skill in enumerate(SKILLS)},
            "confidence": {skill: round(confidence_values[i], 4) for i, skill in enumerate(SKILLS)},
            "weaknesses": weaknesses,
            "nextConcept": {"skill": SKILLS[next_skill_id], "label": SKILL_LABELS[SKILLS[next_skill_id]]},
            "avoidConcepts": ["AC100V", "高電流モーター", "LiPo自作充放電", "複雑な割り込み"],
            "recommendations": recommendations,
        }


def encode_events(events: list[dict[str, Any]], max_events: int, device: torch.device) -> dict[str, torch.Tensor]:
    trimmed = events[-max_events:]
    skill_ids = []
    event_ids = []
    project_ids = []
    outcomes = []
    for event in trimmed:
        skill = event.get("skill", "debugging")
        event_type = event.get("event", "question")
        project = event.get("project", "light_charm")
        skill_ids.append(skill_index(skill) + 1 if skill in SKILLS else skill_index("debugging") + 1)
        event_ids.append(EVENT_TYPES.index(event_type) + 1 if event_type in EVENT_TYPES else EVENT_TYPES.index("question") + 1)
        project_ids.append(project_index(project) + 1 if project in PROJECTS else project_index("light_charm") + 1)
        outcomes.append(int(event.get("outcome", 0)) + 1)
    mask = [1] * len(skill_ids)
    while len(skill_ids) < max_events:
        skill_ids.append(0)
        event_ids.append(0)
        project_ids.append(0)
        outcomes.append(0)
        mask.append(0)
    return {
        "skill_ids": torch.tensor([skill_ids], dtype=torch.long, device=device),
        "event_ids": torch.tensor([event_ids], dtype=torch.long, device=device),
        "project_ids": torch.tensor([project_ids], dtype=torch.long, device=device),
        "outcomes": torch.tensor([outcomes], dtype=torch.long, device=device),
        "attention_mask": torch.tensor([mask], dtype=torch.bool, device=device),
    }


def recommendation_reason(index: int, project: str) -> str:
    if index == 0:
        return "現在のスキルと所持部品から完成可能性が高い候補です。"
    if index == 1:
        return "既に扱った概念を少し広げて、次の理解に進みやすい候補です。"
    return "作品性と学習効果を優先して、憧れに近づく候補です。"


def default_events() -> list[dict[str, Any]]:
    return [
        {"skill": "led_polarity", "event": "project_started", "project": "light_charm", "outcome": 1},
        {"skill": "resistor_usage", "event": "wiring_error", "project": "light_charm", "outcome": 0},
        {"skill": "gnd_common", "event": "hint_viewed", "project": "light_charm", "outcome": 0},
        {"skill": "debugging", "event": "debug_success", "project": "light_charm", "outcome": 1},
        {"skill": "threshold_processing", "event": "completed", "project": "light_charm", "outcome": 1},
    ]


if __name__ == "__main__":
    main()
