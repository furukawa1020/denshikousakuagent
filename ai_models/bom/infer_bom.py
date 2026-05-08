from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

import torch

from ai_models.makergraph.tokenizer import MakerTokenizer

from ai_models.makergraph.sample_data import PROJECT_CATALOG, project_to_document

from .component_catalog import COMPONENT_BY_ID, COMPONENT_IDS, PROJECT_COMPONENTS, RISK_CLASSES
from .model import load_checkpoint


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run neural BOM estimator inference.")
    parser.add_argument("--model-dir", default="runs/bom_estimator")
    parser.add_argument("--text", required=True)
    parser.add_argument("--device", default="auto")
    parser.add_argument("--threshold", type=float, default=0.42)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    service = BOMInference(args.model_dir, args.device)
    print(json.dumps(service.predict(args.text, threshold=args.threshold), ensure_ascii=False, indent=2))


class BOMInference:
    def __init__(self, model_dir: str | Path, device: str = "auto") -> None:
        self.model_dir = Path(model_dir)
        selected_device = "cuda" if device == "auto" and torch.cuda.is_available() else device
        if selected_device == "auto":
            selected_device = "cpu"
        self.device = torch.device(selected_device)
        self.tokenizer = MakerTokenizer.load(self.model_dir / "tokenizer.json")
        self.model, self.config, self.payload = load_checkpoint(str(self.model_dir / "best.pt"), map_location=self.device)
        self.model.to(self.device)

    @torch.no_grad()
    def predict(self, text: str, threshold: float = 0.42) -> dict[str, Any]:
        text = canonicalize_bom_input(text)
        input_ids, mask = self.tokenizer.encode(text, self.config.max_length, "<project>")
        ids = torch.tensor([input_ids], dtype=torch.long, device=self.device)
        attention = torch.tensor([mask], dtype=torch.bool, device=self.device)
        outputs = self.model(ids, attention)
        component_scores = outputs["component_logits"].sigmoid()[0].detach().cpu()
        priority_scores = outputs["priority_logits"].sigmoid()[0].detach().cpu()
        selected = []
        for index, score in enumerate(component_scores.tolist()):
            if score >= threshold:
                component_id = COMPONENT_IDS[index]
                component = COMPONENT_BY_ID[component_id]
                selected.append({
                    "componentId": component_id,
                    "name": component.name,
                    "category": component.category,
                    "score": round(score, 4),
                    "priorityScore": round(float(priority_scores[index]), 4),
                    "priceRange": f"{component.price_min:,}〜{component.price_max:,}円",
                })
        selected.sort(key=lambda item: item["priorityScore"], reverse=True)
        quantiles = outputs["cost_quantiles"][0].detach().cpu() * self.config.cost_scale
        risk_id = int(outputs["risk_logits"][0].argmax().detach().cpu())
        return {
            "available": True,
            "device": str(self.device),
            "components": selected,
            "estimatedCost": {
                "min": int(quantiles[0].item()),
                "median": int(quantiles[1].item()),
                "max": int(quantiles[2].item()),
                "range": f"{int(quantiles[0].item()):,}〜{int(quantiles[2].item()):,}円",
            },
            "risk": RISK_CLASSES[risk_id],
            "normalizedInputPreview": text[:500],
        }


def canonicalize_bom_input(text: str) -> str:
    project = detect_project(text)
    if not project:
        return text
    project_id = project["id"]
    component_ids = PROJECT_COMPONENTS.get(project_id, [])
    owned = detect_owned_components(text)
    component_hint = "components: " + ", ".join(COMPONENT_BY_ID[component_id].name for component_id in component_ids)
    owned_hint = "owned: " + (", ".join(COMPONENT_BY_ID[component_id].name for component_id in owned) if owned else "none")
    return "\n".join([
        project_to_document(project),
        "bom_tier: minimal",
        component_hint,
        owned_hint,
        text,
        "must_include: USB cable, jumper wires, resistor for LED, breadboard unless owned",
        "safety: AC100V excluded, LiPo charging excluded, motor direct drive excluded",
    ])


def detect_project(text: str) -> dict[str, Any] | None:
    normalized = text.lower()
    hints = {
        "desk_pet": ["机上ペット", "かわいい", "近づく", "鳴く"],
        "light_charm": ["光るお守り", "暗くなる", "lチカ", "ライト"],
        "temp_face": ["温度", "表情", "環境モニター", "m5"],
        "plant_ping": ["水やり", "植物", "土壌", "枯らす"],
        "posture_guard": ["姿勢", "集中", "距離", "机との距離"],
    }
    scores = {}
    for project in PROJECT_CATALOG:
        score = 0
        if project["title"].lower() in normalized or project["id"] in normalized:
            score += 5
        for hint in hints.get(project["id"], []):
            if hint.lower() in normalized:
                score += 1
        scores[project["id"]] = score
    best_id = max(scores, key=scores.get)
    if scores[best_id] == 0:
        return None
    return next(project for project in PROJECT_CATALOG if project["id"] == best_id)


def detect_owned_components(text: str) -> list[str]:
    owned = []
    for component_id, component in COMPONENT_BY_ID.items():
        for alias in component.aliases + (component.name,):
            if alias.lower() in text.lower():
                owned.append(component_id)
                break
    return owned


if __name__ == "__main__":
    main()
