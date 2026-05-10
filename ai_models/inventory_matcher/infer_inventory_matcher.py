from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

import torch

from ai_models.bom.component_catalog import COMPONENT_BY_ID, COMPONENT_IDS
from ai_models.makergraph.tokenizer import MakerTokenizer

from .data import build_input_text
from .model import load_checkpoint
from .taxonomy import MATCH_TIER_LABELS, MATCH_TIERS, PROJECT_IDS, PROJECT_LABELS


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run neural inventory-to-project matcher.")
    parser.add_argument("--model-dir", default="runs/inventory_matcher")
    parser.add_argument("--text", default="")
    parser.add_argument("--inventory", default="")
    parser.add_argument("--budget", type=int, default=5000)
    parser.add_argument("--interest", default="かわいい")
    parser.add_argument("--device", default="auto")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    service = InventoryMatcherInference(args.model_dir, args.device)
    print(json.dumps(service.predict({
        "text": args.text,
        "inventory": args.inventory,
        "budget": args.budget,
        "interest": args.interest,
    }), ensure_ascii=False, indent=2))


class InventoryMatcherInference:
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
    def predict(self, payload: dict[str, Any]) -> dict[str, Any]:
        owned = detect_owned_components(payload)
        budget = int(payload.get("budget") or payload.get("budgetLimit") or 5000)
        interest = detect_interest(payload)
        text = build_input_text(owned, budget, interest)
        extra_text = str(payload.get("text") or "").strip()
        if extra_text:
            text = f"{text}\nuser_text: {extra_text}"
        input_ids, mask = self.tokenizer.encode(text, self.config.max_length, "<component>")
        ids = torch.tensor([input_ids], dtype=torch.long, device=self.device)
        attention = torch.tensor([mask], dtype=torch.bool, device=self.device)
        outputs = self.model(ids, attention)
        project_probs = outputs["project_logits"][0].softmax(dim=-1).detach().cpu()
        completion = outputs["completion_logits"][0].sigmoid().detach().cpu()
        budget_fit = outputs["budget_fit_logits"][0].sigmoid().detach().cpu()
        missing_probs = outputs["missing_component_logits"][0].sigmoid().detach().cpu()
        tier_probs = outputs["tier_logits"][0].softmax(dim=-1).detach().cpu()
        ranked_projects = project_probs.argsort(descending=True).tolist()
        recommendations = []
        for index in ranked_projects[:3]:
            project_id = PROJECT_IDS[index]
            recommendations.append({
                "projectId": project_id,
                "title": PROJECT_LABELS[project_id],
                "score": round(float(project_probs[index]), 4),
                "completionProbability": round(float(completion[index]), 4),
                "budgetFit": round(float(budget_fit[index]), 4),
            })
        missing = [
            {
                "componentId": COMPONENT_IDS[index],
                "name": COMPONENT_BY_ID[COMPONENT_IDS[index]].name,
                "score": round(float(missing_probs[index]), 4),
                "priceRange": f"{COMPONENT_BY_ID[COMPONENT_IDS[index]].price_min:,}〜{COMPONENT_BY_ID[COMPONENT_IDS[index]].price_max:,}円",
            }
            for index in missing_probs.argsort(descending=True).tolist()
            if float(missing_probs[index]) >= 0.38
        ][:8]
        tier_index = int(tier_probs.argmax())
        return {
            "available": True,
            "model": "InventoryProjectMatcher",
            "device": str(self.device),
            "inputPreview": text[:500],
            "ownedComponents": [
                {"componentId": component_id, "name": COMPONENT_BY_ID[component_id].name}
                for component_id in owned
            ],
            "matchTier": {
                "id": MATCH_TIERS[tier_index],
                "label": MATCH_TIER_LABELS[MATCH_TIERS[tier_index]],
                "confidence": round(float(tier_probs[tier_index]), 4),
            },
            "recommendations": recommendations,
            "missingComponents": missing,
            "metrics": self.payload.get("metrics", {}),
        }


def detect_owned_components(payload: dict[str, Any]) -> list[str]:
    values = [str(payload.get(key) or "") for key in ["inventory", "text", "owned", "components"]]
    raw = " ".join(values).lower()
    owned = []
    explicit_aliases = {
        "esp32_devkit": ["esp32", "esp32 devkit"],
        "arduino_uno": ["arduino", "arduino uno"],
        "m5stickc": ["m5stickc", "m5stack", "m5stickc plus2"],
        "led_5mm": ["led", "5mm led"],
        "resistor_220": ["resistor", "220", "220ohm", "220 ohm", "220Ω"],
        "breadboard": ["breadboard"],
        "jumper_wires": ["jumper", "jumper wire", "jumper wires"],
        "usb_cable": ["usb", "usb cable"],
        "light_sensor": ["light sensor", "photoresistor", "cds"],
        "distance_sensor": ["distance sensor", "ultrasonic", "hc-sr04", "tof"],
        "buzzer": ["buzzer", "piezo"],
        "servo_sg90": ["servo", "sg90"],
        "dht_sensor": ["dht", "temperature sensor", "humidity sensor"],
        "oled_display": ["oled", "display"],
        "soil_sensor": ["soil", "soil sensor", "moisture sensor"],
        "case_material": ["case", "box", "enclosure"],
    }
    for component_id, component in COMPONENT_BY_ID.items():
        probes = [component_id, component.name, *component.aliases, *explicit_aliases.get(component_id, [])]
        if any(str(probe).lower() in raw for probe in probes):
            owned.append(component_id)
    return owned


def detect_interest(payload: dict[str, Any]) -> str:
    raw = " ".join(str(payload.get(key) or "") for key in ["interest", "mood", "problem", "text"]).lower()
    mapping = [
        ("植物", ["植物", "水やり", "plant", "soil"]),
        ("集中", ["集中", "姿勢", "posture", "focus"]),
        ("光る", ["光", "led", "light", "暗く"]),
        ("便利", ["便利", "困りごと", "useful"]),
        ("研究っぽい", ["研究", "ログ", "monitor"]),
        ("机上", ["机", "desk"]),
        ("かわいい", ["かわいい", "cute", "pet"]),
    ]
    for label, probes in mapping:
        if any(probe in raw for probe in probes):
            return label
    return "かわいい"


if __name__ == "__main__":
    main()
