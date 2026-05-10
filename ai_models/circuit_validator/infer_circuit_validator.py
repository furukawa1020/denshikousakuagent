from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

import torch

from ai_models.makergraph.tokenizer import MakerTokenizer

from .model import load_checkpoint
from .taxonomy import BOARD_CLASSES, CIRCUIT_ISSUES, REPAIR_ACTIONS, RISK_CLASSES


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run neural circuit validation.")
    parser.add_argument("--model-dir", default="runs/circuit_validator")
    parser.add_argument("--text", required=True)
    parser.add_argument("--device", default="auto")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    service = CircuitValidatorInference(args.model_dir, args.device)
    print(json.dumps(service.predict({"text": args.text}), ensure_ascii=False, indent=2))


class CircuitValidatorInference:
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
        text = build_input_text(payload)
        input_ids, mask = self.tokenizer.encode(text, self.config.max_length, "<graph>")
        ids = torch.tensor([input_ids], dtype=torch.long, device=self.device)
        attention = torch.tensor([mask], dtype=torch.bool, device=self.device)
        outputs = self.model(ids, attention)
        issue_probs = outputs["issue_logits"].sigmoid()[0].detach().cpu()
        risk_probs = outputs["risk_logits"][0].softmax(dim=-1).detach().cpu()
        repair_probs = outputs["repair_logits"][0].softmax(dim=-1).detach().cpu()
        board_probs = outputs["board_logits"][0].softmax(dim=-1).detach().cpu()
        issue_items = [
            {"issue": CIRCUIT_ISSUES[index], "score": round(float(issue_probs[index]), 4)}
            for index in issue_probs.argsort(descending=True).tolist()
            if float(issue_probs[index]) >= 0.35
        ][:5]
        has_actionable_issue = any(item["issue"] != "ok" for item in issue_items)
        if not has_actionable_issue:
            ok_score = float(issue_probs[CIRCUIT_ISSUES.index("ok")])
            issues = [{"issue": "ok", "score": round(max(ok_score, 1.0 - float(issue_probs.max())), 4)}]
            risk_index = RISK_CLASSES.index("low")
            repair_index = REPAIR_ACTIONS.index("none")
        else:
            issues = [item for item in issue_items if item["issue"] != "ok"]
            risk_index = int(risk_probs.argmax())
            repair_index = int(repair_probs.argmax())
        board_index = int(board_probs.argmax())
        return {
            "available": True,
            "model": "CircuitValidatorTransformer",
            "device": str(self.device),
            "inputPreview": text[:500],
            "issues": issues,
            "risk": RISK_CLASSES[risk_index],
            "riskDistribution": {label: round(float(risk_probs[index]), 4) for index, label in enumerate(RISK_CLASSES)},
            "repairAction": REPAIR_ACTIONS[repair_index],
            "repairDistribution": {label: round(float(repair_probs[index]), 4) for index, label in enumerate(REPAIR_ACTIONS)},
            "board": BOARD_CLASSES[board_index],
            "boardDistribution": {label: round(float(board_probs[index]), 4) for index, label in enumerate(BOARD_CLASSES)},
            "metrics": self.payload.get("metrics", {}),
        }


def build_input_text(payload: dict[str, Any]) -> str:
    keys = ["text", "board", "connectionTable", "pinMapping", "powerMapping", "circuitGraph", "components", "symptom"]
    parts = []
    for key in keys:
        value = payload.get(key)
        if value:
            parts.append(f"{key}: {json.dumps(value, ensure_ascii=False) if isinstance(value, (dict, list)) else value}")
    return " | ".join(parts) or "board: esp32 connection: LED anode GPIO5 resistor 220ohm sensor VCC 3.3V GND GND SIG GPIO21"


if __name__ == "__main__":
    main()
