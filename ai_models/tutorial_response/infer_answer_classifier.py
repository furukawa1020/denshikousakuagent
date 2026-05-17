from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

import torch
from torch.nn import functional as F

from ai_models.makergraph.tokenizer import MakerTokenizer

from .answer_classifier import ANSWER_KIND_LABELS, load_checkpoint
from .data import build_source
from .taxonomy import PROJECTS


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run neural tutorial answer classifier.")
    parser.add_argument("--model-dir", default="runs/tutorial_answer_classifier")
    parser.add_argument("--question", default="LED long leg goes through the resistor to GPIO?")
    parser.add_argument("--answer", default="yes, it goes through the resistor")
    parser.add_argument("--stage", default="minimal_circuit")
    parser.add_argument("--project-id", default="desk_pet")
    parser.add_argument("--inventory", default="ESP32, LED, 220 ohm resistor")
    parser.add_argument("--budget", default="5000")
    parser.add_argument("--device", default="auto")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    service = TutorialAnswerClassifierInference(args.model_dir, args.device)
    print(json.dumps(service.predict({
        "question": args.question,
        "answer": args.answer,
        "currentStage": args.stage,
        "projectId": args.project_id,
        "inventory": args.inventory,
        "budget": args.budget,
    }), ensure_ascii=False, indent=2))


class TutorialAnswerClassifierInference:
    def __init__(self, model_dir: str | Path, device: str = "auto") -> None:
        self.model_dir = Path(model_dir)
        selected_device = "cuda" if device == "auto" and torch.cuda.is_available() else device
        if selected_device == "auto":
            selected_device = "cpu"
        self.device = torch.device(selected_device)
        self.tokenizer = MakerTokenizer.load(self.model_dir / "tokenizer.json")
        self.model, self.config, self.payload = load_checkpoint(self.model_dir / "best.pt", map_location=self.device)
        self.model.to(self.device)
        self.labels = list(self.payload.get("labels") or ANSWER_KIND_LABELS)

    @torch.no_grad()
    def predict(self, payload: dict[str, Any]) -> dict[str, Any]:
        normalized = normalize_payload(payload)
        source = build_source(normalized)
        source_ids, source_mask = self.tokenizer.encode(source, self.config.max_source_length, "<query>")
        ids = torch.tensor([source_ids], dtype=torch.long, device=self.device)
        mask = torch.tensor([source_mask], dtype=torch.bool, device=self.device)
        logits = self.model(ids, mask)
        probs = F.softmax(logits, dim=-1)[0].detach().cpu()
        ranked = sorted(
            [{"kind": self.labels[index], "probability": float(probs[index])} for index in range(len(self.labels))],
            key=lambda item: item["probability"],
            reverse=True,
        )
        return {
            "available": True,
            "model": "TutorialAnswerClassifierTransformer",
            "device": str(self.device),
            "kind": ranked[0]["kind"],
            "confidence": round(float(ranked[0]["probability"]), 4),
            "ranked": [{"kind": item["kind"], "probability": round(float(item["probability"]), 4)} for item in ranked],
            "inputPreview": source[:500],
            "metrics": self.payload.get("metrics", {}),
        }


def normalize_payload(payload: dict[str, Any]) -> dict[str, Any]:
    project_id = str(payload.get("projectId") or payload.get("project") or "desk_pet")
    return {
        "projectId": project_id if project_id in PROJECTS else "desk_pet",
        "projectTitle": PROJECTS.get(project_id, PROJECTS["desk_pet"]),
        "currentStage": payload.get("currentStage") or payload.get("stage") or "orient",
        "tutorialState": payload.get("tutorialState") or payload.get("tutorial_state") or payload.get("flow") or "",
        "question": payload.get("question") or payload.get("nextQuestion") or "",
        "answer": payload.get("answer") or payload.get("text") or "",
        "inventory": payload.get("inventory") or "",
        "budget": payload.get("budget") or "",
        "symptom": payload.get("symptom") or "none",
        "skill": payload.get("skill") or "",
        "previous": payload.get("previous") or "",
        "lastQuestion": payload.get("lastQuestion") or payload.get("last_question") or "",
        "lastAnswer": payload.get("lastAnswer") or payload.get("last_answer") or "",
        "lastInterpreted": payload.get("lastInterpreted") or payload.get("last_interpreted") or "",
        "tutorialTopic": payload.get("tutorialTopic") or payload.get("tutorial_topic") or "",
    }


if __name__ == "__main__":
    main()
