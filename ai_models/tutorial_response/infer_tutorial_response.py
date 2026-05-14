from __future__ import annotations

import argparse
import json
import re
from pathlib import Path
from typing import Any

import torch

from ai_models.makergraph.tokenizer import MakerTokenizer

from .data import build_source
from .model import load_checkpoint
from .taxonomy import PROJECTS


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run neural free-response tutorial generator.")
    parser.add_argument("--model-dir", default="runs/tutorial_response")
    parser.add_argument("--question", default="センサー、LED、ボードのGNDは同じGND列につながっていますか？")
    parser.add_argument("--answer", default="同じ列か分からないです")
    parser.add_argument("--stage", default="minimal_circuit")
    parser.add_argument("--project-id", default="desk_pet")
    parser.add_argument("--inventory", default="ESP32, LED, 220Ω抵抗, ブレッドボード")
    parser.add_argument("--budget", default="5000")
    parser.add_argument("--board", default="esp32")
    parser.add_argument("--device", default="auto")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    service = TutorialResponseInference(args.model_dir, args.device)
    print(json.dumps(service.predict({
        "question": args.question,
        "answer": args.answer,
        "currentStage": args.stage,
        "projectId": args.project_id,
        "inventory": args.inventory,
        "budget": args.budget,
        "board": args.board,
    }), ensure_ascii=False, indent=2))


class TutorialResponseInference:
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
        source = build_source(normalize_payload(payload))
        source_ids, source_mask = self.tokenizer.encode(source, self.config.max_source_length, "<query>")
        ids = torch.tensor([source_ids], dtype=torch.long, device=self.device)
        mask = torch.tensor([source_mask], dtype=torch.bool, device=self.device)
        output_ids = self.model.generate(ids, mask, max_new_tokens=self.config.max_target_length - 1)[0].detach().cpu().tolist()
        generated = self.tokenizer.decode(output_ids)
        parsed = parse_generated(generated)
        return {
            "available": True,
            "model": "TutorialResponseTransformer",
            "device": str(self.device),
            "inputPreview": source[:500],
            "generated": generated,
            "kind": parsed.get("style", "info"),
            "title": parsed.get("title", "回答を受け取りました"),
            "body": parsed.get("body", "次の作業に反映します。"),
            "nextInstruction": parsed.get("next", "次の短い確認へ進みます。"),
            "nextStage": parsed.get("stage", str(payload.get("currentStage") or "parts_check")),
            "metrics": self.payload.get("metrics", {}),
        }


def normalize_payload(payload: dict[str, Any]) -> dict[str, Any]:
    project_id = str(payload.get("projectId") or payload.get("project") or "desk_pet")
    return {
        "projectId": project_id if project_id in PROJECTS else "desk_pet",
        "projectTitle": PROJECTS.get(project_id, PROJECTS["desk_pet"]),
        "currentStage": payload.get("currentStage") or payload.get("stage") or "orient",
        "question": payload.get("question") or payload.get("nextQuestion") or "",
        "answer": payload.get("answer") or payload.get("text") or "",
        "inventory": payload.get("inventory") or "",
        "budget": payload.get("budget") or "",
        "symptom": payload.get("symptom") or "none",
    }


def parse_generated(text: str) -> dict[str, str]:
    result: dict[str, str] = {}
    current_key = ""
    normalized = re.sub(r"(style:|title:|body:|next:|stage:)", r"\n\1", text)
    for raw_line in normalized.splitlines():
        line = raw_line.strip()
        if not line:
            continue
        if ":" in line:
            key, value = line.split(":", 1)
            key = key.strip()
            if key in {"style", "title", "body", "next", "stage"}:
                result[key] = value.strip()
                current_key = key
                continue
        if current_key:
            result[current_key] = f"{result.get(current_key, '')}{line}".strip()
    if result.get("stage") not in {
        "orient",
        "parts_check",
        "minimal_circuit",
        "firmware_upload",
        "observe_serial",
        "debug_triage",
        "standard_build",
        "enclosure",
        "extension",
        "completion_log",
    }:
        result["stage"] = "minimal_circuit"
    if result.get("style") not in {"good", "warn", "info"}:
        result["style"] = "info"
    return result


if __name__ == "__main__":
    main()
