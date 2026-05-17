from __future__ import annotations

import argparse
import json
import os
import re
from pathlib import Path
from typing import Any

import torch
from torch.nn import functional as F

from ai_models.makergraph.tokenizer import MakerTokenizer

from .data import build_source
from .model import load_checkpoint
from .taxonomy import PROJECTS


VALID_STAGES = {
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
}

STYLE_BY_INTERPRETED_KIND = {
    "confirmed": "good",
    "inventory_report": "good",
    "board_report": "good",
    "negative": "warn",
    "unknown": "warn",
    "problem_report": "warn",
    "photo_request": "warn",
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run neural free-response tutorial generator.")
    parser.add_argument("--model-dir", default="runs/tutorial_response")
    parser.add_argument("--question", default="LED long leg goes through the resistor to GPIO?")
    parser.add_argument("--answer", default="yes, it goes through the resistor")
    parser.add_argument("--stage", default="minimal_circuit")
    parser.add_argument("--project-id", default="desk_pet")
    parser.add_argument("--inventory", default="ESP32, LED, 220 ohm resistor, breadboard")
    parser.add_argument("--budget", default="5000")
    parser.add_argument("--board", default="esp32")
    parser.add_argument("--interpreted-kind", default="")
    parser.add_argument("--fallback-stage", default="")
    parser.add_argument("--candidate-count", type=int, default=3)
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
        "interpretedKind": args.interpreted_kind,
        "fallbackStage": args.fallback_stage,
        "candidateCount": args.candidate_count,
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
        normalized_payload = normalize_payload(payload)
        source = build_source(normalized_payload)
        source_ids, source_mask = self.tokenizer.encode(source, self.config.max_source_length, "<query>")
        ids = torch.tensor([source_ids], dtype=torch.long, device=self.device)
        mask = torch.tensor([source_mask], dtype=torch.bool, device=self.device)

        candidates = self.generate_candidates(ids, mask, normalized_payload, payload)
        best = max(candidates, key=lambda item: float(item["score"]))
        parsed = polish_parsed(dict(best["parsed"]))

        return {
            "available": True,
            "model": "TutorialResponseTransformer",
            "device": str(self.device),
            "inputPreview": source[:500],
            "generated": str(best["generated"]),
            "kind": parsed.get("style", "info"),
            "title": parsed.get("title", "回答を受け取りました"),
            "body": parsed.get("body", "いまの回答をもとに、次の手順へつなげます。"),
            "nextInstruction": parsed.get("next", "次の確認へ進みます。"),
            "nextStage": parsed.get("stage", str(normalized_payload.get("fallbackStage") or normalized_payload.get("currentStage") or "parts_check")),
            "alignmentScore": round(float(best["score"]), 4),
            "candidateCount": len(candidates),
            "candidates": [
                {
                    "rank": index + 1,
                    "score": round(float(item["score"]), 4),
                    "style": item["parsed"].get("style", "info"),
                    "stage": item["parsed"].get("stage", ""),
                    "title": item["parsed"].get("title", ""),
                    "meanLogProb": round(float(item["mean_log_prob"]), 4),
                }
                for index, item in enumerate(sorted(candidates, key=lambda item: float(item["score"]), reverse=True)[:4])
            ],
            "metrics": self.payload.get("metrics", {}),
        }

    @torch.no_grad()
    def generate_candidates(
        self,
        source_ids: torch.Tensor,
        source_mask: torch.Tensor,
        normalized_payload: dict[str, Any],
        original_payload: dict[str, Any],
    ) -> list[dict[str, Any]]:
        requested = original_payload.get("candidateCount") or os.environ.get("TUTORIAL_RESPONSE_CANDIDATES") or 3
        try:
            candidate_count = max(1, min(6, int(requested)))
        except (TypeError, ValueError):
            candidate_count = 3

        temperatures = [0.0, 0.55, 0.8, 1.0, 0.35, 1.2][:candidate_count]
        seen: set[str] = set()
        candidates: list[dict[str, Any]] = []
        base_seed = int(original_payload.get("seed") or 1729)
        prefix_ids = self.decoder_prefix_ids(normalized_payload)
        for index, temperature in enumerate(temperatures):
            generated_ids, mean_log_prob = self.generate_one(
                source_ids,
                source_mask,
                prefix_ids=prefix_ids,
                temperature=temperature,
                seed=base_seed + index,
            )
            generated = self.tokenizer.decode(generated_ids)
            if generated in seen:
                continue
            seen.add(generated)
            parsed = parse_generated(generated)
            sequence_log_prob = self.sequence_mean_log_prob(source_ids, source_mask, generated_ids)
            candidates.append({
                "generated": generated,
                "parsed": parsed,
                "score": sequence_log_prob,
                "mean_log_prob": sequence_log_prob,
                "sample_log_prob": mean_log_prob,
            })

        if not candidates:
            generated_ids, mean_log_prob = self.generate_one(source_ids, source_mask, prefix_ids=prefix_ids, temperature=0.0, seed=base_seed)
            generated = self.tokenizer.decode(generated_ids)
            parsed = parse_generated(generated)
            sequence_log_prob = self.sequence_mean_log_prob(source_ids, source_mask, generated_ids)
            candidates.append({
                "generated": generated,
                "parsed": parsed,
                "score": sequence_log_prob,
                "mean_log_prob": sequence_log_prob,
                "sample_log_prob": mean_log_prob,
            })
        return candidates

    @torch.no_grad()
    def generate_one(
        self,
        source_ids: torch.Tensor,
        source_mask: torch.Tensor,
        prefix_ids: list[int],
        temperature: float,
        seed: int,
    ) -> tuple[list[int], float]:
        generated = torch.tensor([prefix_ids], dtype=torch.long, device=source_ids.device)
        log_probs: list[float] = []
        generator = torch.Generator(device=self.device) if self.device.type == "cuda" else torch.Generator()
        generator.manual_seed(seed)

        for _ in range(self.config.max_target_length - 1):
            decoder_mask = ~generated.eq(self.config.pad_token_id)
            logits = self.model(source_ids, source_mask, generated, decoder_mask)[:, -1, :]
            next_token, next_log_prob = choose_next_token(logits, temperature=temperature, top_k=28, generator=generator)
            generated = torch.cat([generated, next_token], dim=1)
            log_probs.append(float(next_log_prob.detach().cpu().item()))
            if bool((next_token == self.config.eos_token_id).all()):
                break

        mean_log_prob = sum(log_probs) / max(1, len(log_probs))
        return generated[0].detach().cpu().tolist(), mean_log_prob

    def decoder_prefix_ids(self, payload: dict[str, Any]) -> list[int]:
        ids = [self.config.bos_token_id]
        log_id = self.tokenizer.vocab.get("<log>")
        if log_id is not None:
            ids.append(log_id)
        style = decoder_style(payload)
        if style:
            ids.extend(self.tokenizer.vocab.get(token, self.tokenizer.unk_id) for token in self.tokenizer.tokenize(f"style:{style}\n"))
        return ids

    @torch.no_grad()
    def sequence_mean_log_prob(
        self,
        source_ids: torch.Tensor,
        source_mask: torch.Tensor,
        generated_ids: list[int],
    ) -> float:
        if len(generated_ids) < 2:
            return -100.0
        target = torch.tensor([generated_ids], dtype=torch.long, device=self.device)
        decoder_input = target[:, :-1]
        labels = target[:, 1:]
        decoder_mask = ~decoder_input.eq(self.config.pad_token_id)
        logits = self.model(source_ids, source_mask, decoder_input, decoder_mask)
        token_log_probs = F.log_softmax(logits, dim=-1).gather(-1, labels.unsqueeze(-1)).squeeze(-1)
        label_mask = ~labels.eq(self.config.pad_token_id)
        return float((token_log_probs * label_mask).sum().detach().cpu().item() / max(1, int(label_mask.sum().detach().cpu().item())))


def choose_next_token(
    logits: torch.Tensor,
    temperature: float,
    top_k: int,
    generator: torch.Generator,
) -> tuple[torch.Tensor, torch.Tensor]:
    if temperature <= 0.0:
        log_probs = F.log_softmax(logits, dim=-1)
        next_token = log_probs.argmax(dim=-1, keepdim=True)
        next_log_prob = log_probs.gather(-1, next_token).mean()
        return next_token, next_log_prob

    scaled = logits / max(temperature, 0.05)
    k = min(top_k, scaled.shape[-1])
    values, indices = torch.topk(scaled, k=k, dim=-1)
    probs = F.softmax(values, dim=-1)
    sampled = torch.multinomial(probs, num_samples=1, generator=generator)
    next_token = indices.gather(-1, sampled)
    next_log_prob = torch.log(probs.gather(-1, sampled).clamp_min(1e-9)).mean()
    return next_token, next_log_prob


def normalize_payload(payload: dict[str, Any]) -> dict[str, Any]:
    project_id = str(payload.get("projectId") or payload.get("project") or "desk_pet")
    return {
        "projectId": project_id if project_id in PROJECTS else "desk_pet",
        "projectTitle": PROJECTS.get(project_id, PROJECTS["desk_pet"]),
        "currentStage": payload.get("currentStage") or payload.get("stage") or "orient",
        "tutorialState": payload.get("tutorialState") or payload.get("tutorial_state") or payload.get("flow") or "",
        "fallbackStage": payload.get("fallbackStage") or "",
        "interpretedKind": payload.get("interpretedKind") or payload.get("kind") or "",
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


def decoder_style(payload: dict[str, Any]) -> str:
    explicit = str(payload.get("responseStyle") or "")
    if explicit in {"good", "warn", "info"}:
        return explicit
    return STYLE_BY_INTERPRETED_KIND.get(str(payload.get("interpretedKind") or ""), "")


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
    if result.get("stage") not in VALID_STAGES:
        result["stage"] = "minimal_circuit"
    if result.get("style") not in {"good", "warn", "info"}:
        result["style"] = "info"
    return result


def polish_parsed(parsed: dict[str, str]) -> dict[str, str]:
    return {key: polish_text(value) for key, value in parsed.items()}


def polish_text(text: str) -> str:
    replacements = {
        "led": "LED",
        "gpio": "GPIO",
        "gnd": "GND",
        "usb": "USB",
        "serial": "Serial",
    }
    result = str(text or "")
    for source, target in replacements.items():
        result = re.sub(source, target, result, flags=re.IGNORECASE)
    result = result.replace("<num>Ω", "220Ω")
    result = result.replace("<num>オーム", "220オーム")
    result = result.replace("<num>", "")
    result = re.sub(r"Serial\s*monitor", "Serial Monitor", result, flags=re.IGNORECASE)
    return result


if __name__ == "__main__":
    main()
