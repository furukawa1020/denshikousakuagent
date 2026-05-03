from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

import torch

from .model import load_checkpoint
from .sample_data import PROJECT_CATALOG, project_to_document
from .tokenizer import MakerTokenizer


class MakerIntentInference:
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
    def embed(self, text: str, prefix_token: str) -> list[float]:
        input_ids, mask = self.tokenizer.encode(text, self.config.max_length, prefix_token)
        ids = torch.tensor([input_ids], dtype=torch.long, device=self.device)
        attention = torch.tensor([mask], dtype=torch.bool, device=self.device)
        tower_type = 1 if prefix_token == "<query>" else 2
        output = self.model.encode(ids, attention, tower_type=tower_type)
        return output["embedding"][0].detach().cpu().tolist()

    @torch.no_grad()
    def profile(self, text: str) -> dict[str, float]:
        input_ids, mask = self.tokenizer.encode(text, self.config.max_length, "<query>")
        ids = torch.tensor([input_ids], dtype=torch.long, device=self.device)
        attention = torch.tensor([mask], dtype=torch.bool, device=self.device)
        output = self.model.encode(ids, attention, tower_type=1)
        values = output["profile"][0].detach().cpu().tolist()
        return {
            "difficulty_tolerance": values[0],
            "budget_sensitivity": values[1],
            "novelty_preference": values[2],
        }

    @torch.no_grad()
    def rank_projects(self, query: str, top_k: int = 3) -> list[dict[str, Any]]:
        query_ids, query_mask = self.tokenizer.encode(query, self.config.max_length, "<query>")
        q_ids = torch.tensor([query_ids], dtype=torch.long, device=self.device)
        q_mask = torch.tensor([query_mask], dtype=torch.bool, device=self.device)
        query_embedding = self.model.encode(q_ids, q_mask, tower_type=1)["embedding"]

        docs = [project_to_document(project) for project in PROJECT_CATALOG]
        project_ids = []
        project_masks = []
        for doc in docs:
            ids, mask = self.tokenizer.encode(doc, self.config.max_length, "<project>")
            project_ids.append(ids)
            project_masks.append(mask)
        p_ids = torch.tensor(project_ids, dtype=torch.long, device=self.device)
        p_mask = torch.tensor(project_masks, dtype=torch.bool, device=self.device)
        project_embedding = self.model.encode(p_ids, p_mask, tower_type=2)["embedding"]
        scores = (query_embedding @ project_embedding.T)[0]
        top_scores, top_indices = scores.topk(k=min(top_k, len(PROJECT_CATALOG)))
        results = []
        for score, index in zip(top_scores.detach().cpu().tolist(), top_indices.detach().cpu().tolist()):
            project = PROJECT_CATALOG[index]
            results.append({
                "project_id": project["id"],
                "title": project["title"],
                "score": score,
                "budget": project["budget"],
                "difficulty": project["difficulty"],
            })
        return results


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run Maker Intent Transformer inference.")
    parser.add_argument("--model-dir", default="runs/maker_intent_transformer")
    parser.add_argument("--text", required=True)
    parser.add_argument("--device", default="auto")
    parser.add_argument("--top-k", type=int, default=3)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    service = MakerIntentInference(args.model_dir, args.device)
    payload = {
        "text": args.text,
        "device": str(service.device),
        "profile": service.profile(args.text),
        "recommendations": service.rank_projects(args.text, args.top_k),
    }
    print(json.dumps(payload, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
