from __future__ import annotations

import argparse
import json
from pathlib import Path

from .infer import MakerIntentInference
from .sample_data import PROJECT_CATALOG, project_to_document


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Export project embeddings for fast retrieval serving.")
    parser.add_argument("--model-dir", default="runs/maker_intent_transformer")
    parser.add_argument("--device", default="auto")
    parser.add_argument("--output", default="runs/maker_intent_transformer/project_embeddings.json")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    inference = MakerIntentInference(args.model_dir, args.device)
    records = []
    for project in PROJECT_CATALOG:
        embedding = inference.embed(project_to_document(project), "<project>")
        records.append({
            "project_id": project["id"],
            "title": project["title"],
            "budget": project["budget"],
            "difficulty": project["difficulty"],
            "embedding": embedding,
        })
    output = Path(args.output)
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(records, ensure_ascii=False), encoding="utf-8")
    print(json.dumps({"output": str(output), "projects": len(records)}, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
