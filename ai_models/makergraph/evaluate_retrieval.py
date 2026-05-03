from __future__ import annotations

import argparse
import json
from pathlib import Path
from statistics import mean
from typing import Any

from .data import load_jsonl
from .infer import MakerIntentInference


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Evaluate Maker Intent Transformer retrieval quality.")
    parser.add_argument("--model-dir", default="runs/maker_intent_transformer")
    parser.add_argument("--data", default="data/intent_training_synthetic.jsonl")
    parser.add_argument("--device", default="auto")
    parser.add_argument("--top-k", type=int, default=3)
    parser.add_argument("--output", default="runs/maker_intent_transformer/retrieval_eval.json")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    records = load_jsonl(args.data)
    inference = MakerIntentInference(args.model_dir, args.device)
    rows = []
    reciprocal_ranks = []
    recall_at_1 = []
    recall_at_k = []

    for record in records:
        expected = record["positive_project"]["id"]
        predictions = inference.rank_projects(record["query"], top_k=max(args.top_k, 5))
        predicted_ids = [item["project_id"] for item in predictions]
        rank = predicted_ids.index(expected) + 1 if expected in predicted_ids else None
        reciprocal_ranks.append(0.0 if rank is None else 1.0 / rank)
        recall_at_1.append(1.0 if predicted_ids[:1] == [expected] else 0.0)
        recall_at_k.append(1.0 if expected in predicted_ids[: args.top_k] else 0.0)
        rows.append({
            "query": record["query"],
            "expected": expected,
            "rank": rank,
            "predictions": predictions[: args.top_k],
        })

    report: dict[str, Any] = {
        "modelDir": args.model_dir,
        "data": args.data,
        "records": len(records),
        "recall@1": round(mean(recall_at_1), 4) if records else 0.0,
        f"recall@{args.top_k}": round(mean(recall_at_k), 4) if records else 0.0,
        "mrr": round(mean(reciprocal_ranks), 4) if records else 0.0,
        "examples": rows[:20],
    }
    output = Path(args.output)
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps(report, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
