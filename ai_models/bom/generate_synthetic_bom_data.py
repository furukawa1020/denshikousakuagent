from __future__ import annotations

import argparse
import json

from .data import generate_bom_records, write_bom_jsonl


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Generate synthetic BOM estimator training data.")
    parser.add_argument("--output", default="data/bom_training_gpu.jsonl")
    parser.add_argument("--records-per-project", type=int, default=600)
    parser.add_argument("--seed", type=int, default=45)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    records = generate_bom_records(args.records_per_project, seed=args.seed)
    write_bom_jsonl(args.output, records)
    print(json.dumps({"output": args.output, "records": len(records)}, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
