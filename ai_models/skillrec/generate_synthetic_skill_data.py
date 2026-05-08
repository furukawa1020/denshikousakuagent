from __future__ import annotations

import argparse
import json

from .data import generate_skill_records, write_skill_jsonl


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Generate synthetic production-log data for Skill State Model.")
    parser.add_argument("--output", default="data/skillrec_training_gpu.jsonl")
    parser.add_argument("--records", type=int, default=6000)
    parser.add_argument("--seed", type=int, default=47)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    records = generate_skill_records(args.records, seed=args.seed)
    write_skill_jsonl(args.output, records)
    print(json.dumps({"output": args.output, "records": len(records)}, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
