from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any


ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from ai_models.tutorial_response.data import build_source, write_jsonl


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


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Export real tutorial response events into training JSONL records.")
    parser.add_argument("--logs", default="runtime/tutorial_response_events.jsonl")
    parser.add_argument("--output", default="runtime/tutorial_response_real_records.jsonl")
    parser.add_argument("--min-answer-length", type=int, default=1)
    parser.add_argument("--max-records", type=int, default=0)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    events = load_jsonl(Path(args.logs))
    records: list[dict[str, str]] = []
    seen: set[tuple[str, str]] = set()
    skipped = 0
    for event in events:
        record = event_to_record(event, args.min_answer_length)
        if record is None:
            skipped += 1
            continue
        key = (record["source"], record["target"])
        if key in seen:
            skipped += 1
            continue
        seen.add(key)
        records.append(record)
        if args.max_records and len(records) >= args.max_records:
            break
    write_jsonl(args.output, records)
    print(json.dumps({
        "event": "tutorial_logs_exported",
        "logs": args.logs,
        "output": args.output,
        "events": len(events),
        "records": len(records),
        "skipped": skipped,
    }, ensure_ascii=False, indent=2))


def load_jsonl(path: Path) -> list[dict[str, Any]]:
    if not path.exists():
        return []
    with path.open(encoding="utf-8") as handle:
        return [json.loads(line) for line in handle if line.strip()]


def event_to_record(event: dict[str, Any], min_answer_length: int) -> dict[str, str] | None:
    if event.get("schema") != "tutorial_response_event_v1":
        return None
    answer = str(event.get("answer") or "")
    if len(answer.strip()) < min_answer_length:
        return None
    source_payload = dict(event.get("sourcePayload") or {})
    if not source_payload:
        source_payload = {
            "projectId": event.get("projectId") or "desk_pet",
            "projectTitle": event.get("projectTitle") or "",
            "currentStage": event.get("currentStage") or "orient",
            "question": event.get("question") or "",
            "answer": answer,
            "inventory": "",
            "budget": "",
            "symptom": "",
        }
    target = str(event.get("target") or "").strip()
    if not target:
        target = target_from_response(event.get("response") or {}, str(source_payload.get("currentStage") or "orient"))
    if not target:
        return None
    return {
        "source": build_source(source_payload),
        "target": target,
    }


def target_from_response(response: dict[str, Any], fallback_stage: str) -> str:
    style = str(response.get("kind") or "info")
    if style not in {"good", "warn", "info"}:
        style = "info"
    stage = str(response.get("nextStage") or fallback_stage or "parts_check")
    if stage not in VALID_STAGES:
        stage = fallback_stage if fallback_stage in VALID_STAGES else "parts_check"
    return "\n".join([
        f"style: {style}",
        f"title: {response.get('title') or ''}",
        f"body: {response.get('body') or ''}",
        f"next: {response.get('nextInstruction') or ''}",
        f"stage: {stage}",
    ])


if __name__ == "__main__":
    main()
