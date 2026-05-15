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
    parser.add_argument("--feedback-logs", default="runtime/tutorial_feedback_events.jsonl")
    parser.add_argument("--output", default="runtime/tutorial_response_real_records.jsonl")
    parser.add_argument("--min-answer-length", type=int, default=1)
    parser.add_argument("--max-records", type=int, default=0)
    parser.add_argument("--positive-only", action="store_true", help="Keep only events that received positive feedback.")
    parser.add_argument("--drop-negative", action="store_true", help="Drop events that received negative feedback.")
    parser.add_argument("--include-test-sessions", action="store_true", help="Include smoke/deploy test sessions in the export.")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    events = load_jsonl(Path(args.logs))
    feedback = feedback_index(load_jsonl(Path(args.feedback_logs)))
    records: list[dict[str, str]] = []
    seen: set[tuple[str, str]] = set()
    skipped = 0
    feedback_counts = {"up": 0, "down": 0, "fix": 0, "none": 0}
    for event in events:
        if not args.include_test_sessions and is_test_session(event):
            skipped += 1
            continue
        feedback_event = feedback.get(event_key(event), {})
        rating = feedback_rating(feedback_event)
        if rating in feedback_counts:
            feedback_counts[rating] += 1
        else:
            feedback_counts["none"] += 1
        if args.positive_only and rating not in {"up", "fix"}:
            skipped += 1
            continue
        if args.drop_negative and rating == "down":
            skipped += 1
            continue
        record = event_to_record(event, args.min_answer_length, feedback_event)
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
        "feedback": feedback_counts,
    }, ensure_ascii=False, indent=2))


def load_jsonl(path: Path) -> list[dict[str, Any]]:
    if not path.exists():
        return []
    with path.open(encoding="utf-8") as handle:
        return [json.loads(line) for line in handle if line.strip()]


def event_to_record(event: dict[str, Any], min_answer_length: int, feedback_event: dict[str, Any] | None = None) -> dict[str, str] | None:
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
    feedback_event = feedback_event or {}
    rating = feedback_rating(feedback_event)
    target = target_from_feedback(feedback_event, event) if rating == "fix" else ""
    if not target:
        target = str(event.get("target") or "").strip()
    if not target:
        target = target_from_response(event.get("response") or {}, str(source_payload.get("currentStage") or "orient"))
    if not target:
        return None
    return {
        "source": build_source(source_payload),
        "target": target,
        "feedback": rating or "none",
    }


def feedback_index(events: list[dict[str, Any]]) -> dict[tuple[str, str, str], dict[str, Any]]:
    result: dict[tuple[str, str, str], dict[str, Any]] = {}
    for event in events:
        if event.get("schema") != "tutorial_feedback_event_v1":
            continue
        rating = str(event.get("rating") or "")
        if rating not in {"up", "down", "fix"}:
            continue
        result[event_key(event)] = event
    return result


def feedback_rating(event: dict[str, Any] | None) -> str:
    rating = str((event or {}).get("rating") or "")
    return rating if rating in {"up", "down", "fix"} else ""


def event_key(event: dict[str, Any]) -> tuple[str, str, str]:
    return (
        str(event.get("sessionId") or "anonymous"),
        str(event.get("question") or ""),
        str(event.get("answer") or ""),
    )


def is_test_session(event: dict[str, Any]) -> bool:
    session_id = str(event.get("sessionId") or "").lower()
    client_version = str(event.get("clientVersion") or "").lower()
    return any(token in session_id or token in client_version for token in ["smoke", "deploy-test", "deploy-smoke", "local-test"])


def target_from_feedback(feedback_event: dict[str, Any], response_event: dict[str, Any]) -> str:
    correction = str(feedback_event.get("correction") or "").strip()
    if not correction:
        return ""
    if all(marker in correction for marker in ["style:", "title:", "body:", "next:", "stage:"]):
        return correction
    response = response_event.get("response") or {}
    stage = str(feedback_event.get("nextStage") or response_event.get("nextStage") or response.get("nextStage") or response_event.get("currentStage") or "parts_check")
    if stage not in VALID_STAGES:
        stage = "parts_check"
    style = str(response.get("kind") or "info")
    if style not in {"good", "warn", "info"}:
        style = "info"
    title = "返答を直しました"
    next_instruction = str(response.get("nextInstruction") or "")
    return "\n".join([
        f"style: {style}",
        f"title: {title}",
        f"body: {correction}",
        f"next: {next_instruction}",
        f"stage: {stage}",
    ])


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
