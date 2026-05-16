from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any


ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from ai_models.tutorial_response.answer_classifier import ANSWER_KIND_LABELS
from ai_models.tutorial_response.data import build_source, write_jsonl


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Export tutorial interaction logs into answer-classifier fine-tuning records.")
    parser.add_argument("--logs", default="runtime/tutorial_response_events.jsonl")
    parser.add_argument("--feedback-logs", default="runtime/tutorial_feedback_events.jsonl")
    parser.add_argument("--output", default="runtime/tutorial_answer_classifier_real_records.jsonl")
    parser.add_argument("--min-answer-length", type=int, default=1)
    parser.add_argument("--min-confidence", type=float, default=0.0)
    parser.add_argument("--max-records", type=int, default=0)
    parser.add_argument("--positive-only", action="store_true")
    parser.add_argument("--drop-negative", action="store_true")
    parser.add_argument("--include-test-sessions", action="store_true")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    events = load_jsonl(Path(args.logs))
    feedback = feedback_index(load_jsonl(Path(args.feedback_logs)))
    records: list[dict[str, Any]] = []
    seen: set[tuple[str, str]] = set()
    skipped = 0
    feedback_counts = {"up": 0, "down": 0, "fix": 0, "none": 0}

    for event in events:
        if event.get("schema") != "tutorial_response_event_v1":
            skipped += 1
            continue
        if not args.include_test_sessions and is_test_session(event):
            skipped += 1
            continue
        feedback_event = feedback.get(event_key(event), {})
        rating = feedback_rating(feedback_event)
        feedback_counts[rating or "none"] += 1
        if args.positive_only and rating not in {"up", "fix"}:
            skipped += 1
            continue
        if args.drop_negative and rating == "down":
            skipped += 1
            continue
        record = event_to_classifier_record(event, feedback_event, args.min_answer_length, args.min_confidence)
        if record is None:
            skipped += 1
            continue
        key = (record["source"], record["kind"])
        if key in seen:
            skipped += 1
            continue
        seen.add(key)
        records.append(record)
        if args.max_records and len(records) >= args.max_records:
            break

    write_jsonl(args.output, records)
    label_counts: dict[str, int] = {label: 0 for label in ANSWER_KIND_LABELS}
    for record in records:
        label_counts[record["kind"]] = label_counts.get(record["kind"], 0) + 1
    print(json.dumps({
        "event": "tutorial_answer_classifier_logs_exported",
        "logs": args.logs,
        "feedbackLogs": args.feedback_logs,
        "output": args.output,
        "events": len(events),
        "records": len(records),
        "skipped": skipped,
        "labels": label_counts,
        "feedback": feedback_counts,
    }, ensure_ascii=False, indent=2))


def load_jsonl(path: Path) -> list[dict[str, Any]]:
    if not path.exists():
        return []
    with path.open(encoding="utf-8") as handle:
        return [json.loads(line) for line in handle if line.strip()]


def event_to_classifier_record(
    event: dict[str, Any],
    feedback_event: dict[str, Any] | None,
    min_answer_length: int,
    min_confidence: float,
) -> dict[str, Any] | None:
    answer = str(event.get("answer") or "")
    if len(answer.strip()) < min_answer_length:
        return None
    label = label_from_feedback(feedback_event)
    source = "feedback" if label else "event"
    if not label:
        label = str(event.get("interpreted") or "")
    if label not in ANSWER_KIND_LABELS:
        return None
    classifier = event.get("answerClassifier") if isinstance(event.get("answerClassifier"), dict) else {}
    confidence = confidence_value(classifier)
    if min_confidence and confidence is not None and confidence < min_confidence:
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
            "skill": "",
            "previous": "",
        }
    return {
        "source": build_source(source_payload),
        "kind": label,
        "target": label,
        "feedback": feedback_rating(feedback_event) or "none",
        "labelSource": source,
        "classifierConfidence": confidence,
    }


def label_from_feedback(event: dict[str, Any] | None) -> str:
    if not event:
        return ""
    expected = str(event.get("expectedKind") or event.get("kind") or "")
    if expected in ANSWER_KIND_LABELS:
        return expected
    reply = event.get("reply") if isinstance(event.get("reply"), dict) else {}
    interpreted = str(reply.get("interpreted") or "")
    if interpreted in ANSWER_KIND_LABELS and feedback_rating(event) == "up":
        return interpreted
    return ""


def confidence_value(classifier: dict[str, Any]) -> float | None:
    try:
        return float(classifier.get("confidence"))
    except (TypeError, ValueError):
        return None


def feedback_index(events: list[dict[str, Any]]) -> dict[tuple[str, str, str], dict[str, Any]]:
    result: dict[tuple[str, str, str], dict[str, Any]] = {}
    for event in events:
        if event.get("schema") != "tutorial_feedback_event_v1":
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


if __name__ == "__main__":
    main()
