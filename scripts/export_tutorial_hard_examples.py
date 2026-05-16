from __future__ import annotations

import argparse
import json
import math
import sys
from pathlib import Path
from typing import Any


ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from ai_models.tutorial_response.answer_classifier import ANSWER_KIND_LABELS
from ai_models.tutorial_response.data import build_source, write_jsonl


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Export high-value hard examples for tutorial answer-classifier retraining.")
    parser.add_argument("--logs", default="runtime/tutorial_response_events.jsonl")
    parser.add_argument("--feedback-logs", default="runtime/tutorial_feedback_events.jsonl")
    parser.add_argument("--output", default="runtime/tutorial_answer_classifier_hard_examples.jsonl")
    parser.add_argument("--report", default="runtime/tutorial_hard_examples_report.json")
    parser.add_argument("--max-records", type=int, default=0)
    parser.add_argument("--low-confidence", type=float, default=0.72)
    parser.add_argument("--high-entropy", type=float, default=1.25)
    parser.add_argument("--margin-threshold", type=float, default=0.25)
    parser.add_argument("--include-test-sessions", action="store_true")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    events = load_jsonl(Path(args.logs))
    feedback = feedback_index(load_jsonl(Path(args.feedback_logs)))
    records: list[dict[str, Any]] = []
    seen: set[tuple[str, str, str]] = set()
    report = {
        "events": len(events),
        "records": 0,
        "reasons": {},
        "labels": {label: 0 for label in ANSWER_KIND_LABELS},
    }

    for event in events:
        if event.get("schema") != "tutorial_response_event_v1":
            continue
        if not args.include_test_sessions and is_test_session(event):
            continue
        feedback_event = feedback.get(event_key(event), {})
        record = event_to_hard_record(
            event,
            feedback_event,
            low_confidence=args.low_confidence,
            high_entropy=args.high_entropy,
            margin_threshold=args.margin_threshold,
        )
        if record is None:
            continue
        key = (record["source"], record["kind"], record["hardReason"])
        if key in seen:
            continue
        seen.add(key)
        records.append(record)
        report["reasons"][record["hardReason"]] = report["reasons"].get(record["hardReason"], 0) + 1
        report["labels"][record["kind"]] = report["labels"].get(record["kind"], 0) + 1
        if args.max_records and len(records) >= args.max_records:
            break

    report["records"] = len(records)
    write_jsonl(args.output, records)
    Path(args.report).parent.mkdir(parents=True, exist_ok=True)
    Path(args.report).write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps({
        "event": "tutorial_hard_examples_exported",
        "output": args.output,
        "report": args.report,
        **report,
    }, ensure_ascii=False, indent=2))


def event_to_hard_record(
    event: dict[str, Any],
    feedback_event: dict[str, Any],
    low_confidence: float,
    high_entropy: float,
    margin_threshold: float,
) -> dict[str, Any] | None:
    label = label_from_feedback(feedback_event)
    rating = feedback_rating(feedback_event)
    classifier = event.get("answerClassifier") if isinstance(event.get("answerClassifier"), dict) else {}
    confidence = confidence_value(classifier)
    ranked = ranked_items(classifier)
    entropy = distribution_entropy(ranked)
    margin = top_margin(ranked)
    reason = ""

    if label:
        reason = f"feedback_{rating or 'label'}"
    elif rating in {"down", "fix"}:
        label = str(event.get("interpreted") or "")
        reason = f"feedback_{rating}"
    elif confidence is not None and confidence <= low_confidence:
        label = str(classifier.get("kind") or event.get("interpreted") or "")
        reason = "low_confidence"
    elif entropy >= high_entropy:
        label = str(classifier.get("kind") or event.get("interpreted") or "")
        reason = "high_entropy"
    elif margin is not None and margin <= margin_threshold:
        label = str(classifier.get("kind") or event.get("interpreted") or "")
        reason = "low_margin"
    else:
        return None

    if label not in ANSWER_KIND_LABELS:
        return None
    source_payload = dict(event.get("sourcePayload") or {})
    if not source_payload:
        source_payload = {
            "projectId": event.get("projectId") or "desk_pet",
            "projectTitle": event.get("projectTitle") or "",
            "currentStage": event.get("currentStage") or "orient",
            "question": event.get("question") or "",
            "answer": event.get("answer") or "",
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
        "feedback": rating or "none",
        "hardReason": reason,
        "classifierConfidence": confidence,
        "classifierEntropy": round(entropy, 6),
        "classifierMargin": None if margin is None else round(margin, 6),
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


def ranked_items(classifier: dict[str, Any]) -> list[dict[str, Any]]:
    ranked = classifier.get("ranked") if isinstance(classifier.get("ranked"), list) else []
    return [item for item in ranked if isinstance(item, dict)]


def distribution_entropy(ranked: list[dict[str, Any]]) -> float:
    entropy = 0.0
    for item in ranked:
        probability = probability_value(item)
        if probability and probability > 0:
            entropy -= probability * math.log(probability)
    return entropy


def top_margin(ranked: list[dict[str, Any]]) -> float | None:
    probabilities = sorted((probability_value(item) or 0.0 for item in ranked), reverse=True)
    if len(probabilities) < 2:
        return None
    return probabilities[0] - probabilities[1]


def probability_value(item: dict[str, Any]) -> float | None:
    try:
        return float(item.get("probability"))
    except (TypeError, ValueError):
        return None


def confidence_value(classifier: dict[str, Any]) -> float | None:
    try:
        return float(classifier.get("confidence"))
    except (TypeError, ValueError):
        return None


def load_jsonl(path: Path) -> list[dict[str, Any]]:
    if not path.exists():
        return []
    with path.open(encoding="utf-8") as handle:
        return [json.loads(line) for line in handle if line.strip()]


def feedback_index(events: list[dict[str, Any]]) -> dict[tuple[str, str, str], dict[str, Any]]:
    result: dict[tuple[str, str, str], dict[str, Any]] = {}
    for event in events:
        if event.get("schema") == "tutorial_feedback_event_v1":
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
