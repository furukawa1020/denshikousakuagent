from __future__ import annotations

import argparse
import random
import sys
from pathlib import Path
from typing import Any


ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from ai_models.tutorial_response.data import build_source, write_jsonl
from ai_models.tutorial_response.taxonomy import PROJECTS


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Generate hard classifier records for firmware and serial progress states.")
    parser.add_argument("--output", default="runtime/tutorial_answer_classifier_progress_v1.jsonl")
    parser.add_argument("--samples", type=int, default=60000)
    parser.add_argument("--seed", type=int, default=51901)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    rng = random.Random(args.seed)
    rows = [make_record(rng) for _ in range(args.samples)]
    write_jsonl(args.output, rows)
    counts: dict[str, int] = {}
    topics: dict[str, int] = {}
    for row in rows:
        counts[row["kind"]] = counts.get(row["kind"], 0) + 1
        topics[row["topic"]] = topics.get(row["topic"], 0) + 1
    print({"event": "tutorial_progress_classifier_data_generated", "output": args.output, "records": len(rows), "kinds": counts, "topics": topics})


def make_record(rng: random.Random) -> dict[str, Any]:
    scenario = rng.choices(SCENARIOS, weights=[item["weight"] for item in SCENARIOS], k=1)[0]
    project_id = rng.choice(list(PROJECTS))
    include_context = rng.random() < 0.55
    payload = {
        "projectId": project_id,
        "projectTitle": PROJECTS[project_id],
        "currentStage": scenario["stage"],
        "tutorialState": scenario["state"],
        "question": rng.choice(scenario["questions"]),
        "answer": mutate(rng.choice(scenario["answers"]), rng),
        "inventory": rng.choice(INVENTORIES),
        "budget": rng.choice(["3000", "5000", "10000"]),
        "symptom": scenario.get("symptom", "none") if rng.random() < 0.45 else "none",
        "skill": rng.choice(["Lチカ経験あり", "書き込みは初めて", "シリアルモニタは初めて", ""]) if include_context else "",
        "previous": rng.choice(["GNDとLEDと抵抗は確認済み", "最小回路までは進んだ", ""]) if include_context else "",
        "lastQuestion": rng.choice(scenario.get("lastQuestions") or ["LEDとGPIOの間に抵抗は入っていますか？"]) if include_context else "",
        "lastAnswer": rng.choice(scenario.get("lastAnswers") or ["抵抗は入っています"]) if include_context else "",
        "lastInterpreted": "confirmed" if include_context else "",
        "interpretedKind": "",
        "fallbackStage": scenario["fallbackStage"] if rng.random() < 0.45 else "",
        "tutorialTopic": scenario["topic"] if rng.random() < 0.45 else "",
    }
    kind = scenario["kind"]
    return {
        "source": build_source(payload),
        "kind": kind,
        "target": kind,
        "feedback": "fix",
        "generator": "tutorial_progress_classifier_v1",
        "topic": scenario["topic"],
    }


def mutate(text: str, rng: random.Random) -> str:
    prefix = rng.choice(["", "", "確認したら、", "いまは、", "たぶん", "Arduino IDEでは、"])
    suffix = rng.choice(["", "", "です", "でした", "。次は？", "と思います"])
    return f"{prefix}{text}{suffix}"


INVENTORIES = [
    "ESP32、LED、220Ω抵抗、ブレッドボード、USBケーブル",
    "Arduino Uno、LED、330Ω抵抗、ジャンパ線",
    "M5Stack、Groveケーブル、LED",
]


SCENARIOS = [
    {
        "topic": "firmware_upload_success",
        "kind": "confirmed",
        "stage": "firmware_upload",
        "fallbackStage": "observe_serial",
        "state": {"gnd": True, "led": True, "resistor": True, "firmware": False},
        "weight": 18,
        "questions": ["確認コードを書き込めましたか？", "ボードとポートを選んで書き込めましたか？", "アップロードは成功しましたか？"],
        "answers": ["書き込みできました", "アップロード成功しました", "エラーなしで入りました", "Done uploading と出ました", "COMポートを選んで書き込めました", "ボードに入りました"],
    },
    {
        "topic": "firmware_upload_problem",
        "kind": "problem_report",
        "stage": "firmware_upload",
        "fallbackStage": "debug_triage",
        "state": {"gnd": True, "led": True, "resistor": True, "firmware": False},
        "weight": 10,
        "questions": ["確認コードを書き込めましたか？", "ボードとポートを選んで書き込めましたか？", "アップロードは成功しましたか？"],
        "answers": ["COMポートが出ません", "Failed to connect と出ます", "checkpoint loadedで止まります", "書き込みエラーが出ます", "A fatal error occurred と出ます", "ボードが認識されません"],
        "symptom": "upload_failed",
    },
    {
        "topic": "serial_success",
        "kind": "confirmed",
        "stage": "observe_serial",
        "fallbackStage": "standard_build",
        "state": {"gnd": True, "led": True, "resistor": True, "firmware": True},
        "weight": 18,
        "questions": ["シリアルモニタに起動メッセージや値は出ていますか？", "Serial Monitorに値が出ていますか？", "手を近づけると値は変わりますか？"],
        "answers": ["start と値が出ています", "ログが見えています", "数値が出ています", "手を近づけると値が変わります", "起動メッセージが出ました", "センサー値が変わりました"],
    },
    {
        "topic": "serial_problem",
        "kind": "problem_report",
        "stage": "observe_serial",
        "fallbackStage": "debug_triage",
        "state": {"gnd": True, "led": True, "resistor": True, "firmware": True},
        "weight": 10,
        "questions": ["シリアルモニタに起動メッセージや値は出ていますか？", "Serial Monitorに値が出ていますか？", "手を近づけると値は変わりますか？"],
        "answers": ["何も出ません", "文字化けしています", "ずっと0です", "値が変わりません", "真っ白です", "エラーが出ます"],
        "symptom": "serial_problem",
    },
    {
        "topic": "minimal_confirmation",
        "kind": "confirmed",
        "stage": "minimal_circuit",
        "fallbackStage": "firmware_upload",
        "state": {"gnd": True, "led": True, "resistor": False, "firmware": False},
        "weight": 5,
        "questions": ["LEDとGPIOの間に220Ω前後の抵抗は入っていますか？", "LEDの長い足は抵抗を通ってGPIO側ですか？"],
        "answers": ["抵抗を一本はさんでいます", "長い足が抵抗経由でGPIO側です", "直結ではなく抵抗経由です"],
    },
]


if __name__ == "__main__":
    main()
