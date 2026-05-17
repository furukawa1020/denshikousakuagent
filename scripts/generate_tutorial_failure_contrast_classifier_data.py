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
    parser = argparse.ArgumentParser(description="Generate contrastive success/failure classifier records for upload and serial wording.")
    parser.add_argument("--output", default="runtime/tutorial_answer_classifier_failure_contrast_v1.jsonl")
    parser.add_argument("--samples", type=int, default=120000)
    parser.add_argument("--seed", type=int, default=51920)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    rng = random.Random(args.seed)
    rows = [make_record(rng) for _ in range(args.samples)]
    write_jsonl(args.output, rows)
    counts: dict[str, int] = {}
    for row in rows:
        counts[row["kind"]] = counts.get(row["kind"], 0) + 1
    print({"event": "tutorial_failure_contrast_classifier_data_generated", "output": args.output, "records": len(rows), "kinds": counts})


def make_record(rng: random.Random) -> dict[str, Any]:
    scenario = rng.choices(SCENARIOS, weights=[item["weight"] for item in SCENARIOS], k=1)[0]
    project_id = rng.choice(list(PROJECTS))
    payload = {
        "projectId": project_id,
        "projectTitle": PROJECTS[project_id],
        "currentStage": scenario["stage"],
        "tutorialState": scenario["state"],
        "question": rng.choice(scenario["questions"]),
        "answer": mutate(rng.choice(scenario["answers"]), rng),
        "inventory": "ESP32、LED、220Ω抵抗、ブレッドボード、USBケーブル",
        "budget": "5000",
        "symptom": scenario.get("symptom", "none") if rng.random() < 0.4 else "none",
        "skill": "" if rng.random() < 0.75 else rng.choice(["Lチカ経験あり", "書き込みは初めて", "シリアルモニタは初めて"]),
        "previous": "" if rng.random() < 0.75 else "GNDとLEDと抵抗は確認済み",
        "lastQuestion": "" if rng.random() < 0.75 else "LEDとGPIOの間に抵抗は入っていますか？",
        "lastAnswer": "" if rng.random() < 0.75 else "抵抗は入っています",
        "lastInterpreted": "" if rng.random() < 0.75 else "confirmed",
        "interpretedKind": "",
        "fallbackStage": "" if rng.random() < 0.75 else scenario["fallbackStage"],
        "tutorialTopic": "" if rng.random() < 0.75 else scenario["topic"],
    }
    return {
        "source": build_source(payload),
        "kind": scenario["kind"],
        "target": scenario["kind"],
        "feedback": "fix",
        "generator": "tutorial_failure_contrast_classifier_v1",
        "topic": scenario["topic"],
    }


def mutate(text: str, rng: random.Random) -> str:
    prefix = rng.choice(["", "", "", "いま見ると、", "確認したら、", "Arduino IDEでは、"])
    suffix = rng.choice(["", "", "です", "でした", "。", "、次は？"])
    return f"{prefix}{text}{suffix}"


SCENARIOS = [
    {
        "topic": "upload_success_contrast",
        "kind": "confirmed",
        "stage": "firmware_upload",
        "fallbackStage": "observe_serial",
        "state": {"gnd": True, "led": True, "resistor": True, "firmware": False},
        "weight": 12,
        "questions": ["確認コードを書き込めましたか？", "アップロードは成功しましたか？", "ボードとポートを選んで書き込めましたか？"],
        "answers": ["書き込みできました", "アップロードできました", "Done uploading と出ています", "エラーなしで入りました", "COMポートを選んで書き込めました", "成功しました"],
    },
    {
        "topic": "upload_failure_contrast",
        "kind": "problem_report",
        "stage": "firmware_upload",
        "fallbackStage": "debug_triage",
        "state": {"gnd": True, "led": True, "resistor": True, "firmware": False},
        "weight": 18,
        "questions": ["確認コードを書き込めましたか？", "アップロードは成功しましたか？", "ボードとポートを選んで書き込めましたか？"],
        "answers": ["書き込みできません", "アップロードできません", "COMポートが出ません", "COMポートが見えません", "Failed to connect と出ます", "A fatal error occurred と出ます", "checkpoint loadedで止まります", "ボードが認識されません", "エラーが出ています"],
        "symptom": "upload_failed",
    },
    {
        "topic": "serial_success_contrast",
        "kind": "confirmed",
        "stage": "observe_serial",
        "fallbackStage": "standard_build",
        "state": {"gnd": True, "led": True, "resistor": True, "firmware": True},
        "weight": 12,
        "questions": ["シリアルモニタに起動メッセージや値は出ていますか？", "Serial Monitorに値が出ていますか？", "手を近づけると値は変わりますか？"],
        "answers": ["start と値が出ています", "数値が出ています", "ログが見えています", "手を近づけると値が変わります", "起動メッセージが出ています", "センサー値が変わりました"],
    },
    {
        "topic": "serial_failure_contrast",
        "kind": "problem_report",
        "stage": "observe_serial",
        "fallbackStage": "debug_triage",
        "state": {"gnd": True, "led": True, "resistor": True, "firmware": True},
        "weight": 18,
        "questions": ["シリアルモニタに起動メッセージや値は出ていますか？", "Serial Monitorに値が出ていますか？", "手を近づけると値は変わりますか？"],
        "answers": ["何も出ません", "値が出ません", "ログが出ません", "起動メッセージが出ません", "文字化けしています", "ずっと0です", "値が変わりません", "真っ白です", "エラーが出ています"],
        "symptom": "serial_problem",
    },
]


if __name__ == "__main__":
    main()
