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
    parser = argparse.ArgumentParser(description="Generate explicit next-stage supervision for autonomous tutorial routing.")
    parser.add_argument("--output", default="runtime/tutorial_stage_autonomy_v1.jsonl")
    parser.add_argument("--samples", type=int, default=120000)
    parser.add_argument("--seed", type=int, default=52019)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    rng = random.Random(args.seed)
    rows = [make_record(rng) for _ in range(args.samples)]
    write_jsonl(args.output, rows)
    counts: dict[str, int] = {}
    for row in rows:
        counts[row["stage"]] = counts.get(row["stage"], 0) + 1
    print({"event": "tutorial_stage_autonomy_data_generated", "output": args.output, "records": len(rows), "stages": counts})


def make_record(rng: random.Random) -> dict[str, Any]:
    scenario = rng.choices(SCENARIOS, weights=[item["weight"] for item in SCENARIOS], k=1)[0]
    project_id = rng.choice(list(PROJECTS))
    payload = {
        "projectId": project_id,
        "projectTitle": PROJECTS[project_id],
        "currentStage": scenario["currentStage"],
        "tutorialState": dict(scenario["state"]),
        "question": rng.choice(scenario["questions"]),
        "answer": mutate(rng.choice(scenario["answers"]), rng),
        "interpretedKind": scenario["kind"],
        "fallbackStage": "" if rng.random() < 0.6 else scenario["stage"],
        "tutorialTopic": "" if rng.random() < 0.6 else scenario["topic"],
        "inventory": rng.choice(INVENTORIES),
        "budget": rng.choice(["3000", "5000", "10000"]),
        "symptom": scenario.get("symptom", "none") if rng.random() < 0.3 else "none",
        "skill": rng.choice(["", "Lチカだけ経験あり", "GND共有が少し不安", "書き込みは初めて"]),
        "previous": rng.choice(["", "部品確認から進んだ", "GNDとLEDは見た", "抵抗まで確認した"]),
        "lastQuestion": rng.choice(scenario.get("lastQuestions") or [""]),
        "lastAnswer": rng.choice(scenario.get("lastAnswers") or [""]),
        "lastInterpreted": scenario.get("lastInterpreted", ""),
    }
    return {
        "source": build_source(payload),
        "stage": scenario["stage"],
        "target": scenario["stage"],
        "feedback": "fix",
        "generator": "tutorial_stage_autonomy_v1",
        "topic": scenario["topic"],
    }


def mutate(text: str, rng: random.Random) -> str:
    prefix = rng.choice(["", "", "", "確認したら、", "いま見ると、", "たぶん、"])
    suffix = rng.choice(["", "", "です", "でした", "。次は？", "と思います"])
    return f"{prefix}{text}{suffix}"


INVENTORIES = [
    "ESP32、LED、220Ω抵抗、ブレッドボード、USBケーブル",
    "Arduino Uno、LED、330Ω抵抗、ジャンパ線",
    "M5Stack、Groveケーブル、LED",
    "LEDと抵抗とブレッドボードがあります",
]


SCENARIOS = [
    {
        "topic": "inventory_to_minimal",
        "kind": "inventory_report",
        "currentStage": "parts_check",
        "stage": "minimal_circuit",
        "state": {"gnd": False, "led": False, "resistor": False, "firmware": False},
        "weight": 6,
        "questions": ["手元にある部品を、分かる範囲で一行で書けますか？"],
        "answers": ["ESP32、LED、抵抗、ブレッドボードがあります", "ArduinoとLEDとジャンパ線があります"],
    },
    {
        "topic": "gnd_to_led",
        "kind": "confirmed",
        "currentStage": "minimal_circuit",
        "stage": "minimal_circuit",
        "state": {"gnd": False, "led": False, "resistor": False, "firmware": False},
        "weight": 8,
        "questions": ["センサー、LED、ボードのGNDは同じGND列につながっていますか？"],
        "answers": ["同じGND列につながっています", "全部同じグランドに入っています"],
    },
    {
        "topic": "gnd_problem_stays",
        "kind": "negative",
        "currentStage": "minimal_circuit",
        "stage": "minimal_circuit",
        "state": {"gnd": False, "led": False, "resistor": False, "firmware": False},
        "weight": 5,
        "questions": ["センサー、LED、ボードのGNDは同じGND列につながっていますか？"],
        "answers": ["別の列かもしれません", "つながっていません", "同じか分かりません"],
    },
    {
        "topic": "led_to_resistor",
        "kind": "confirmed",
        "currentStage": "minimal_circuit",
        "stage": "minimal_circuit",
        "state": {"gnd": True, "led": False, "resistor": False, "firmware": False},
        "weight": 8,
        "questions": ["LEDの長い足は、抵抗を通ってGPIO側につながっていますか？", "LEDの向きは合っていますか？"],
        "answers": ["長い足が抵抗を通ってGPIO側につながっています", "LEDの長い足はGPIO側、短い足はGND側です"],
    },
    {
        "topic": "resistor_to_firmware",
        "kind": "confirmed",
        "currentStage": "minimal_circuit",
        "stage": "firmware_upload",
        "state": {"gnd": True, "led": True, "resistor": False, "firmware": False},
        "lastQuestions": ["LEDの長い足は、抵抗を通ってGPIO側につながっていますか？"],
        "lastAnswers": ["長い足が抵抗経由でGPIO側です"],
        "lastInterpreted": "confirmed",
        "weight": 18,
        "questions": ["LEDとGPIOの間に220Ω前後の抵抗は入っていますか？", "LEDとGPIOの間に抵抗が入っていますか？"],
        "answers": ["220Ωの抵抗を一本はさんでいます", "抵抗は直列に入っています", "LEDとGPIOの間に抵抗があります", "直結ではなく抵抗経由です"],
    },
    {
        "topic": "resistor_missing_stays",
        "kind": "negative",
        "currentStage": "minimal_circuit",
        "stage": "minimal_circuit",
        "state": {"gnd": True, "led": True, "resistor": False, "firmware": False},
        "weight": 8,
        "questions": ["LEDとGPIOの間に220Ω前後の抵抗は入っていますか？", "LEDとGPIOの間に抵抗が入っていますか？"],
        "answers": ["抵抗なしでつないでいます", "ジャンパ線だけです", "抵抗が見当たりません", "直結かもしれません"],
    },
    {
        "topic": "upload_to_serial",
        "kind": "confirmed",
        "currentStage": "firmware_upload",
        "stage": "observe_serial",
        "state": {"gnd": True, "led": True, "resistor": True, "firmware": False},
        "weight": 12,
        "questions": ["確認コードを書き込めましたか？", "アップロードは成功しましたか？"],
        "answers": ["書き込みできました", "Done uploading と出ています", "エラーなしで入りました"],
    },
    {
        "topic": "upload_to_debug",
        "kind": "problem_report",
        "currentStage": "firmware_upload",
        "stage": "debug_triage",
        "state": {"gnd": True, "led": True, "resistor": True, "firmware": False},
        "weight": 10,
        "questions": ["確認コードを書き込めましたか？", "アップロードは成功しましたか？"],
        "answers": ["checkpoint loadedで止まります", "COMポートが出ません", "Failed to connect と出ます", "書き込みエラーが出ます"],
    },
    {
        "topic": "serial_to_standard",
        "kind": "confirmed",
        "currentStage": "observe_serial",
        "stage": "standard_build",
        "state": {"gnd": True, "led": True, "resistor": True, "firmware": True},
        "weight": 10,
        "questions": ["シリアルモニタに起動メッセージや値は出ていますか？"],
        "answers": ["start と値が出ています", "数値が出ています", "手を近づけると値が変わります"],
    },
    {
        "topic": "serial_to_debug",
        "kind": "problem_report",
        "currentStage": "observe_serial",
        "stage": "debug_triage",
        "state": {"gnd": True, "led": True, "resistor": True, "firmware": True},
        "weight": 8,
        "questions": ["シリアルモニタに起動メッセージや値は出ていますか？"],
        "answers": ["何も出ません", "文字化けしています", "ずっと0です", "値が変わりません"],
    },
    {
        "topic": "standard_to_enclosure",
        "kind": "confirmed",
        "currentStage": "standard_build",
        "stage": "enclosure",
        "state": {"gnd": True, "led": True, "resistor": True, "firmware": True},
        "weight": 4,
        "questions": ["反応条件と見た目を整えられましたか？"],
        "answers": ["反応するようになりました", "標準構成までできました"],
    },
    {
        "topic": "enclosure_to_extension",
        "kind": "confirmed",
        "currentStage": "enclosure",
        "stage": "extension",
        "state": {"gnd": True, "led": True, "resistor": True, "firmware": True},
        "weight": 3,
        "questions": ["ケースや固定はできましたか？"],
        "answers": ["ケースに入れました", "固定できました"],
    },
]


if __name__ == "__main__":
    main()
