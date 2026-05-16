from __future__ import annotations

import argparse
import random
import sys
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from ai_models.tutorial_response.data import build_source, write_jsonl
from ai_models.tutorial_response.taxonomy import PROJECTS


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Generate hard free-answer classifier data for tutorial flow.")
    parser.add_argument("--output", default="runtime/tutorial_answer_classifier_hard_generated_v1.jsonl")
    parser.add_argument("--samples", type=int, default=24000)
    parser.add_argument("--seed", type=int, default=916)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    rng = random.Random(args.seed)
    rows = [make_record(rng) for _ in range(args.samples)]
    write_jsonl(args.output, rows)
    counts: dict[str, int] = {}
    for row in rows:
        counts[row["kind"]] = counts.get(row["kind"], 0) + 1
    print({"event": "answer_classifier_hard_data_generated", "output": args.output, "records": len(rows), "counts": counts})


def make_record(rng: random.Random) -> dict[str, str]:
    scenario = rng.choices(SCENARIOS, weights=[item.get("weight", 1) for item in SCENARIOS], k=1)[0]
    project_id = rng.choice(list(PROJECTS))
    question = rng.choice(scenario["questions"])
    answer = mutate(rng.choice(scenario["answers"]), rng)
    source = build_source({
        "projectId": project_id,
        "projectTitle": PROJECTS[project_id],
        "currentStage": scenario["stage"],
        "question": question,
        "answer": answer,
        "inventory": rng.choice([
            "ESP32, LED, 220Ω抵抗, ブレッドボード, ジャンパ線",
            "Arduino Uno, LED, 330Ω resistor, breadboard",
            "M5Stack, Grove sensor, LED",
            "LEDと抵抗とジャンパ線はあります",
            "部品名はあいまい。LEDっぽいものと線があります",
        ]),
        "budget": rng.choice(["1000", "3000", "5000", "10000"]),
        "symptom": scenario.get("symptom", "none"),
        "skill": rng.choice(["", "LEDは少し分かる", "GND共有が不安", "コードは初めて"]),
        "previous": rng.choice(["", "Lチカだけやった", "同じ質問で止まった", "配線で詰まった"]),
        "interpretedKind": "",
        "fallbackStage": scenario.get("fallbackStage", scenario["stage"]),
        "tutorialTopic": scenario["topic"],
    })
    return {
        "source": source,
        "kind": scenario["kind"],
        "target": scenario["kind"],
        "feedback": "fix",
        "generator": "tutorial_answer_classifier_hard_generated",
        "topic": scenario["topic"],
    }


def mutate(text: str, rng: random.Random) -> str:
    prefix = rng.choice(["", "", "今見たら、", "たぶん、", "確認したら、", "I think ", "maybe "])
    suffix = rng.choice(["", "", "です", "です。", "と思います", "、次は？", ". what next?"])
    text = text.replace("GND", rng.choice(["GND", "gnd", "グランド"]) if rng.random() < 0.18 else "GND")
    text = text.replace("GPIO", rng.choice(["GPIO", "gpio", "ピン"]) if rng.random() < 0.12 else "GPIO")
    text = text.replace("抵抗", rng.choice(["抵抗", "220Ω", "330Ω", "resistor"]) if rng.random() < 0.16 else "抵抗")
    return f"{prefix}{text}{suffix}"


SCENARIOS = [
    {
        "topic": "led_polarity_confirmed",
        "kind": "confirmed",
        "stage": "minimal_circuit",
        "fallbackStage": "minimal_circuit",
        "weight": 6,
        "questions": [
            "LEDの長い足は、抵抗を通ってGPIO側につながっていますか？",
            "LEDの向きは合っていますか？",
            "Is the long leg of the LED on the GPIO side through a resistor?",
        ],
        "answers": [
            "長い足が抵抗を通ってGPIO側につながっています",
            "LEDの長い足はGPIO側、短い足はGND側です",
            "long leg goes through resistor to GPIO",
            "アノードが抵抗経由でGPIO、カソードがGNDです",
            "向きは合っています",
        ],
    },
    {
        "topic": "resistor_confirmed",
        "kind": "confirmed",
        "stage": "minimal_circuit",
        "fallbackStage": "firmware_upload",
        "weight": 6,
        "questions": [
            "LEDとGPIOの間に220Ω前後の抵抗は入っていますか？",
            "LEDとGPIOの間に抵抗は入っていますか？",
            "Is there a resistor between LED and GPIO?",
        ],
        "answers": [
            "抵抗を一本はさんでいます",
            "220Ωを直列に入れています",
            "resistor is in series between LED and GPIO",
            "LEDとGPIOの間に抵抗が入っています",
            "直結ではなく抵抗経由です",
        ],
    },
    {
        "topic": "gnd_confirmed",
        "kind": "confirmed",
        "stage": "minimal_circuit",
        "fallbackStage": "minimal_circuit",
        "weight": 5,
        "questions": [
            "センサー、LED、ボードのGNDは同じGND列につながっていますか？",
            "GNDは共通ですか？",
            "Are all grounds shared?",
        ],
        "answers": [
            "同じGND列につながっています",
            "全部同じグランドに入っています",
            "all GND are connected to the same row",
            "GNDは共通です",
            "ボードのGNDとLEDのマイナス側は同じ列です",
        ],
    },
    {
        "topic": "negative_or_missing",
        "kind": "negative",
        "stage": "minimal_circuit",
        "weight": 4,
        "questions": [
            "LEDとGPIOの間に抵抗は入っていますか？",
            "GNDは同じ列につながっていますか？",
            "LEDの向きは合っていますか？",
        ],
        "answers": [
            "まだです",
            "入っていません",
            "別の列かもしれない",
            "直結しているかも",
            "向きが逆かもしれません",
            "not yet",
            "no, it is not connected",
        ],
    },
    {
        "topic": "unknown_uncertain",
        "kind": "unknown",
        "stage": "minimal_circuit",
        "weight": 4,
        "questions": [
            "LEDの長い足は、抵抗を通ってGPIO側につながっていますか？",
            "GNDは同じ列につながっていますか？",
            "シリアルモニタに数値や起動メッセージは出ていますか？",
        ],
        "answers": [
            "分からない",
            "自信がない",
            "どの列か分からない",
            "たぶん違うかも",
            "見ても判断できない",
            "not sure",
            "I cannot tell",
        ],
    },
    {
        "topic": "problem_report",
        "kind": "problem_report",
        "stage": "firmware_upload",
        "symptom": "upload_failed",
        "weight": 3,
        "questions": [
            "Arduino IDEやエディタで、ボード名とポートは選べていますか？",
            "書き込みはできましたか？",
            "シリアルモニタに数値や起動メッセージは出ていますか？",
        ],
        "answers": [
            "COMポートが出ません",
            "書き込みできない",
            "checkpoint loadedで止まる",
            "エラーが出ています",
            "Failed to connect",
            "何も出ない",
        ],
    },
    {
        "topic": "photo_request",
        "kind": "photo_request",
        "stage": "minimal_circuit",
        "weight": 2,
        "questions": [
            "GNDは同じ列につながっていますか？",
            "LEDの向きは合っていますか？",
            "LEDとGPIOの間に抵抗は入っていますか？",
        ],
        "answers": [
            "写真で見てほしい",
            "画像を送ります",
            "配線写真で確認したい",
            "Can I send a photo?",
            "写真チェックしたい",
        ],
    },
    {
        "topic": "inventory_report",
        "kind": "inventory_report",
        "stage": "parts_check",
        "weight": 3,
        "questions": [
            "手元にある部品を、分かる範囲で一行で書けますか？",
            "持っている部品を教えてください",
        ],
        "answers": [
            "ESP32, LED, 220Ω抵抗, ブレッドボード, ジャンパ線",
            "ArduinoとLEDと抵抗があります",
            "M5StackとGroveセンサーがあります",
            "LED、センサー、USBケーブル",
        ],
    },
    {
        "topic": "board_report",
        "kind": "board_report",
        "stage": "parts_check",
        "weight": 2,
        "questions": [
            "使うボードは Arduino、ESP32、M5Stack、Pico のどれですか？",
            "ボード名は分かりますか？",
        ],
        "answers": [
            "ESP32です",
            "Arduino Unoを使います",
            "M5Stackです",
            "Raspberry Pi Picoだと思います",
            "micro:bitです",
        ],
    },
]


if __name__ == "__main__":
    main()
