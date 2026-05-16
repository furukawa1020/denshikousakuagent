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
    parser = argparse.ArgumentParser(description="Generate targeted hard data for tutorial response generation.")
    parser.add_argument("--output", default="runtime/tutorial_response_targeted_hard_data.jsonl")
    parser.add_argument("--samples", type=int, default=2400)
    parser.add_argument("--seed", type=int, default=515)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    rng = random.Random(args.seed)
    records = [make_record(rng, index) for index in range(args.samples)]
    write_jsonl(args.output, records)
    print({"event": "targeted_hard_data_generated", "output": args.output, "records": len(records)})


def make_record(rng: random.Random, index: int) -> dict[str, str]:
    scenario = rng.choices(SCENARIOS, weights=[scenario.get("weight", 1) for scenario in SCENARIOS], k=1)[0]
    project_id = rng.choice(list(PROJECTS))
    source = build_source({
        "projectId": project_id,
        "projectTitle": PROJECTS[project_id],
        "currentStage": scenario["stage"],
        "question": rng.choice(scenario["questions"]),
        "answer": mutate(rng.choice(scenario["answers"]), rng),
        "inventory": rng.choice([
            "ESP32, LED, resistor",
            "ESP32, LED, 220 ohm resistor, breadboard",
            "Arduino Uno, LED, resistor, jumper wires",
            "ESP32 LED 抵抗 ブレッドボード",
            "not sure, maybe LED and wires",
        ]),
        "budget": rng.choice(["1000", "3000", "5000", "10000"]),
        "symptom": scenario.get("symptom", "none"),
        "skill": rng.choice(["", "GNDが不安", "LEDだけは少し分かる"]),
        "previous": rng.choice(["", "Lチカだけやった", "配線で止まった"]),
        "interpretedKind": scenario["interpretedKind"],
        "fallbackStage": scenario["nextStage"],
        "tutorialTopic": scenario["topic"] if rng.random() < scenario.get("topicProbability", 0.5) else "",
    })
    title, body, instruction = rng.choice(scenario["targets"])
    target = "\n".join([
        f"style:{scenario['style']}",
        f"title:{title}",
        f"body:{body}",
        f"next:{instruction}",
        f"stage:{scenario['nextStage']}",
    ])
    return {
        "source": source,
        "target": target,
        "feedback": "fix",
        "generator": "targeted_hard_data",
        "topic": scenario["topic"],
    }


def mutate(text: str, rng: random.Random) -> str:
    prefix = rng.choice(["", "", "今は、", "見た感じ、", "I think ", "maybe "])
    suffix = rng.choice(["", "", "です", "と思います", ". what next?", "、次は？"])
    return f"{prefix}{text}{suffix}"


SCENARIOS = [
    {
        "topic": "gnd_unknown",
        "stage": "minimal_circuit",
        "nextStage": "minimal_circuit",
        "style": "warn",
        "interpretedKind": "unknown",
        "weight": 4,
        "topicProbability": 0.2,
        "questions": [
            "I am not sure if GND is shared. What should I check?",
            "センサー、LED、ボードのGNDは同じGND列につながっていますか？",
            "GNDが同じか自信がないです。次は何を見る？",
        ],
        "answers": ["not sure, maybe not connected", "GNDは自信ない", "同じ列か分からない", "maybe separated ground"],
        "targets": [
            ("GND共有を先に確認します", "GNDが不安な状態として受け取りました。USBを抜いて、ボードのGND、LEDの短い足側、センサーのGNDが同じマイナス列へ入っているかだけ見ます。", "同じ列なら「同じGND」、違いそうなら「別かも」と答えてください。"),
            ("ここはGNDで止まって正解です", "GNDがそろっていないと、コードが合っていても反応しません。まず一本ずつGND線だけを追います。", "USBを抜いてからGND線を確認し、分からなければ写真確認へ進みます。"),
        ],
        "symptom": "not working",
    },
    {
        "topic": "gnd_confirmed",
        "stage": "minimal_circuit",
        "nextStage": "firmware_upload",
        "style": "good",
        "interpretedKind": "confirmed",
        "weight": 2,
        "questions": [
            "Are the sensor, LED, and board GND on the same ground row?",
            "GNDは同じ列につながっていますか？",
        ],
        "answers": ["same ground row", "同じGND列です", "yes, all GND are connected", "つながっています"],
        "targets": [
            ("GND共有はクリアです", "GNDが同じ列なら、最小回路は次へ進めます。次はコード側のGPIO番号と配線先を合わせます。", "ボード設定とポートを確認して、シリアル出力つきコードを書き込みます。"),
            ("GNDラインは大丈夫そうです", "基準電位がそろっているので、LEDやセンサーの確認に進めます。", "次は確認コードを書き込んで start 表示を見ます。"),
        ],
        "symptom": "none",
    },
    {
        "topic": "led_polarity_confirmed",
        "stage": "minimal_circuit",
        "nextStage": "minimal_circuit",
        "style": "good",
        "interpretedKind": "confirmed",
        "weight": 2,
        "questions": [
            "LED long leg goes through the resistor to GPIO?",
            "LEDの長い足は、抵抗を通ってGPIO側につながっていますか？",
            "LEDの向きは合っていますか？",
        ],
        "answers": ["yes, it goes through the resistor", "長い足が抵抗を通ってGPIO側です", "LED long leg -> resistor -> GPIO", "向きは合っています"],
        "targets": [
            ("LEDの向きは大丈夫そうです", "長い足が抵抗を通ってGPIO側、短い足がGND側なら、LEDの極性は一旦クリアです。", "次は抵抗がLEDとGPIOの間に一本入っているかを確認します。"),
            ("LEDの極性は進められます", "LEDの向きが合っているので、次は保護抵抗とピン番号を分けて確認します。", "抵抗確認に進み、直結になっていないかだけ見ます。"),
        ],
        "symptom": "none",
    },
    {
        "topic": "resistor_confirmed",
        "stage": "minimal_circuit",
        "nextStage": "firmware_upload",
        "style": "good",
        "interpretedKind": "confirmed",
        "weight": 2,
        "questions": [
            "LEDとGPIOの間に220Ω前後の抵抗は入っていますか？",
            "LEDとGPIOの間に抵抗は入っていますか？",
            "Is there a resistor between LED and GPIO?",
        ],
        "answers": ["抵抗を一本はさんでいます", "220Ωを直列に入れています", "resistor is in series", "直結ではなく抵抗経由です"],
        "targets": [
            ("抵抗は入っています", "抵抗が直列に入っていれば、LEDをGPIOへ直結していないので安全側です。", "次はコードのGPIO番号と配線先を一致させます。"),
            ("LED保護はできています", "LEDとGPIOの間に抵抗があるので、次はソフト側の確認へ進めます。", "ボードとポートを確認して、シリアル出力つきコードを書き込みます。"),
        ],
        "symptom": "none",
    },
    {
        "topic": "resistor_missing",
        "stage": "minimal_circuit",
        "nextStage": "minimal_circuit",
        "style": "warn",
        "interpretedKind": "negative",
        "weight": 2,
        "questions": [
            "Is there a resistor between LED and GPIO?",
            "LEDとGPIOの間に抵抗は入っていますか？",
        ],
        "answers": ["no resistor", "抵抗がない", "direct to GPIO", "たぶん直結です"],
        "targets": [
            ("抵抗を直列に入れます", "抵抗なしでLEDをGPIOへ直結するのは避けます。USBを抜いて、LEDとGPIOの間に220Ω前後の抵抗を一本入れます。", "抵抗を足したら「入れた」と答えてください。"),
            ("LED保護を先に入れます", "ここは先へ進まず、抵抗を追加してから確認します。抵抗には向きはありません。", "USBを抜いて抵抗を追加し、LEDだけで再確認します。"),
        ],
        "symptom": "not working",
    },
    {
        "topic": "upload_problem",
        "stage": "firmware_upload",
        "nextStage": "debug_triage",
        "style": "warn",
        "interpretedKind": "problem_report",
        "weight": 1,
        "questions": [
            "Can you upload the sketch?",
            "ボードとポートを選んで書き込めますか？",
        ],
        "answers": ["COM port missing", "Failed to connect", "書き込めない", "checkpoint loadedで止まる"],
        "targets": [
            ("書き込み前の接続確認に戻ります", "書き込みで止まっているので、まずボード名、ポート、USBケーブルを一つずつ確認します。配線はいったん触らなくて大丈夫です。", "Arduino IDEで選んでいるボード名とポート名をそのまま書いてください。"),
            ("PC接続を一つずつ確認します", "エラーが出ている状態として受け取りました。最初にUSBケーブルとポート表示を見ます。", "表示されているエラー文を一行だけ貼ってください。"),
        ],
        "symptom": "upload error",
    },
]


if __name__ == "__main__":
    main()
