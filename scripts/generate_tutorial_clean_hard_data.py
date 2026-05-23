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


PROJECT_IDS = list(PROJECTS)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Generate clean Japanese hard examples for tutorial neural models.")
    parser.add_argument("--answer-output", default="runtime/tutorial_answer_classifier_clean_hard_v1.jsonl")
    parser.add_argument("--stage-output", default="runtime/tutorial_stage_classifier_clean_hard_v1.jsonl")
    parser.add_argument("--samples", type=int, default=36000)
    parser.add_argument("--seed", type=int, default=80523)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    rng = random.Random(args.seed)
    answer_rows: list[dict[str, Any]] = []
    stage_rows: list[dict[str, Any]] = []
    scenarios = build_scenarios()
    for index in range(args.samples):
        scenario = scenarios[index % len(scenarios)]
        payload = make_payload(scenario, rng)
        answer_payload = {
            **payload,
            "interpretedKind": "",
            "fallbackStage": "",
            "tutorialTopic": "",
        }
        source = build_source(answer_payload)
        answer_rows.append({
            "source": source,
            "kind": scenario["kind"],
            "target": scenario["kind"],
            "feedback": "fix",
            "hardReason": scenario["topic"],
            "generator": "clean_japanese_hard_v1",
        })
        stage_rows.append({
            "source": build_source(payload),
            "stage": scenario["nextStage"],
            "nextStage": scenario["nextStage"],
            "target": f"stage:{scenario['nextStage']}",
            "feedback": "fix",
            "hardReason": scenario["topic"],
            "generator": "clean_japanese_hard_v1",
        })
    rng.shuffle(answer_rows)
    rng.shuffle(stage_rows)
    write_jsonl(args.answer_output, answer_rows)
    write_jsonl(args.stage_output, stage_rows)
    print({
        "event": "clean_hard_data_generated",
        "answer_output": args.answer_output,
        "stage_output": args.stage_output,
        "records": len(answer_rows),
        "answer_counts": count(answer_rows, "kind"),
        "stage_counts": count(stage_rows, "stage"),
    })


def make_payload(scenario: dict[str, Any], rng: random.Random) -> dict[str, Any]:
    project_id = rng.choice(PROJECT_IDS)
    answer = rng.choice(scenario["answers"])
    if rng.random() < 0.35:
        answer = rng.choice(["今の状態は、", "見た感じ、", "たぶん、", "確認したら、", ""]) + answer
    if rng.random() < 0.25:
        answer = answer + rng.choice(["です", "でした", "と思います", "。次どうすればいい？", ""])
    return {
        "projectId": project_id,
        "projectTitle": PROJECTS[project_id],
        "currentStage": scenario["stage"],
        "tutorialState": scenario.get("state", {}),
        "question": rng.choice(scenario["questions"]),
        "answer": answer,
        "inventory": rng.choice([
            "ESP32 LED 抵抗 ブレッドボード ジャンパ線 USBケーブル",
            "Arduino Uno LED 220Ω抵抗 ジャンパ線",
            "Raspberry Pi Pico LED ブザー 距離センサー",
            "M5Stack Groveケーブル LED",
            "何を持っているか少しあいまい",
        ]),
        "budget": rng.choice(["0", "1000", "3000", "5000", "10000"]),
        "symptom": scenario.get("symptom", "none"),
        "skill": rng.choice(["Lチカだけやった", "GNDが少し不安", "Arduino IDEは初めて", "配線は苦手", ""]),
        "previous": rng.choice(["部品確認までは終わった", "最小回路を作っている", "書き込みで止まった", ""]),
        "lastQuestion": rng.choice(scenario.get("lastQuestions") or scenario["questions"]),
        "lastAnswer": rng.choice(scenario.get("lastAnswers") or scenario["answers"]),
        "lastInterpreted": scenario.get("lastInterpreted", ""),
        "interpretedKind": scenario["kind"],
        "fallbackStage": scenario["nextStage"],
        "tutorialTopic": scenario["topic"],
    }


def build_scenarios() -> list[dict[str, Any]]:
    return [
        scenario(
            "minimal_led_confirmed",
            "minimal_circuit",
            "confirmed",
            "firmware_upload",
            [
                "LEDの長い足は、抵抗を通ってGPIO側につながっていますか？",
                "LEDとGPIOの間に220Ωくらいの抵抗は入っていますか？",
                "GNDは同じGND列につながっていますか？",
            ],
            ["はい、つながっています", "抵抗を通ってGPIO側です", "同じGND列につながっています", "LEDの向きも抵抗も大丈夫です", "できました"],
            {"gnd": True, "led": True, "resistor": True},
        ),
        scenario(
            "minimal_led_unknown",
            "minimal_circuit",
            "unknown",
            "minimal_circuit",
            [
                "LEDの長い足は、抵抗を通ってGPIO側につながっていますか？",
                "GNDは同じGND列につながっていますか？",
            ],
            ["わかりません", "自信ないです", "たぶん合ってるけど不安", "どこを見ればいいかわからない", "写真で見てほしい"],
            {"gnd": False, "led": False, "resistor": False},
        ),
        scenario(
            "minimal_led_negative",
            "minimal_circuit",
            "negative",
            "minimal_circuit",
            [
                "LEDの長い足は、抵抗を通ってGPIO側につながっていますか？",
                "LEDとGPIOの間に抵抗は入っていますか？",
            ],
            ["いいえ、つながっていません", "抵抗を入れていません", "GNDが違う列かも", "まだできていません", "光りません"],
            {"gnd": False, "led": False, "resistor": False},
        ),
        scenario(
            "firmware_upload_success",
            "firmware_upload",
            "confirmed",
            "observe_serial",
            [
                "Arduino IDEなどで、ボード名とポートは選べていますか？",
                "確認用コードを書き込めましたか？",
            ],
            ["書き込めました", "Done uploading と出ました", "エラーなしで入りました", "ボードとポートを選べています", "アップロード成功しました"],
            {"firmware": True},
        ),
        scenario(
            "firmware_upload_problem",
            "firmware_upload",
            "problem_report",
            "debug_triage",
            [
                "Arduino IDEなどで、ボード名とポートは選べていますか？",
                "確認用コードを書き込めましたか？",
            ],
            ["COMポートが出ません", "書き込めません", "Failed to connect と出ます", "checkpoint loaded で止まります", "A fatal error occurred と出ます", "ボードが認識されません"],
            {"firmware": False},
            symptom="upload_failed",
        ),
        scenario(
            "serial_success",
            "observe_serial",
            "confirmed",
            "standard_build",
            [
                "シリアルモニタに start や sensor の値は表示されていますか？",
                "手を近づけると値は変わりますか？",
            ],
            ["start と出ました", "値が出ています", "手を近づけると値が変わります", "ログが見えています", "sensor の数字が動きます"],
            {"firmware": True, "serial": True},
        ),
        scenario(
            "serial_problem",
            "observe_serial",
            "problem_report",
            "debug_triage",
            [
                "シリアルモニタに start や sensor の値は表示されていますか？",
                "手を近づけると値は変わりますか？",
            ],
            ["何も出ません", "値が出ません", "真っ白です", "0のままです", "文字化けしています", "手を近づけても変わりません"],
            {"firmware": True, "serial": False},
            symptom="serial_problem",
        ),
        scenario(
            "inventory_report",
            "parts_check",
            "inventory_report",
            "minimal_circuit",
            ["手元にある部品を、分かる範囲で一行で書けますか？"],
            ["ESP32、LED、抵抗、ブレッドボードがあります", "ArduinoとLEDとジャンパ線があります", "センサー、LED、ボードがあります", "M5StackとGroveケーブルがあります"],
        ),
        scenario(
            "board_report",
            "parts_check",
            "board_report",
            "minimal_circuit",
            ["使うボードは Arduino、ESP32、M5Stack、Pico のどれですか？"],
            ["ESP32です", "Arduino Unoを使います", "Picoです", "M5Stackです", "micro:bitです"],
        ),
        scenario(
            "photo_request",
            "minimal_circuit",
            "photo_request",
            "minimal_circuit",
            ["配線が不安なら写真で確認できます。写真を撮れますか？"],
            ["写真で見てほしい", "画像を送ります", "配線写真を確認してください", "カメラで撮りました"],
        ),
    ]


def scenario(
    topic: str,
    stage: str,
    kind: str,
    next_stage: str,
    questions: list[str],
    answers: list[str],
    state: dict[str, bool] | None = None,
    symptom: str = "none",
) -> dict[str, Any]:
    return {
        "topic": topic,
        "stage": stage,
        "kind": kind,
        "nextStage": next_stage,
        "questions": questions,
        "answers": answers,
        "state": state or {},
        "symptom": symptom,
    }


def count(rows: list[dict[str, Any]], key: str) -> dict[str, int]:
    result: dict[str, int] = {}
    for row in rows:
        result[str(row[key])] = result.get(str(row[key]), 0) + 1
    return dict(sorted(result.items()))


if __name__ == "__main__":
    main()
