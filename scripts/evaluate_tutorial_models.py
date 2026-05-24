from __future__ import annotations

import argparse
import json
import sys
from dataclasses import dataclass, asdict
from pathlib import Path
from typing import Any


ROOT = Path(__file__).resolve().parents[1]
BACKEND = ROOT / "backend"
if str(BACKEND) not in sys.path:
    sys.path.insert(0, str(BACKEND))
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from ai_bridge import runtime_health, tutorial_answer_classifier_inference, tutorial_stage_classifier_inference
from server import tutorial_free_response


@dataclass(frozen=True)
class TutorialCase:
    name: str
    stage: str
    question: str
    answer: str
    expected_kind: str
    expected_stage: str
    state: dict[str, bool]


CASES = [
    TutorialCase(
        "minimal_led_confirmed",
        "minimal_circuit",
        "LEDの長い足は、抵抗を通ってGPIO側につながっていますか？",
        "はい、つながっています",
        "confirmed",
        "firmware_upload",
        {"inventory": True, "board": True, "gnd": True, "resistor": True},
    ),
    TutorialCase(
        "minimal_led_unknown",
        "minimal_circuit",
        "LEDの長い足は、抵抗を通ってGPIO側につながっていますか？",
        "わかりません",
        "unknown",
        "minimal_circuit",
        {"inventory": True, "board": True, "gnd": False, "resistor": False},
    ),
    TutorialCase(
        "minimal_led_negative",
        "minimal_circuit",
        "LEDの長い足は、抵抗を通ってGPIO側につながっていますか？",
        "抵抗が入っていません",
        "negative",
        "minimal_circuit",
        {"inventory": True, "board": True, "gnd": True, "resistor": False},
    ),
    TutorialCase(
        "firmware_com_missing",
        "firmware_upload",
        "Arduino IDEなどで、ボード名とポートは選べていますか？",
        "COMポートが出ません",
        "problem_report",
        "debug_triage",
        {"firmware": False},
    ),
    TutorialCase(
        "firmware_upload_success",
        "firmware_upload",
        "確認用コードを書き込めましたか？",
        "Done uploading と出ました",
        "confirmed",
        "observe_serial",
        {"firmware": True},
    ),
    TutorialCase(
        "serial_blank",
        "observe_serial",
        "シリアルモニタに start や sensor の値は表示されていますか？",
        "何も出ません",
        "problem_report",
        "debug_triage",
        {"firmware": True, "serial": False},
    ),
    TutorialCase(
        "serial_success",
        "observe_serial",
        "シリアルモニタに start や sensor の値は表示されていますか？",
        "start と値が出ています",
        "confirmed",
        "standard_build",
        {"firmware": True, "serial": True},
    ),
    TutorialCase(
        "inventory_report",
        "parts_check",
        "手元にある部品を、分かる範囲で一行で書けますか？",
        "ESP32、LED、抵抗、ブレッドボードがあります",
        "inventory_report",
        "minimal_circuit",
        {"inventory": True},
    ),
    TutorialCase(
        "board_report",
        "parts_check",
        "使うボードは Arduino、ESP32、M5Stack、Pico のどれですか？",
        "ESP32です",
        "board_report",
        "minimal_circuit",
        {"board": True},
    ),
    TutorialCase(
        "photo_request",
        "minimal_circuit",
        "配線が不安なら写真で確認できます。写真を撮れますか？",
        "写真で見てほしいです",
        "photo_request",
        "debug_triage",
        {"inventory": True, "board": True},
    ),
]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Evaluate tutorial classifiers and backend progress contract.")
    parser.add_argument("--json", action="store_true", help="Print machine-readable JSON only.")
    parser.add_argument("--sync-neural", action="store_true", help="Ask backend to use synchronous neural path.")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    rows: list[dict[str, Any]] = []
    for case in CASES:
        payload = {
            "projectId": "desk_pet",
            "currentStage": case.stage,
            "question": case.question,
            "answer": case.answer,
            "tutorialState": case.state,
        }
        if args.sync_neural:
            payload["syncNeural"] = True
        answer_model = tutorial_answer_classifier_inference(payload)
        stage_model = tutorial_stage_classifier_inference({
            **payload,
            "interpretedKind": answer_model.get("kind") or "",
            "fallbackStage": case.expected_stage,
        })
        backend = tutorial_free_response(payload)
        row = {
            **asdict(case),
            "answerModel": compact_answer(answer_model),
            "stageModel": compact_stage(stage_model),
            "backend": {
                "mode": backend.get("mode"),
                "kind": backend.get("interpreted"),
                "interpretedBy": backend.get("interpretedBy"),
                "stage": (backend.get("progress") or {}).get("toStage") or backend.get("nextStage"),
                "question": (backend.get("uiPatch") or {}).get("nextQuestion"),
            },
        }
        row["backendOk"] = row["backend"]["kind"] == case.expected_kind and row["backend"]["stage"] == case.expected_stage
        row["answerModelOk"] = row["answerModel"]["kind"] == case.expected_kind
        row["stageModelOk"] = row["stageModel"]["stage"] == case.expected_stage
        rows.append(row)

    summary = {
        "cases": len(rows),
        "backend_passed": sum(1 for row in rows if row["backendOk"]),
        "answer_model_passed": sum(1 for row in rows if row["answerModelOk"]),
        "stage_model_passed": sum(1 for row in rows if row["stageModelOk"]),
        "health": runtime_health(),
        "rows": rows,
    }
    if args.json:
        print(json.dumps(summary, ensure_ascii=False, indent=2))
    else:
        print(f"backend: {summary['backend_passed']}/{summary['cases']}")
        print(f"answer model: {summary['answer_model_passed']}/{summary['cases']}")
        print(f"stage model: {summary['stage_model_passed']}/{summary['cases']}")
        for row in rows:
            mark = "OK" if row["backendOk"] else "NG"
            print(
                f"{mark} {row['name']}: backend=({row['backend']['kind']}, {row['backend']['stage']}) "
                f"answerModel=({row['answerModel']['kind']}, {row['answerModel']['confidence']}) "
                f"stageModel=({row['stageModel']['stage']}, {row['stageModel']['confidence']})"
            )
    raise SystemExit(0 if summary["backend_passed"] == summary["cases"] else 1)


def compact_answer(result: dict[str, Any]) -> dict[str, Any]:
    ranked = result.get("ranked") if isinstance(result.get("ranked"), list) else []
    return {
        "available": bool(result.get("available")),
        "kind": result.get("kind"),
        "confidence": result.get("confidence"),
        "modelDir": result.get("modelDir"),
        "top3": ranked[:3],
    }


def compact_stage(result: dict[str, Any]) -> dict[str, Any]:
    ranked = result.get("ranked") if isinstance(result.get("ranked"), list) else []
    return {
        "available": bool(result.get("available")),
        "stage": result.get("stage"),
        "confidence": result.get("confidence"),
        "modelDir": result.get("modelDir"),
        "top3": ranked[:3],
    }


if __name__ == "__main__":
    main()
