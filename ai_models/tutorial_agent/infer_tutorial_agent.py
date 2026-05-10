from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

import torch

from ai_models.makergraph.tokenizer import MakerTokenizer

from .data import build_input_text
from .model import load_checkpoint
from .taxonomy import (
    ACTION_LABELS,
    ACTIONS,
    CHECKPOINT_LABELS,
    CHECKPOINTS,
    CONCEPT_LABELS,
    CONCEPTS,
    PROJECT_IDS,
    PROJECT_TITLES,
    QUESTION_TEXT,
    QUESTIONS,
    ROUTE_LABELS,
    ROUTES,
    STAGE_LABELS,
    TUTORIAL_STAGES,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run neural autonomous tutorial agent.")
    parser.add_argument("--model-dir", default="runs/tutorial_agent")
    parser.add_argument("--text", default="予算5000円でLチカの次に進みたい。ESP32とLEDがあります。")
    parser.add_argument("--project-id", default="desk_pet")
    parser.add_argument("--stage", default="orient")
    parser.add_argument("--symptom", default="none")
    parser.add_argument("--inventory", default="ESP32 LED resistor breadboard jumper wires USB")
    parser.add_argument("--budget", type=int, default=5000)
    parser.add_argument("--board", default="esp32")
    parser.add_argument("--device", default="auto")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    service = TutorialAgentInference(args.model_dir, args.device)
    print(json.dumps(service.predict({
        "text": args.text,
        "projectId": args.project_id,
        "currentStage": args.stage,
        "symptom": args.symptom,
        "inventory": args.inventory,
        "budget": args.budget,
        "board": args.board,
    }), ensure_ascii=False, indent=2))


class TutorialAgentInference:
    def __init__(self, model_dir: str | Path, device: str = "auto") -> None:
        self.model_dir = Path(model_dir)
        selected_device = "cuda" if device == "auto" and torch.cuda.is_available() else device
        if selected_device == "auto":
            selected_device = "cpu"
        self.device = torch.device(selected_device)
        self.tokenizer = MakerTokenizer.load(self.model_dir / "tokenizer.json")
        self.model, self.config, self.payload = load_checkpoint(str(self.model_dir / "best.pt"), map_location=self.device)
        self.model.to(self.device)

    @torch.no_grad()
    def predict(self, payload: dict[str, Any]) -> dict[str, Any]:
        text = build_input_text(normalize_payload(payload))
        input_ids, mask = self.tokenizer.encode(text, self.config.max_length, "<tutorial>")
        ids = torch.tensor([input_ids], dtype=torch.long, device=self.device)
        attention = torch.tensor([mask], dtype=torch.bool, device=self.device)
        outputs = self.model(ids, attention)
        project = top_class(outputs["project_logits"][0], PROJECT_IDS)
        stage = top_class(outputs["stage_logits"][0], TUTORIAL_STAGES)
        action = top_class(outputs["action_logits"][0], ACTIONS)
        question = top_class(outputs["question_logits"][0], QUESTIONS)
        checkpoint = top_class(outputs["checkpoint_logits"][0], CHECKPOINTS)
        route = top_class(outputs["route_logits"][0], ROUTES)
        concept_probs = outputs["concept_logits"][0].sigmoid().detach().cpu()
        concepts = [
            {"id": CONCEPTS[index], "label": CONCEPT_LABELS[CONCEPTS[index]], "score": round(float(concept_probs[index]), 4)}
            for index in concept_probs.argsort(descending=True).tolist()[:5]
        ]
        autonomy_score = round(float(outputs["autonomy_logits"][0].sigmoid().detach().cpu()), 4)
        rendered = render_tutorial(project["id"], stage["id"], action["id"], question["id"], checkpoint["id"], route["id"], concepts, autonomy_score)
        return {
            "available": True,
            "model": "TutorialAgentTransformer",
            "device": str(self.device),
            "inputPreview": text[:500],
            "project": {**project, "title": PROJECT_TITLES[project["id"]]},
            "stage": {**stage, "label": STAGE_LABELS[stage["id"]]},
            "action": {**action, "label": ACTION_LABELS[action["id"]]},
            "question": {**question, "text": QUESTION_TEXT[question["id"]]},
            "checkpoint": {**checkpoint, "label": CHECKPOINT_LABELS[checkpoint["id"]]},
            "route": {**route, "label": ROUTE_LABELS[route["id"]]},
            "concepts": concepts,
            "autonomyScore": autonomy_score,
            "tutorial": rendered,
            "metrics": self.payload.get("metrics", {}),
        }


def normalize_payload(payload: dict[str, Any]) -> dict[str, Any]:
    normalized = dict(payload)
    text = str(normalized.get("text") or normalized.get("message") or "")
    lowered = text.lower()
    if not normalized.get("symptom"):
        if "光らない" in text or "led not" in lowered:
            normalized["symptom"] = "led_not_lighting"
        elif "ポート" in text or "書き込め" in text or "upload" in lowered:
            normalized["symptom"] = "upload_failed"
        elif "センサー" in text or "sensor" in lowered:
            normalized["symptom"] = "sensor_static"
        elif "checkpoint loaded" in lowered or "止ま" in text:
            normalized["symptom"] = "app_stuck_after_checkpoint"
        else:
            normalized["symptom"] = "none"
    if not normalized.get("currentStage") and not normalized.get("stage"):
        normalized["currentStage"] = "debug_triage" if normalized.get("symptom") != "none" else "orient"
    if not normalized.get("projectId"):
        normalized["projectId"] = detect_project(text)
    return normalized


def detect_project(text: str) -> str:
    lowered = text.lower()
    if "植物" in text or "水やり" in text or "plant" in lowered:
        return "plant_ping"
    if "姿勢" in text or "posture" in lowered:
        return "posture_guard"
    if "温度" in text or "顔" in text or "temperature" in lowered:
        return "temp_face"
    if "光" in text or "お守り" in text or "led" in lowered:
        return "light_charm"
    return "desk_pet"


def top_class(logits: torch.Tensor, labels: list[str]) -> dict[str, Any]:
    probs = logits.softmax(dim=-1).detach().cpu()
    index = int(probs.argmax())
    return {"id": labels[index], "confidence": round(float(probs[index]), 4)}


def render_tutorial(
    project_id: str,
    stage: str,
    action: str,
    question: str,
    checkpoint: str,
    route: str,
    concepts: list[dict[str, Any]],
    autonomy_score: float,
) -> dict[str, Any]:
    project_title = PROJECT_TITLES[project_id]
    stage_order = TUTORIAL_STAGES
    start_index = stage_order.index(stage)
    runbook = []
    for offset, next_stage in enumerate(stage_order[start_index:start_index + 4], start=1):
        runbook.append({
            "step": offset,
            "stage": next_stage,
            "title": STAGE_LABELS[next_stage],
            "goal": stage_goal(next_stage, project_title),
        })
    return {
        "title": f"{project_title}：{STAGE_LABELS[stage]}",
        "mode": "autonomous_neural_tutorial",
        "route": ROUTE_LABELS[route],
        "checkpoint": CHECKPOINT_LABELS[checkpoint],
        "nextAction": ACTION_LABELS[action],
        "nextQuestion": QUESTION_TEXT[question],
        "teacherNote": teacher_note(stage, action, autonomy_score),
        "checklist": checklist_for(stage),
        "expectedSignal": expected_signal(stage),
        "conceptsToTeach": [concept["label"] for concept in concepts[:4]],
        "threeChoices": three_choices(project_id),
        "runbook": runbook,
        "stopRules": [
            "AC100Vや高電圧は扱わない",
            "USBを挿したまま配線を差し替えない",
            "LEDには抵抗を入れる",
            "モーターやLiPoは別途安全確認が済むまで使わない",
        ],
    }


def stage_goal(stage: str, project_title: str) -> str:
    goals = {
        "orient": f"{project_title}を最初の一作として選べる状態にする",
        "parts_check": "買うもの、家にあるもの、後回しにできるものを分ける",
        "minimal_circuit": "LEDまたはセンサー一つだけで動作を確認する",
        "firmware_upload": "シリアル出力入りの確認コードを書き込む",
        "observe_serial": "値が変わるか、起動メッセージが出るかを見る",
        "debug_triage": "物理、電源、コードの順に一つだけ原因を潰す",
        "standard_build": "ブザー、センサー、外装などを足して作品に近づける",
        "enclosure": "紙箱やケースに固定して見せられる形にする",
        "extension": "見た目、音、ログのどれか一つを追加する",
        "completion_log": "完成写真、詰まった点、次に使う部品を保存する",
    }
    return goals[stage]


def teacher_note(stage: str, action: str, autonomy_score: float) -> str:
    if stage == "debug_triage":
        return "今は説明を増やすより、確認を一つに絞る段階です。答えに応じて次の分岐へ進みます。"
    if action == "show_three_choices":
        return "初心者が止まらないように、一番おすすめ・安い案・少し挑戦案の3つだけに絞ります。"
    if autonomy_score >= 0.7:
        return "かなり自走できます。次の作業カードまで進めて、詰まったら写真かログを受け取ります。"
    return "まだ不確実なので、手順を短く区切って確認しながら進めます。"


def checklist_for(stage: str) -> list[str]:
    table = {
        "orient": ["予算を決める", "使うボードを一つに決める", "最小版から始める"],
        "parts_check": ["マイコン", "USBデータケーブル", "ブレッドボード", "ジャンパ線", "抵抗を確認"],
        "minimal_circuit": ["USBを抜く", "GNDを共通にする", "LEDの向きと抵抗を確認", "コードのピン番号と合わせる"],
        "firmware_upload": ["ボードを選ぶ", "ポートを選ぶ", "Serial.beginを入れる", "起動メッセージを出す"],
        "observe_serial": ["シリアルモニタを開く", "速度を合わせる", "値が変化するか見る"],
        "debug_triage": ["物理接続を見る", "電源を見る", "コードのピンを見る", "ログを一つ取る"],
        "standard_build": ["最小版が動くことを確認", "出力部品を一つだけ足す", "電源容量を見る"],
        "enclosure": ["金属ケースを避ける", "配線を引っ張らない", "USB端子をふさがない"],
        "extension": ["追加は一機能だけ", "変更前コードを残す", "動いた状態をログに残す"],
        "completion_log": ["完成写真", "使った部品", "詰まった原因", "次に作る候補を保存"],
    }
    return table[stage]


def expected_signal(stage: str) -> str:
    table = {
        "orient": "候補が3つに絞られている",
        "parts_check": "不足部品と後回し部品が分かる",
        "minimal_circuit": "LEDが点く、またはセンサー値が読める",
        "firmware_upload": "シリアルモニタに start と表示される",
        "observe_serial": "手を近づける、暗くするなどで値が変わる",
        "debug_triage": "次に確認する原因が一つに絞られる",
        "standard_build": "入力に対してLEDやブザーが反応する",
        "enclosure": "机に置いても配線が抜けにくい",
        "extension": "追加機能だけを切り戻せる",
        "completion_log": "次作品推薦に使えるログが残る",
    }
    return table[stage]


def three_choices(project_id: str) -> list[dict[str, str]]:
    related = {
        "light_charm": ["暗くなると光る小さなお守り", "LEDだけのお守り", "音に反応するライト"],
        "desk_pet": ["近づくと鳴く机上ペット", "近づくと光るだけ版", "サーボで揺れる机上ペット"],
        "plant_ping": ["水やり通知", "乾いたらLEDだけ版", "温湿度も見る植物モニター"],
        "posture_guard": ["姿勢注意デバイス", "近すぎ通知だけ版", "ログ付き集中サポート"],
        "temp_face": ["温度で表情が変わるミニキャラ", "温度を表示するだけ版", "表情とログ付き環境キャラ"],
    }[project_id]
    return [
        {"type": "一番おすすめ", "title": related[0]},
        {"type": "安い案", "title": related[1]},
        {"type": "少し挑戦案", "title": related[2]},
    ]


if __name__ == "__main__":
    main()
