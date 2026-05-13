from __future__ import annotations

import json
import random
from pathlib import Path
from typing import Any

import torch
from torch.utils.data import Dataset

from ai_models.makergraph.tokenizer import MakerTokenizer

from .taxonomy import ANSWER_PATTERNS, PROJECTS, QUESTIONS, STAGES


def generate_records(samples: int, seed: int = 131) -> list[dict[str, Any]]:
    rng = random.Random(seed)
    records: list[dict[str, Any]] = []
    kinds = list(ANSWER_PATTERNS)
    for _ in range(samples):
        project_id = rng.choice(list(PROJECTS))
        current_stage = rng.choice(STAGES[:-1])
        question = question_for_stage(current_stage, rng)
        kind = rng.choices(kinds, weights=[18, 14, 14, 8, 12, 12], k=1)[0]
        answer = mutate_answer(rng.choice(ANSWER_PATTERNS[kind]), rng)
        if "部品" in question or "手元" in question:
            kind = "inventory_report" if rng.random() < 0.7 else kind
            answer = mutate_answer(rng.choice(ANSWER_PATTERNS[kind]), rng)
        if "GND" in question and rng.random() < 0.6:
            kind = rng.choice(["confirmed", "negative", "unknown", "photo_request"])
            answer = mutate_answer(rng.choice(ANSWER_PATTERNS[kind]), rng)
        if "LED" in question and rng.random() < 0.55:
            kind = rng.choice(["confirmed", "negative", "unknown"])
            answer = mutate_answer(rng.choice(ANSWER_PATTERNS[kind]), rng)

        next_stage = choose_next_stage(question, current_stage, kind)
        target = build_target(project_id, question, answer, current_stage, next_stage, kind, rng)
        source = build_source({
            "projectId": project_id,
            "projectTitle": PROJECTS[project_id],
            "currentStage": current_stage,
            "question": question,
            "answer": answer,
            "inventory": rng.choice([
                "ESP32, LED, 220Ω抵抗, ブレッドボード, ジャンパ線, USBケーブル",
                "Arduino Uno, LED, 抵抗, USBケーブル",
                "不明。スターター構成で進みたい",
                "M5Stack, Groveケーブル, LED",
            ]),
            "budget": rng.choice([1000, 3000, 5000, 10000]),
            "symptom": symptom_for_kind(kind, answer),
        })
        records.append({"source": source, "target": target, "nextStage": next_stage, "kind": target_kind(target)})
    return records


def question_for_stage(stage: str, rng: random.Random) -> str:
    table = {
        "orient": QUESTIONS[0],
        "parts_check": QUESTIONS[2],
        "minimal_circuit": rng.choice([QUESTIONS[4], QUESTIONS[5], QUESTIONS[6]]),
        "firmware_upload": rng.choice([QUESTIONS[3], QUESTIONS[7]]),
        "observe_serial": QUESTIONS[8],
        "debug_triage": rng.choice([QUESTIONS[4], QUESTIONS[5], QUESTIONS[6], QUESTIONS[7], QUESTIONS[8]]),
        "standard_build": QUESTIONS[6],
        "enclosure": QUESTIONS[9],
        "extension": QUESTIONS[9],
    }
    return table.get(stage, QUESTIONS[2])


def mutate_answer(answer: str, rng: random.Random) -> str:
    prefix = rng.choice(["", "", "今の状態は、", "たぶん", "確認したら、"])
    suffix = rng.choice(["", "", "。", "、次に何をすればいいですか？", "。ちょっと不安です。"])
    return f"{prefix}{answer}{suffix}"


def build_source(payload: dict[str, Any]) -> str:
    return "\n".join([
        f"project: {payload.get('projectId')} {payload.get('projectTitle')}",
        f"stage: {payload.get('currentStage')}",
        f"budget: {payload.get('budget')}",
        f"inventory: {payload.get('inventory')}",
        f"symptom: {payload.get('symptom')}",
        f"question: {payload.get('question')}",
        f"answer: {payload.get('answer')}",
        "task: generate empathetic autonomous tutorial response, next instruction, and next stage",
    ])


def build_target(project_id: str, question: str, answer: str, current_stage: str, next_stage: str, kind: str, rng: random.Random) -> str:
    if "GND" in question:
        if kind == "confirmed":
            title = "GND共有はクリアです"
            body = "LED・センサー・ボードのGNDが同じ基準になっています。次は確認コードを書き込む準備へ進めます。"
            instruction = "ボードとポートを確認して、シリアル出力つきコードを書き込みます。"
            style = "good"
        elif kind == "photo_request":
            title = "写真でGNDを確認しましょう"
            body = "ボードのGNDピン、ブレッドボードのGND列、LEDやセンサーのマイナス側が見えるように撮ると判断できます。"
            instruction = "USBを抜いてから真上の写真を撮ってください。"
            style = "warn"
        else:
            title = "GNDを同じ列につなぎ直します"
            body = "USBを抜いて、LEDのマイナス側、センサーのGND、ボードのGNDを同じGND列へまとめます。"
            instruction = "つなぎ直せたら「つなぎ直した」と答えてください。"
            style = "warn"
    elif "部品" in question or "手元" in question:
        if kind in {"unknown", "photo_request"}:
            title = "部品が分からなくても進めます"
            body = "スターター構成として扱い、必要部品込みで見積もります。ここで止まらず最小回路へ進みます。"
            instruction = "LED・抵抗・GNDだけの最小確認から始めます。"
            style = "warn"
        else:
            title = "部品情報を受け取りました"
            body = "書いてくれた部品を所持品として扱い、不足分は部品表で分けて出します。"
            instruction = "次はUSBを抜いた状態で最小回路を確認します。"
            style = "good"
    elif "ボード" in question or "Arduino" in question or "ESP32" in question:
        title = "ボード情報を反映しました"
        body = "このボードに合わせてピン番号と確認コードを組み立てます。"
        instruction = "次は最小回路とコードのピン番号を合わせます。"
        style = "good"
    elif "LED" in question:
        if kind == "confirmed":
            title = "LEDの向きは大丈夫そうです"
            body = "長い足が抵抗を通ってGPIO側、短い足がGND側なら次へ進めます。"
            instruction = "抵抗とコードのピン番号を確認します。"
            style = "good"
        else:
            title = "LEDの向きを直してから進みます"
            body = "LEDは向きがあります。長い足をGPIO側、短い足をGND側にします。"
            instruction = "USBを抜いて向きを直し、LEDだけで点灯確認します。"
            style = "warn"
    elif "抵抗" in question:
        if kind == "confirmed":
            title = "抵抗は入っています"
            body = "LEDをGPIOへ直結していないので、まず安全な形です。"
            instruction = "コードのGPIO番号と配線先を一致させます。"
            style = "good"
        else:
            title = "抵抗を直列に入れます"
            body = "LEDとGPIOの間に220Ω前後の抵抗を入れてください。抵抗なしの直結は避けます。"
            instruction = "抵抗を足したらLEDだけで再確認します。"
            style = "warn"
    elif "シリアル" in question or "値" in question:
        if kind == "confirmed":
            title = "ログが見えています"
            body = "起動メッセージや値が見えているので、コードは最低限動いています。"
            instruction = "入力に応じて値が変わるか観察します。"
            style = "good"
        else:
            title = "ログを見る準備をします"
            body = "シリアルモニタを開き、速度をSerial.beginの値に合わせます。"
            instruction = "起動メッセージが出るかだけ確認します。"
            style = "warn"
    elif kind == "problem_report":
        title = "動かない状態として受け取りました"
        body = "原因を広げず、物理接続、電源、コードの順に一つずつ潰します。"
        instruction = "まずUSBを抜いてGND共有とピン番号を確認します。"
        style = "warn"
    else:
        title = "回答を受け取りました"
        body = "今の回答を次の作業に反映します。迷わないように確認点を一つに絞ります。"
        instruction = "次のカードで短い作業だけを進めます。"
        style = "good" if kind == "confirmed" else "info"

    if rng.random() < 0.18:
        body += " テスターがなくても、目視確認から進めます。"
    return f"style:{style}\ntitle:{title}\nbody:{body}\nnext:{instruction}\nstage:{next_stage}"


def choose_next_stage(question: str, current_stage: str, kind: str) -> str:
    if "部品" in question or "手元" in question:
        return "minimal_circuit"
    if "GND" in question:
        return "firmware_upload" if kind == "confirmed" else "minimal_circuit"
    if "LED" in question or "抵抗" in question:
        return "firmware_upload" if kind == "confirmed" else "minimal_circuit"
    if "ボード" in question or "ポート" in question:
        return "observe_serial" if kind == "confirmed" else "debug_triage"
    if "シリアル" in question or "値" in question:
        return "standard_build" if kind == "confirmed" else "debug_triage"
    if kind in {"problem_report", "photo_request"}:
        return "debug_triage"
    order = STAGES
    if kind == "confirmed" and current_stage in order and order.index(current_stage) < len(order) - 1:
        return order[order.index(current_stage) + 1]
    return current_stage


def symptom_for_kind(kind: str, answer: str) -> str:
    if kind != "problem_report":
        return "none"
    lowered = answer.lower()
    if "led" in lowered or "光" in answer:
        return "led_not_lighting"
    if "upload" in lowered or "ポート" in answer or "書き込" in answer:
        return "upload_failed"
    if "値" in answer or "sensor" in lowered:
        return "sensor_static"
    return "unknown"


def target_kind(target: str) -> str:
    for line in target.splitlines():
        if line.startswith("style:"):
            return line.split(":", 1)[1].strip()
    return "info"


def collect_texts(records: list[dict[str, Any]]) -> list[str]:
    return [item["source"] for item in records] + [item["target"] for item in records]


def write_jsonl(path: str | Path, records: list[dict[str, Any]]) -> None:
    target = Path(path)
    target.parent.mkdir(parents=True, exist_ok=True)
    with target.open("w", encoding="utf-8") as handle:
        for record in records:
            handle.write(json.dumps(record, ensure_ascii=False) + "\n")


def load_jsonl(path: str | Path) -> list[dict[str, Any]]:
    with Path(path).open(encoding="utf-8") as handle:
        return [json.loads(line) for line in handle if line.strip()]


class TutorialResponseDataset(Dataset):
    def __init__(self, records: list[dict[str, Any]], tokenizer: MakerTokenizer, max_source_length: int, max_target_length: int) -> None:
        self.records = records
        self.tokenizer = tokenizer
        self.max_source_length = max_source_length
        self.max_target_length = max_target_length

    def __len__(self) -> int:
        return len(self.records)

    def __getitem__(self, index: int) -> dict[str, torch.Tensor]:
        record = self.records[index]
        source_ids, source_mask = self.tokenizer.encode(record["source"], self.max_source_length, "<query>")
        target_ids, target_mask = self.tokenizer.encode(record["target"], self.max_target_length, "<log>")
        decoder_input = target_ids[:-1]
        labels = target_ids[1:]
        label_mask = target_mask[1:]
        return {
            "source_ids": torch.tensor(source_ids, dtype=torch.long),
            "source_mask": torch.tensor(source_mask, dtype=torch.bool),
            "decoder_input_ids": torch.tensor(decoder_input, dtype=torch.long),
            "labels": torch.tensor(labels, dtype=torch.long),
            "label_mask": torch.tensor(label_mask, dtype=torch.bool),
        }


def collate_response_batch(items: list[dict[str, torch.Tensor]]) -> dict[str, torch.Tensor]:
    return {key: torch.stack([item[key] for item in items]) for key in items[0]}

