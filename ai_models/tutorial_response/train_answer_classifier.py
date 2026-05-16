from __future__ import annotations

import argparse
import json
import random
import time
from pathlib import Path
from typing import Any

import torch
from torch import nn
from torch.utils.data import DataLoader, Dataset, random_split

from ai_models.makergraph.tokenizer import MakerTokenizer

from .answer_classifier import ANSWER_KIND_LABELS, TutorialAnswerClassifier, TutorialAnswerClassifierConfig, count_parameters
from .data import build_source, collect_texts, load_jsonl, write_jsonl
from .taxonomy import PROJECTS


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Train neural classifier for tutorial free-answer intent.")
    parser.add_argument("--data", default="data/tutorial_answer_classifier_training.jsonl")
    parser.add_argument("--extra-data", action="append", default=[], help="Additional classifier JSONL records exported from real logs.")
    parser.add_argument("--extra-weight", type=int, default=3)
    parser.add_argument("--positive-feedback-weight", type=int, default=3)
    parser.add_argument("--fix-feedback-weight", type=int, default=6)
    parser.add_argument("--keep-negative-extra", action="store_true")
    parser.add_argument("--output", default="runs/tutorial_answer_classifier")
    parser.add_argument("--samples", type=int, default=70000)
    parser.add_argument("--robust-ratio", type=float, default=0.85)
    parser.add_argument("--regenerate", action="store_true")
    parser.add_argument("--epochs", type=int, default=5)
    parser.add_argument("--batch-size", type=int, default=96)
    parser.add_argument("--lr", type=float, default=4e-4)
    parser.add_argument("--weight-decay", type=float, default=0.01)
    parser.add_argument("--max-source-length", type=int, default=320)
    parser.add_argument("--d-model", type=int, default=160)
    parser.add_argument("--n-heads", type=int, default=5)
    parser.add_argument("--n-layers", type=int, default=3)
    parser.add_argument("--seed", type=int, default=271)
    parser.add_argument("--device", default="auto", choices=["auto", "cpu", "cuda"])
    parser.add_argument("--amp", action="store_true")
    return parser.parse_args()


def main() -> None:
    started = time.perf_counter()
    args = parse_args()
    random.seed(args.seed)
    torch.manual_seed(args.seed)
    data_path = Path(args.data)
    if args.regenerate or not data_path.exists():
        records = make_classifier_records(args.samples, seed=args.seed, robust_ratio=args.robust_ratio)
        write_jsonl(data_path, records)
    records = load_jsonl(data_path)
    records.extend(load_extra_records(
        args.extra_data,
        base_weight=max(1, args.extra_weight),
        positive_weight=max(1, args.positive_feedback_weight),
        fix_weight=max(1, args.fix_feedback_weight),
        keep_negative=args.keep_negative_extra,
    ))
    tokenizer = MakerTokenizer(domain_terms=[])
    tokenizer.build_vocab(collect_texts(records), min_freq=1, max_vocab_size=22000)
    config = TutorialAnswerClassifierConfig(
        vocab_size=len(tokenizer.vocab),
        max_source_length=args.max_source_length,
        d_model=args.d_model,
        n_heads=args.n_heads,
        n_layers=args.n_layers,
    )
    dataset = TutorialAnswerDataset(records, tokenizer, config.max_source_length)
    train_size = int(len(dataset) * 0.9)
    valid_size = len(dataset) - train_size
    train_dataset, valid_dataset = random_split(dataset, [train_size, valid_size], generator=torch.Generator().manual_seed(args.seed))
    device = select_device(args.device)
    train_loader = DataLoader(train_dataset, batch_size=args.batch_size, shuffle=True, collate_fn=collate_batch)
    valid_loader = DataLoader(valid_dataset, batch_size=args.batch_size, shuffle=False, collate_fn=collate_batch)
    model = TutorialAnswerClassifier(config).to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)
    scaler = torch.cuda.amp.GradScaler(enabled=args.amp and device.type == "cuda")
    output_dir = Path(args.output)
    output_dir.mkdir(parents=True, exist_ok=True)
    tokenizer.save(output_dir / "tokenizer.json")
    config.save(output_dir / "config.json")
    print(json.dumps({
        "event": "start",
        "device": str(device),
        "cuda_available": torch.cuda.is_available(),
        "cuda_name": torch.cuda.get_device_name(0) if torch.cuda.is_available() else None,
        "records": len(records),
        "vocab_size": len(tokenizer.vocab),
        "parameters": count_parameters(model),
    }, ensure_ascii=False, indent=2), flush=True)

    best_valid = float("inf")
    history: list[dict[str, float]] = []
    for epoch in range(1, args.epochs + 1):
        train_metrics = run_epoch(model, train_loader, optimizer, scaler, device, train=True, amp=args.amp)
        valid_metrics = run_epoch(model, valid_loader, optimizer, scaler, device, train=False, amp=False)
        row = {"epoch": epoch, **prefix("train", train_metrics), **prefix("valid", valid_metrics)}
        history.append(row)
        print(json.dumps({"event": "epoch_complete", **row, "elapsed_sec": round(time.perf_counter() - started, 2)}, ensure_ascii=False), flush=True)
        if valid_metrics["loss"] <= best_valid:
            best_valid = valid_metrics["loss"]
            save_checkpoint(output_dir / "best.pt", model, config, epoch, valid_metrics)
    save_checkpoint(output_dir / "last.pt", model, config, args.epochs, history[-1])
    (output_dir / "metrics.json").write_text(json.dumps(history, ensure_ascii=False, indent=2), encoding="utf-8")


class TutorialAnswerDataset(Dataset):
    def __init__(self, records: list[dict[str, Any]], tokenizer: MakerTokenizer, max_source_length: int) -> None:
        self.records = records
        self.tokenizer = tokenizer
        self.max_source_length = max_source_length

    def __len__(self) -> int:
        return len(self.records)

    def __getitem__(self, index: int) -> dict[str, torch.Tensor]:
        record = self.records[index]
        source_ids, source_mask = self.tokenizer.encode(record["source"], self.max_source_length, "<query>")
        return {
            "source_ids": torch.tensor(source_ids, dtype=torch.long),
            "source_mask": torch.tensor(source_mask, dtype=torch.bool),
            "label": torch.tensor(ANSWER_KIND_LABELS.index(record["kind"]), dtype=torch.long),
        }


def collate_batch(items: list[dict[str, torch.Tensor]]) -> dict[str, torch.Tensor]:
    return {key: torch.stack([item[key] for item in items]) for key in items[0]}


def run_epoch(
    model: TutorialAnswerClassifier,
    loader: DataLoader,
    optimizer: torch.optim.Optimizer,
    scaler: torch.cuda.amp.GradScaler,
    device: torch.device,
    train: bool,
    amp: bool,
) -> dict[str, float]:
    model.train(train)
    total_loss = 0.0
    total_items = 0
    total_correct = 0
    for batch in loader:
        source_ids = batch["source_ids"].to(device)
        source_mask = batch["source_mask"].to(device)
        labels = batch["label"].to(device)
        if train:
            optimizer.zero_grad(set_to_none=True)
        with torch.set_grad_enabled(train), torch.cuda.amp.autocast(enabled=amp and device.type == "cuda"):
            logits = model(source_ids, source_mask)
            loss = nn.functional.cross_entropy(logits, labels)
        if train:
            scaler.scale(loss).backward()
            scaler.unscale_(optimizer)
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            scaler.step(optimizer)
            scaler.update()
        total_loss += float(loss.detach().cpu()) * labels.numel()
        total_items += labels.numel()
        total_correct += int((logits.argmax(dim=-1) == labels).sum().detach().cpu())
    return {"loss": total_loss / max(1, total_items), "accuracy": total_correct / max(1, total_items)}


def make_classifier_records(samples: int, seed: int, robust_ratio: float) -> list[dict[str, Any]]:
    rng = random.Random(seed)
    records = make_handwritten_variation_records(rng, samples)
    rng.shuffle(records)
    return records


def load_extra_records(
    paths: list[str],
    base_weight: int,
    positive_weight: int,
    fix_weight: int,
    keep_negative: bool,
) -> list[dict[str, Any]]:
    records: list[dict[str, Any]] = []
    seen: set[tuple[str, str]] = set()
    counts = {"up": 0, "down": 0, "fix": 0, "none": 0}
    for raw_path in paths:
        path = Path(raw_path)
        if not path.exists():
            print(json.dumps({"event": "classifier_extra_data_missing", "path": str(path)}, ensure_ascii=False), flush=True)
            continue
        for record in load_jsonl(path):
            source = str(record.get("source") or "")
            kind = str(record.get("kind") or record.get("target") or "")
            if not source or kind not in ANSWER_KIND_LABELS:
                continue
            feedback = str(record.get("feedback") or "none")
            if feedback == "down" and not keep_negative:
                counts["down"] += 1
                continue
            key = (source, kind)
            if key in seen:
                continue
            seen.add(key)
            counts[feedback if feedback in counts else "none"] += 1
            repeat = base_weight
            if feedback == "up":
                repeat *= positive_weight
            elif feedback == "fix":
                repeat *= fix_weight
            for _ in range(repeat):
                records.append({"source": source, "kind": kind, "target": kind})
        print(json.dumps({
            "event": "classifier_extra_data_loaded",
            "path": str(path),
            "unique_records": len(seen),
            "weighted_records": len(records),
            "feedback": counts,
        }, ensure_ascii=False), flush=True)
    return records


def make_handwritten_variation_records(rng: random.Random, samples: int) -> list[dict[str, str]]:
    examples = {
        "confirmed": [
            "はい", "できた", "つながっています", "同じGNDです", "抵抗を通っています", "LED光りました",
            "yes", "ok", "done", "connected", "same ground", "it goes through the resistor",
        ],
        "negative": [
            "いいえ", "まだです", "つながってません", "抵抗ないです", "光りません", "違うかも",
            "no", "not yet", "not connected", "does not light", "missing resistor",
        ],
        "unknown": [
            "分からない", "自信ない", "どこを見るの", "たぶん", "何を書けばいいか分からない",
            "i don't know", "not sure", "maybe", "unclear",
        ],
        "photo_request": [
            "写真で見て", "画像送ります", "配線写真を確認して", "写真で判断したい",
            "check my photo", "i will upload image", "look at the wiring photo",
        ],
        "problem_report": [
            "エラーが出た", "動かない", "checkpoint loadedで止まる", "COMポートが出ない", "書き込めない",
            "error", "stuck", "cannot upload", "COM port missing", "checkpoint loaded",
        ],
        "inventory_report": [
            "ESP32 LED 抵抗 ブレッドボード", "Arduinoとセンサーがあります", "ジャンパ線とLEDがあります",
            "I have ESP32 LED resistor breadboard", "parts are servo sensor wires",
        ],
        "board_report": [
            "ESP32です", "Arduino Unoです", "M5Stackを使います", "Picoです", "microbitです",
            "board is esp32", "arduino uno", "using m5stack", "raspberry pi pico",
        ],
    }
    records: list[dict[str, str]] = []
    projects = list(PROJECTS)
    stages = ["orient", "parts_check", "minimal_circuit", "firmware_upload", "observe_serial", "debug_triage"]
    questions = [
        "いまの状態をそのまま教えてください",
        "LEDの長い足は抵抗を通ってGPIO側につながっていますか？",
        "センサー、LED、ボードのGNDは同じGND列につながっていますか？",
        "手元にある部品を一行で書けますか？",
        "ボード名は分かりますか？",
        "エラーや画面表示はありますか？",
    ]
    for index in range(samples):
        kind = ANSWER_KIND_LABELS[index % len(ANSWER_KIND_LABELS)]
        answer = rng.choice(examples[kind])
        project_id = rng.choice(projects)
        payload = {
            "projectId": project_id,
            "projectTitle": PROJECTS[project_id],
            "currentStage": rng.choice(stages),
            "question": rng.choice(questions_for_kind(kind, questions, rng)),
            "answer": mutate_short_answer(answer, rng),
            "inventory": rng.choice(["", "ESP32, LED, resistor", "Arduino, sensor, wires", "M5Stack Grove cable", "not sure"]),
            "budget": rng.choice(["0", "1000", "3000", "5000", "10000", "20000"]),
            "symptom": "none" if kind in {"confirmed", "inventory_report", "board_report"} else rng.choice(["none", "not working", "error", "stuck"]),
            "skill": rng.choice(["", "LEDは少し分かる", "GNDが不安", "Arduino IDEは初めて"]),
            "previous": rng.choice(["", "Lチカだけやった", "配線で止まった", "部品名が不明"]),
        }
        records.append({"source": build_source(payload), "kind": kind, "target": kind})
    return records


def questions_for_kind(kind: str, common_questions: list[str], rng: random.Random) -> list[str]:
    focused = {
        "inventory_report": ["手元にある部品を一行で書けますか？", "持っている部品は何がありますか？"],
        "board_report": ["使うボードはArduino、ESP32、M5Stack、Picoのどれですか？", "ボード名は分かりますか？"],
        "photo_request": ["配線が不安なら写真で見ます。どうしますか？", "写真で確認したいですか？"],
        "problem_report": ["エラーや画面表示はありますか？", "いま何が起きていますか？"],
        "confirmed": ["LEDの長い足は抵抗を通ってGPIO側につながっていますか？", "GNDは同じ列につながっていますか？"],
        "negative": ["LEDの長い足は抵抗を通ってGPIO側につながっていますか？", "GNDは同じ列につながっていますか？"],
        "unknown": ["いま見えている状態をそのまま書けますか？", "どこが分からないですか？"],
    }
    return focused.get(kind, common_questions) + rng.sample(common_questions, k=min(2, len(common_questions)))


def mutate_short_answer(answer: str, rng: random.Random) -> str:
    prefix = rng.choice(["", "", "今は、", "たぶん、", "見た感じ、", "すみません、"])
    suffix = rng.choice(["", "", "です", "と思います", "。次は？", "、これで合ってますか"])
    if rng.random() < 0.08:
        answer = answer.replace("GND", "gnd").replace("LED", "led")
    return f"{prefix}{answer}{suffix}"


def select_device(requested: str) -> torch.device:
    if requested == "auto":
        return torch.device("cuda" if torch.cuda.is_available() else "cpu")
    return torch.device(requested)


def prefix(name: str, metrics: dict[str, float]) -> dict[str, float]:
    return {f"{name}_{key}": value for key, value in metrics.items()}


def save_checkpoint(
    path: Path,
    model: TutorialAnswerClassifier,
    config: TutorialAnswerClassifierConfig,
    epoch: int,
    metrics: dict[str, float],
) -> None:
    torch.save({
        "model_state": model.state_dict(),
        "config": config.to_json(),
        "epoch": epoch,
        "metrics": metrics,
        "labels": ANSWER_KIND_LABELS,
    }, path)


if __name__ == "__main__":
    main()
