from __future__ import annotations

import argparse
import json
import random
import re
import time
from pathlib import Path
from typing import Any

import torch
from torch import nn
from torch.utils.data import DataLoader, Dataset, random_split

from ai_models.makergraph.tokenizer import MakerTokenizer

from .answer_classifier import TutorialAnswerClassifier, TutorialAnswerClassifierConfig, count_parameters
from .data import load_jsonl
from .taxonomy import STAGES


STAGE_LABELS = list(STAGES)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Train neural tutorial next-stage classifier.")
    parser.add_argument("--data", action="append", required=True, help="JSONL response records with source and target stage lines.")
    parser.add_argument("--output", default="runs/tutorial_stage_classifier")
    parser.add_argument("--epochs", type=int, default=4)
    parser.add_argument("--batch-size", type=int, default=192)
    parser.add_argument("--lr", type=float, default=4e-4)
    parser.add_argument("--weight-decay", type=float, default=0.01)
    parser.add_argument("--max-source-length", type=int, default=360)
    parser.add_argument("--d-model", type=int, default=160)
    parser.add_argument("--n-heads", type=int, default=5)
    parser.add_argument("--n-layers", type=int, default=3)
    parser.add_argument("--seed", type=int, default=52018)
    parser.add_argument("--device", default="auto", choices=["auto", "cpu", "cuda"])
    parser.add_argument("--amp", action="store_true")
    return parser.parse_args()


def main() -> None:
    started = time.perf_counter()
    args = parse_args()
    random.seed(args.seed)
    torch.manual_seed(args.seed)
    records = load_stage_records(args.data)
    if not records:
        raise SystemExit("no stage records loaded")
    random.Random(args.seed).shuffle(records)

    tokenizer = MakerTokenizer(domain_terms=[])
    tokenizer.build_vocab(collect_stage_texts(records), min_freq=1, max_vocab_size=22000)
    config = TutorialAnswerClassifierConfig(
        vocab_size=len(tokenizer.vocab),
        max_source_length=args.max_source_length,
        d_model=args.d_model,
        n_heads=args.n_heads,
        n_layers=args.n_layers,
    )
    dataset = TutorialStageDataset(records, tokenizer, config.max_source_length)
    train_size = int(len(dataset) * 0.9)
    valid_size = len(dataset) - train_size
    train_dataset, valid_dataset = random_split(dataset, [train_size, valid_size], generator=torch.Generator().manual_seed(args.seed))
    device = select_device(args.device)
    train_loader = DataLoader(train_dataset, batch_size=args.batch_size, shuffle=True, collate_fn=collate_batch)
    valid_loader = DataLoader(valid_dataset, batch_size=args.batch_size, shuffle=False, collate_fn=collate_batch)
    model = TutorialAnswerClassifier(config, num_labels=len(STAGE_LABELS)).to(device)
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
        "distribution": dict(sorted(stage_distribution(records).items())),
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


def load_stage_records(paths: list[str]) -> list[dict[str, str]]:
    records: list[dict[str, str]] = []
    seen: set[tuple[str, str]] = set()
    for raw_path in paths:
        path = Path(raw_path)
        if not path.exists():
            print(json.dumps({"event": "stage_data_missing", "path": str(path)}, ensure_ascii=False), flush=True)
            continue
        loaded = 0
        for record in load_jsonl(path):
            source = str(record.get("source") or "")
            stage = str(record.get("stage") or record.get("nextStage") or "")
            if not stage:
                stage = parse_stage_from_target(str(record.get("target") or ""))
            if not source or stage not in STAGE_LABELS:
                continue
            key = (source, stage)
            if key in seen:
                continue
            seen.add(key)
            records.append({"source": source, "stage": stage})
            loaded += 1
        print(json.dumps({"event": "stage_data_loaded", "path": str(path), "records": loaded}, ensure_ascii=False), flush=True)
    return records


def parse_stage_from_target(target: str) -> str:
    match = re.search(r"(?:^|\n)stage:([a-z_]+)", target)
    return match.group(1).strip() if match else ""


def stage_distribution(records: list[dict[str, str]]) -> dict[str, int]:
    counts: dict[str, int] = {}
    for record in records:
        counts[record["stage"]] = counts.get(record["stage"], 0) + 1
    return counts


def collect_stage_texts(records: list[dict[str, str]]) -> list[str]:
    texts: list[str] = []
    for record in records:
        texts.append(record["source"])
        texts.append(record["stage"])
    return texts


class TutorialStageDataset(Dataset):
    def __init__(self, records: list[dict[str, str]], tokenizer: MakerTokenizer, max_source_length: int) -> None:
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
            "label": torch.tensor(STAGE_LABELS.index(record["stage"]), dtype=torch.long),
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


def select_device(requested: str) -> torch.device:
    if requested == "cuda":
        return torch.device("cuda" if torch.cuda.is_available() else "cpu")
    if requested == "cpu":
        return torch.device("cpu")
    return torch.device("cuda" if torch.cuda.is_available() else "cpu")


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
        "labels": STAGE_LABELS,
    }, path)


if __name__ == "__main__":
    main()
