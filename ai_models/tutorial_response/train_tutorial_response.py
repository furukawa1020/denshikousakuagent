from __future__ import annotations

import argparse
import json
import random
import time
from pathlib import Path

import torch
from torch import nn
from torch.utils.data import DataLoader, random_split

from ai_models.makergraph.tokenizer import MakerTokenizer

from .config import TutorialResponseConfig
from .data import TutorialResponseDataset, collect_texts, collate_response_batch, generate_records, load_jsonl, write_jsonl
from .model import TutorialResponseTransformer, count_parameters


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Train neural free-response tutorial generator.")
    parser.add_argument("--data", default="data/tutorial_response_training.jsonl")
    parser.add_argument("--output", default="runs/tutorial_response")
    parser.add_argument("--samples", type=int, default=14000)
    parser.add_argument("--robust-ratio", type=float, default=0.7)
    parser.add_argument("--regenerate", action="store_true")
    parser.add_argument("--epochs", type=int, default=8)
    parser.add_argument("--batch-size", type=int, default=48)
    parser.add_argument("--lr", type=float, default=3e-4)
    parser.add_argument("--weight-decay", type=float, default=0.01)
    parser.add_argument("--max-source-length", type=int, default=320)
    parser.add_argument("--max-target-length", type=int, default=160)
    parser.add_argument("--d-model", type=int, default=192)
    parser.add_argument("--n-heads", type=int, default=6)
    parser.add_argument("--n-layers", type=int, default=3)
    parser.add_argument("--seed", type=int, default=131)
    parser.add_argument("--device", default="auto", choices=["auto", "cpu", "cuda"])
    parser.add_argument("--amp", action="store_true")
    return parser.parse_args()


def main() -> None:
    started = time.perf_counter()
    args = parse_args()
    set_seed(args.seed)
    print(json.dumps({"event": "start", "data": args.data, "output": args.output, "epochs": args.epochs}, ensure_ascii=False), flush=True)
    data_path = Path(args.data)
    if args.regenerate or not data_path.exists():
        write_jsonl(data_path, generate_records(args.samples, seed=args.seed, robust_ratio=args.robust_ratio))
    records = load_jsonl(data_path)
    print(json.dumps({"event": "data_loaded", "records": len(records), "elapsed_sec": round(time.perf_counter() - started, 2)}, ensure_ascii=False), flush=True)
    tokenizer = MakerTokenizer(domain_terms=[])
    tokenizer.build_vocab(collect_texts(records), min_freq=1, max_vocab_size=22000)
    print(json.dumps({"event": "vocab_built", "vocab_size": len(tokenizer.vocab), "elapsed_sec": round(time.perf_counter() - started, 2)}, ensure_ascii=False), flush=True)
    config = TutorialResponseConfig(
        vocab_size=len(tokenizer.vocab),
        max_source_length=args.max_source_length,
        max_target_length=args.max_target_length,
        d_model=args.d_model,
        n_heads=args.n_heads,
        n_encoder_layers=args.n_layers,
        n_decoder_layers=args.n_layers,
    )
    dataset = TutorialResponseDataset(records, tokenizer, config.max_source_length, config.max_target_length)
    train_size = int(len(dataset) * 0.9)
    valid_size = len(dataset) - train_size
    train_dataset, valid_dataset = random_split(dataset, [train_size, valid_size], generator=torch.Generator().manual_seed(args.seed))
    device = select_device(args.device)
    train_loader = DataLoader(train_dataset, batch_size=args.batch_size, shuffle=True, collate_fn=collate_response_batch)
    valid_loader = DataLoader(valid_dataset, batch_size=args.batch_size, shuffle=False, collate_fn=collate_response_batch)
    model = TutorialResponseTransformer(config).to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)
    scaler = torch.cuda.amp.GradScaler(enabled=args.amp and device.type == "cuda")
    output_dir = Path(args.output)
    output_dir.mkdir(parents=True, exist_ok=True)
    tokenizer.save(output_dir / "tokenizer.json")
    config.save(output_dir / "config.json")
    print(json.dumps({
        "device": str(device),
        "cuda_available": torch.cuda.is_available(),
        "cuda_name": torch.cuda.get_device_name(0) if torch.cuda.is_available() else None,
        "records": len(records),
        "vocab_size": len(tokenizer.vocab),
        "parameters": count_parameters(model),
        "elapsed_sec": round(time.perf_counter() - started, 2),
    }, ensure_ascii=False, indent=2), flush=True)

    best_valid = float("inf")
    history: list[dict[str, float]] = []
    for epoch in range(1, args.epochs + 1):
        train_metrics = run_epoch(model, train_loader, optimizer, scaler, device, train=True, amp=args.amp, phase="train", epoch=epoch)
        valid_metrics = run_epoch(model, valid_loader, optimizer, scaler, device, train=False, amp=False, phase="valid", epoch=epoch)
        row = {"epoch": epoch, **prefix("train", train_metrics), **prefix("valid", valid_metrics)}
        history.append(row)
        print(json.dumps({"event": "epoch_complete", **row}, ensure_ascii=False), flush=True)
        if valid_metrics["loss"] <= best_valid:
            best_valid = valid_metrics["loss"]
            save_checkpoint(output_dir / "best.pt", model, config, epoch, valid_metrics)
    save_checkpoint(output_dir / "last.pt", model, config, args.epochs, history[-1])
    (output_dir / "metrics.json").write_text(json.dumps(history, ensure_ascii=False, indent=2), encoding="utf-8")


def run_epoch(
    model: TutorialResponseTransformer,
    loader: DataLoader,
    optimizer: torch.optim.Optimizer,
    scaler: torch.cuda.amp.GradScaler,
    device: torch.device,
    train: bool,
    amp: bool,
    phase: str,
    epoch: int,
) -> dict[str, float]:
    model.train(train)
    criterion = nn.CrossEntropyLoss(ignore_index=model.config.pad_token_id)
    total_loss = 0.0
    total_tokens = 0
    total_correct = 0
    steps = 0
    for batch in loader:
        batch = {key: value.to(device) for key, value in batch.items()}
        with torch.set_grad_enabled(train):
            with torch.cuda.amp.autocast(enabled=amp and device.type == "cuda"):
                logits = model(batch["source_ids"], batch["source_mask"], batch["decoder_input_ids"], batch["label_mask"])
                loss = criterion(logits.reshape(-1, logits.shape[-1]), batch["labels"].reshape(-1))
            if train:
                optimizer.zero_grad(set_to_none=True)
                scaler.scale(loss).backward()
                scaler.unscale_(optimizer)
                torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
                scaler.step(optimizer)
                scaler.update()
        labels = batch["labels"]
        mask = labels.ne(model.config.pad_token_id)
        predictions = logits.detach().argmax(dim=-1)
        total_correct += int((predictions.eq(labels) & mask).sum().detach().cpu())
        total_tokens += int(mask.sum().detach().cpu())
        total_loss += float(loss.detach().cpu())
        steps += 1
        if steps == 1 or steps % 100 == 0 or steps == len(loader):
            print(
                json.dumps(
                    {
                        "event": "batch_progress",
                        "epoch": epoch,
                        "phase": phase,
                        "step": steps,
                        "total_steps": len(loader),
                        "loss": total_loss / max(1, steps),
                        "token_acc": total_correct / max(1, total_tokens),
                    },
                    ensure_ascii=False,
                ),
                flush=True,
            )
    return {"loss": total_loss / max(1, steps), "token_acc": total_correct / max(1, total_tokens)}


def save_checkpoint(path: Path, model: TutorialResponseTransformer, config: TutorialResponseConfig, epoch: int, metrics: dict[str, float]) -> None:
    torch.save({"model_type": "tutorial_response_transformer", "model_state": model.state_dict(), "config": config.to_json(), "epoch": epoch, "metrics": metrics}, path)


def select_device(value: str) -> torch.device:
    if value == "cuda":
        if not torch.cuda.is_available():
            raise RuntimeError("CUDA was requested, but torch.cuda.is_available() is false.")
        return torch.device("cuda")
    if value == "cpu":
        return torch.device("cpu")
    return torch.device("cuda" if torch.cuda.is_available() else "cpu")


def set_seed(seed: int) -> None:
    random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def prefix(name: str, metrics: dict[str, float]) -> dict[str, float]:
    return {f"{name}_{key}": value for key, value in metrics.items()}


if __name__ == "__main__":
    main()
