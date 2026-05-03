from __future__ import annotations

import argparse
import json
import random
from pathlib import Path

import torch
from torch import nn
from torch.utils.data import DataLoader, random_split

from .config import MakerIntentConfig
from .data import (
    ProjectGraphDataset,
    collect_training_texts,
    collate_graph_batch,
    load_jsonl,
    write_seed_jsonl,
)
from .model import ProjectGraphSequenceTransformer, count_parameters
from .tokenizer import MakerTokenizer


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Train Transformer Project Graph Generator.")
    parser.add_argument("--data", default="data/intent_training_synthetic.jsonl")
    parser.add_argument("--output", default="runs/project_graph_transformer")
    parser.add_argument("--epochs", type=int, default=35)
    parser.add_argument("--batch-size", type=int, default=8)
    parser.add_argument("--lr", type=float, default=3e-4)
    parser.add_argument("--weight-decay", type=float, default=0.01)
    parser.add_argument("--max-length", type=int, default=192)
    parser.add_argument("--d-model", type=int, default=256)
    parser.add_argument("--n-heads", type=int, default=8)
    parser.add_argument("--n-layers", type=int, default=4)
    parser.add_argument("--min-freq", type=int, default=1)
    parser.add_argument("--max-vocab-size", type=int, default=14000)
    parser.add_argument("--seed", type=int, default=43)
    parser.add_argument("--device", default="auto", choices=["auto", "cpu", "cuda"])
    parser.add_argument("--amp", action="store_true")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    set_seed(args.seed)
    data_path = Path(args.data)
    if not data_path.exists():
        write_seed_jsonl(data_path)
    records = load_jsonl(data_path)

    tokenizer = MakerTokenizer()
    tokenizer.build_vocab(collect_training_texts(records), min_freq=args.min_freq, max_vocab_size=args.max_vocab_size)
    config = MakerIntentConfig(
        vocab_size=len(tokenizer.vocab),
        max_length=args.max_length,
        d_model=args.d_model,
        n_heads=args.n_heads,
        n_layers=args.n_layers,
    )
    device = select_device(args.device)
    dataset = ProjectGraphDataset(records, tokenizer, config.max_length)
    train_size = max(1, int(len(dataset) * 0.88))
    valid_size = max(0, len(dataset) - train_size)
    if valid_size:
        train_dataset, valid_dataset = random_split(dataset, [train_size, valid_size], generator=torch.Generator().manual_seed(args.seed))
    else:
        train_dataset, valid_dataset = dataset, dataset

    train_loader = DataLoader(train_dataset, batch_size=args.batch_size, shuffle=True, collate_fn=collate_graph_batch)
    valid_loader = DataLoader(valid_dataset, batch_size=args.batch_size, shuffle=False, collate_fn=collate_graph_batch)

    model = ProjectGraphSequenceTransformer(config).to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)
    scaler = torch.cuda.amp.GradScaler(enabled=args.amp and device.type == "cuda")
    loss_fn = nn.CrossEntropyLoss(ignore_index=config.pad_token_id, label_smoothing=0.04)

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
    }, ensure_ascii=False, indent=2))

    best_valid = float("inf")
    history = []
    for epoch in range(1, args.epochs + 1):
        train_metrics = run_epoch(model, train_loader, optimizer, scaler, loss_fn, device, True, args.amp)
        valid_metrics = run_epoch(model, valid_loader, optimizer, scaler, loss_fn, device, False, False)
        row = {"epoch": epoch, **prefix("train", train_metrics), **prefix("valid", valid_metrics)}
        history.append(row)
        print(json.dumps(row, ensure_ascii=False))
        if valid_metrics["loss"] <= best_valid:
            best_valid = valid_metrics["loss"]
            save_checkpoint(output_dir / "best.pt", model, config, epoch, valid_metrics)

    save_checkpoint(output_dir / "last.pt", model, config, args.epochs, history[-1])
    (output_dir / "metrics.json").write_text(json.dumps(history, ensure_ascii=False, indent=2), encoding="utf-8")


def run_epoch(
    model: ProjectGraphSequenceTransformer,
    loader: DataLoader,
    optimizer: torch.optim.Optimizer,
    scaler: torch.cuda.amp.GradScaler,
    loss_fn: nn.CrossEntropyLoss,
    device: torch.device,
    train: bool,
    amp: bool,
) -> dict[str, float]:
    model.train(train)
    total_loss = 0.0
    total_token_acc = 0.0
    steps = 0
    for batch in loader:
        source_ids = batch["source_ids"].to(device)
        source_mask = batch["source_mask"].to(device)
        decoder_input_ids = batch["decoder_input_ids"].to(device)
        decoder_input_mask = batch["decoder_input_mask"].to(device)
        labels = batch["labels"].to(device)
        label_mask = batch["label_mask"].to(device)

        with torch.set_grad_enabled(train):
            with torch.cuda.amp.autocast(enabled=amp and device.type == "cuda"):
                logits = model(source_ids, source_mask, decoder_input_ids, decoder_input_mask)
                loss = loss_fn(logits.reshape(-1, logits.size(-1)), labels.reshape(-1))
            if train:
                optimizer.zero_grad(set_to_none=True)
                scaler.scale(loss).backward()
                scaler.unscale_(optimizer)
                torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
                scaler.step(optimizer)
                scaler.update()

        total_loss += float(loss.detach().cpu())
        total_token_acc += token_accuracy(logits.detach(), labels, label_mask)
        steps += 1
    return {"loss": total_loss / max(1, steps), "token_accuracy": total_token_acc / max(1, steps)}


@torch.no_grad()
def token_accuracy(logits: torch.Tensor, labels: torch.Tensor, label_mask: torch.Tensor) -> float:
    predictions = logits.argmax(dim=-1)
    correct = ((predictions == labels) & label_mask).float().sum()
    total = label_mask.float().sum().clamp(min=1)
    return float((correct / total).detach().cpu())


def save_checkpoint(path: Path, model: ProjectGraphSequenceTransformer, config: MakerIntentConfig, epoch: int, metrics: dict[str, float]) -> None:
    torch.save(
        {
            "model_type": "project_graph_transformer",
            "model_state": model.state_dict(),
            "config": config.to_json(),
            "epoch": epoch,
            "metrics": metrics,
        },
        path,
    )


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
