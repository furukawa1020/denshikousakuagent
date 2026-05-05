from __future__ import annotations

import argparse
import json
import random
from pathlib import Path

import torch
from torch.utils.data import DataLoader, random_split

from .config import MakerIntentConfig
from .data import IntentProjectDataset, collect_training_texts, collate_batch, load_jsonl, write_seed_jsonl
from .losses import labeled_retrieval_accuracy, multi_positive_contrastive_loss, profile_regression_loss
from .model import MakerGraphDualEncoder, count_parameters
from .tokenizer import MakerTokenizer


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Train Maker Intent Transformer with contrastive learning.")
    parser.add_argument("--data", default="data/intent_training_seed.jsonl")
    parser.add_argument("--output", default="runs/maker_intent_transformer")
    parser.add_argument("--epochs", type=int, default=25)
    parser.add_argument("--batch-size", type=int, default=8)
    parser.add_argument("--lr", type=float, default=3e-4)
    parser.add_argument("--weight-decay", type=float, default=0.01)
    parser.add_argument("--max-length", type=int, default=256)
    parser.add_argument("--d-model", type=int, default=256)
    parser.add_argument("--n-heads", type=int, default=8)
    parser.add_argument("--n-layers", type=int, default=4)
    parser.add_argument("--embedding-dim", type=int, default=128)
    parser.add_argument("--min-freq", type=int, default=1)
    parser.add_argument("--max-vocab-size", type=int, default=12000)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--device", default="auto", choices=["auto", "cpu", "cuda"])
    parser.add_argument("--amp", action="store_true", help="Use CUDA mixed precision when available.")
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
        embedding_dim=args.embedding_dim,
    )
    device = select_device(args.device)
    dataset = IntentProjectDataset(records, tokenizer, config.max_length)
    train_size = max(1, int(len(dataset) * 0.85))
    valid_size = max(0, len(dataset) - train_size)
    if valid_size:
        train_dataset, valid_dataset = random_split(dataset, [train_size, valid_size], generator=torch.Generator().manual_seed(args.seed))
    else:
        train_dataset, valid_dataset = dataset, dataset

    train_loader = DataLoader(train_dataset, batch_size=args.batch_size, shuffle=True, collate_fn=collate_batch)
    valid_loader = DataLoader(valid_dataset, batch_size=args.batch_size, shuffle=False, collate_fn=collate_batch)

    model = MakerGraphDualEncoder(config).to(device)
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
    }, ensure_ascii=False, indent=2))

    best_valid = float("inf")
    history: list[dict[str, float]] = []
    for epoch in range(1, args.epochs + 1):
        train_metrics = run_epoch(model, train_loader, optimizer, scaler, device, train=True, amp=args.amp)
        valid_metrics = run_epoch(model, valid_loader, optimizer, scaler, device, train=False, amp=False)
        row = {"epoch": epoch, **prefix("train", train_metrics), **prefix("valid", valid_metrics)}
        history.append(row)
        print(json.dumps(row, ensure_ascii=False))

        if valid_metrics["loss"] <= best_valid:
            best_valid = valid_metrics["loss"]
            save_checkpoint(output_dir / "best.pt", model, config, epoch, valid_metrics)

    save_checkpoint(output_dir / "last.pt", model, config, args.epochs, history[-1])
    (output_dir / "metrics.json").write_text(json.dumps(history, ensure_ascii=False, indent=2), encoding="utf-8")


def run_epoch(
    model: MakerGraphDualEncoder,
    loader: DataLoader,
    optimizer: torch.optim.Optimizer,
    scaler: torch.cuda.amp.GradScaler,
    device: torch.device,
    train: bool,
    amp: bool,
) -> dict[str, float]:
    model.train(train)
    total_loss = 0.0
    total_retrieval = 0.0
    total_profile = 0.0
    steps = 0
    for batch in loader:
        query_ids = batch["query_ids"].to(device)
        query_mask = batch["query_mask"].to(device)
        project_ids = batch["project_ids"].to(device)
        project_mask = batch["project_mask"].to(device)
        profile = batch["profile"].to(device)

        with torch.set_grad_enabled(train):
            with torch.cuda.amp.autocast(enabled=amp and device.type == "cuda"):
                outputs = model(query_ids, query_mask, project_ids, project_mask)
                contrastive = multi_positive_contrastive_loss(outputs["logits"], batch["project_id"])
                profile_loss = profile_regression_loss(outputs["query_profile"], profile)
                loss = contrastive + 0.15 * profile_loss

            if train:
                optimizer.zero_grad(set_to_none=True)
                scaler.scale(loss).backward()
                scaler.unscale_(optimizer)
                torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
                scaler.step(optimizer)
                scaler.update()

        total_loss += float(loss.detach().cpu())
        total_retrieval += labeled_retrieval_accuracy(outputs["logits"].detach(), batch["project_id"], top_k=1)
        total_profile += float(profile_loss.detach().cpu())
        steps += 1
    return {
        "loss": total_loss / max(1, steps),
        "retrieval_top1": total_retrieval / max(1, steps),
        "profile_mse": total_profile / max(1, steps),
    }


def save_checkpoint(path: Path, model: MakerGraphDualEncoder, config: MakerIntentConfig, epoch: int, metrics: dict[str, float]) -> None:
    torch.save(
        {
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
