from __future__ import annotations

import argparse
import json
import random
from pathlib import Path

import torch
from torch.utils.data import DataLoader, random_split

from ai_models.makergraph.tokenizer import MakerTokenizer

from .config import NeuralAgentConfig
from .data import NeuralAgentDataset, collect_texts, collate_agent_batch, generate_records, load_jsonl, write_jsonl
from .losses import accuracy, multilabel_f1, neural_agent_loss
from .model import NeuralAgentTransformer, count_parameters
from .taxonomy import DEBUG_CAUSES, FIRMWARE_CLASSES, RISK_CLASSES, SAFETY_LABELS


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Train multitask neural agents for safety, debug, and firmware.")
    parser.add_argument("--data", default="data/neural_agents_training.jsonl")
    parser.add_argument("--output", default="runs/neural_agents")
    parser.add_argument("--records-per-project", type=int, default=1200)
    parser.add_argument("--epochs", type=int, default=18)
    parser.add_argument("--batch-size", type=int, default=64)
    parser.add_argument("--lr", type=float, default=3e-4)
    parser.add_argument("--weight-decay", type=float, default=0.01)
    parser.add_argument("--max-length", type=int, default=224)
    parser.add_argument("--d-model", type=int, default=192)
    parser.add_argument("--n-heads", type=int, default=6)
    parser.add_argument("--n-layers", type=int, default=4)
    parser.add_argument("--seed", type=int, default=47)
    parser.add_argument("--device", default="auto", choices=["auto", "cpu", "cuda"])
    parser.add_argument("--amp", action="store_true")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    set_seed(args.seed)
    data_path = Path(args.data)
    if not data_path.exists():
        write_jsonl(data_path, generate_records(args.records_per_project, seed=args.seed))
    records = load_jsonl(data_path)
    tokenizer = MakerTokenizer()
    tokenizer.build_vocab(collect_texts(records), min_freq=1, max_vocab_size=14000)
    config = NeuralAgentConfig(
        vocab_size=len(tokenizer.vocab),
        safety_count=len(SAFETY_LABELS),
        debug_count=len(DEBUG_CAUSES),
        risk_count=len(RISK_CLASSES),
        firmware_count=len(FIRMWARE_CLASSES),
        max_length=args.max_length,
        d_model=args.d_model,
        n_heads=args.n_heads,
        n_layers=args.n_layers,
    )
    dataset = NeuralAgentDataset(records, tokenizer, config.max_length)
    train_size = int(len(dataset) * 0.88)
    valid_size = len(dataset) - train_size
    train_dataset, valid_dataset = random_split(dataset, [train_size, valid_size], generator=torch.Generator().manual_seed(args.seed))
    device = select_device(args.device)
    train_loader = DataLoader(train_dataset, batch_size=args.batch_size, shuffle=True, collate_fn=collate_agent_batch)
    valid_loader = DataLoader(valid_dataset, batch_size=args.batch_size, shuffle=False, collate_fn=collate_agent_batch)
    model = NeuralAgentTransformer(config).to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)
    scaler = torch.cuda.amp.GradScaler(enabled=args.amp and device.type == "cuda")
    output_dir = Path(args.output)
    output_dir.mkdir(parents=True, exist_ok=True)
    tokenizer.save(output_dir / "tokenizer.json")
    config.save(output_dir / "config.json")
    (output_dir / "labels.json").write_text(json.dumps({
        "safety": SAFETY_LABELS,
        "risk": RISK_CLASSES,
        "debug": DEBUG_CAUSES,
        "firmware": FIRMWARE_CLASSES,
    }, ensure_ascii=False, indent=2), encoding="utf-8")
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
    model: NeuralAgentTransformer,
    loader: DataLoader,
    optimizer: torch.optim.Optimizer,
    scaler: torch.cuda.amp.GradScaler,
    device: torch.device,
    train: bool,
    amp: bool,
) -> dict[str, float]:
    model.train(train)
    total = {"loss": 0.0, "safety_f1": 0.0, "risk_acc": 0.0, "debug_acc": 0.0, "firmware_acc": 0.0}
    steps = 0
    for batch in loader:
        batch = move_batch(batch, device)
        with torch.set_grad_enabled(train):
            with torch.cuda.amp.autocast(enabled=amp and device.type == "cuda"):
                outputs = model(batch["input_ids"], batch["attention_mask"])
                loss, _parts = neural_agent_loss(outputs, batch)
            if train:
                optimizer.zero_grad(set_to_none=True)
                scaler.scale(loss).backward()
                scaler.unscale_(optimizer)
                torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
                scaler.step(optimizer)
                scaler.update()
        total["loss"] += float(loss.detach().cpu())
        total["safety_f1"] += multilabel_f1(outputs["safety_logits"].detach(), batch["safety_targets"])
        total["risk_acc"] += accuracy(outputs["risk_logits"].detach(), batch["risk_class"])
        total["debug_acc"] += accuracy(outputs["debug_logits"].detach(), batch["debug_class"])
        total["firmware_acc"] += accuracy(outputs["firmware_logits"].detach(), batch["firmware_class"])
        steps += 1
    return {key: value / max(1, steps) for key, value in total.items()}


def move_batch(batch: dict[str, object], device: torch.device) -> dict[str, object]:
    return {key: value.to(device) if isinstance(value, torch.Tensor) else value for key, value in batch.items()}


def save_checkpoint(path: Path, model: NeuralAgentTransformer, config: NeuralAgentConfig, epoch: int, metrics: dict[str, float]) -> None:
    torch.save({"model_type": "neural_agent_transformer", "model_state": model.state_dict(), "config": config.to_json(), "epoch": epoch, "metrics": metrics}, path)


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

