from __future__ import annotations

import argparse
import json
import random
from pathlib import Path

import torch
from torch.utils.data import DataLoader, random_split

from ai_models.makergraph.tokenizer import MakerTokenizer

from .config import TutorialAgentConfig
from .data import TutorialDataset, collect_texts, collate_tutorial_batch, generate_records, load_jsonl, write_jsonl
from .losses import accuracy, mean_abs_error, multilabel_f1, tutorial_loss
from .model import TutorialAgentTransformer, count_parameters
from .taxonomy import ACTIONS, CHECKPOINTS, CONCEPTS, PROJECT_IDS, QUESTIONS, ROUTES, TUTORIAL_STAGES


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Train neural autonomous tutorial agent.")
    parser.add_argument("--data", default="data/tutorial_agent_training.jsonl")
    parser.add_argument("--output", default="runs/tutorial_agent")
    parser.add_argument("--samples", type=int, default=16000)
    parser.add_argument("--epochs", type=int, default=12)
    parser.add_argument("--batch-size", type=int, default=64)
    parser.add_argument("--lr", type=float, default=3e-4)
    parser.add_argument("--weight-decay", type=float, default=0.01)
    parser.add_argument("--max-length", type=int, default=256)
    parser.add_argument("--d-model", type=int, default=192)
    parser.add_argument("--n-heads", type=int, default=6)
    parser.add_argument("--n-layers", type=int, default=4)
    parser.add_argument("--seed", type=int, default=89)
    parser.add_argument("--device", default="auto", choices=["auto", "cpu", "cuda"])
    parser.add_argument("--amp", action="store_true")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    set_seed(args.seed)
    data_path = Path(args.data)
    if not data_path.exists():
        write_jsonl(data_path, generate_records(args.samples, seed=args.seed))
    records = load_jsonl(data_path)
    tokenizer = MakerTokenizer()
    tokenizer.build_vocab(collect_texts(records), min_freq=1, max_vocab_size=18000)
    config = TutorialAgentConfig(
        vocab_size=len(tokenizer.vocab),
        project_count=len(PROJECT_IDS),
        stage_count=len(TUTORIAL_STAGES),
        action_count=len(ACTIONS),
        question_count=len(QUESTIONS),
        checkpoint_count=len(CHECKPOINTS),
        route_count=len(ROUTES),
        concept_count=len(CONCEPTS),
        max_length=args.max_length,
        d_model=args.d_model,
        n_heads=args.n_heads,
        n_layers=args.n_layers,
    )
    dataset = TutorialDataset(records, tokenizer, config.max_length)
    train_size = int(len(dataset) * 0.88)
    valid_size = len(dataset) - train_size
    train_dataset, valid_dataset = random_split(dataset, [train_size, valid_size], generator=torch.Generator().manual_seed(args.seed))
    device = select_device(args.device)
    train_loader = DataLoader(train_dataset, batch_size=args.batch_size, shuffle=True, collate_fn=collate_tutorial_batch)
    valid_loader = DataLoader(valid_dataset, batch_size=args.batch_size, shuffle=False, collate_fn=collate_tutorial_batch)
    model = TutorialAgentTransformer(config).to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)
    scaler = torch.cuda.amp.GradScaler(enabled=args.amp and device.type == "cuda")
    output_dir = Path(args.output)
    output_dir.mkdir(parents=True, exist_ok=True)
    tokenizer.save(output_dir / "tokenizer.json")
    config.save(output_dir / "config.json")
    (output_dir / "labels.json").write_text(json.dumps({
        "projects": PROJECT_IDS,
        "stages": TUTORIAL_STAGES,
        "actions": ACTIONS,
        "questions": QUESTIONS,
        "checkpoints": CHECKPOINTS,
        "routes": ROUTES,
        "concepts": CONCEPTS,
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
    model: TutorialAgentTransformer,
    loader: DataLoader,
    optimizer: torch.optim.Optimizer,
    scaler: torch.cuda.amp.GradScaler,
    device: torch.device,
    train: bool,
    amp: bool,
) -> dict[str, float]:
    model.train(train)
    total = {
        "loss": 0.0,
        "stage_acc": 0.0,
        "action_acc": 0.0,
        "question_acc": 0.0,
        "checkpoint_acc": 0.0,
        "route_acc": 0.0,
        "concept_f1": 0.0,
        "autonomy_mae": 0.0,
    }
    steps = 0
    for batch in loader:
        batch = {key: value.to(device) for key, value in batch.items()}
        with torch.set_grad_enabled(train):
            with torch.cuda.amp.autocast(enabled=amp and device.type == "cuda"):
                outputs = model(batch["input_ids"], batch["attention_mask"])
                loss, _parts = tutorial_loss(outputs, batch)
            if train:
                optimizer.zero_grad(set_to_none=True)
                scaler.scale(loss).backward()
                scaler.unscale_(optimizer)
                torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
                scaler.step(optimizer)
                scaler.update()
        total["loss"] += float(loss.detach().cpu())
        total["stage_acc"] += accuracy(outputs["stage_logits"].detach(), batch["stage_class"])
        total["action_acc"] += accuracy(outputs["action_logits"].detach(), batch["action_class"])
        total["question_acc"] += accuracy(outputs["question_logits"].detach(), batch["question_class"])
        total["checkpoint_acc"] += accuracy(outputs["checkpoint_logits"].detach(), batch["checkpoint_class"])
        total["route_acc"] += accuracy(outputs["route_logits"].detach(), batch["route_class"])
        total["concept_f1"] += multilabel_f1(outputs["concept_logits"].detach(), batch["concept_targets"])
        total["autonomy_mae"] += mean_abs_error(outputs["autonomy_logits"].detach(), batch["autonomy_target"])
        steps += 1
    return {key: value / max(1, steps) for key, value in total.items()}


def save_checkpoint(path: Path, model: TutorialAgentTransformer, config: TutorialAgentConfig, epoch: int, metrics: dict[str, float]) -> None:
    torch.save({"model_type": "tutorial_agent_transformer", "model_state": model.state_dict(), "config": config.to_json(), "epoch": epoch, "metrics": metrics}, path)


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
