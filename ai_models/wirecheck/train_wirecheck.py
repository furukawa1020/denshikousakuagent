from __future__ import annotations

import argparse
import json
import random
from pathlib import Path

import torch
from torch import nn
from torch.utils.data import DataLoader, random_split

from .config import WireCheckConfig
from .model import WireCheckNet, count_parameters
from .synthetic import SyntheticWiringDataset


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Train WireCheckNet on synthetic wiring-photo data.")
    parser.add_argument("--output", default="runs/wirechecknet")
    parser.add_argument("--samples", type=int, default=8000)
    parser.add_argument("--epochs", type=int, default=25)
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--lr", type=float, default=3e-4)
    parser.add_argument("--weight-decay", type=float, default=0.01)
    parser.add_argument("--image-size", type=int, default=224)
    parser.add_argument("--d-model", type=int, default=192)
    parser.add_argument("--n-heads", type=int, default=6)
    parser.add_argument("--encoder-layers", type=int, default=4)
    parser.add_argument("--decoder-layers", type=int, default=3)
    parser.add_argument("--seed", type=int, default=44)
    parser.add_argument("--device", default="auto", choices=["auto", "cpu", "cuda"])
    parser.add_argument("--amp", action="store_true")
    parser.add_argument("--num-workers", type=int, default=0)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    set_seed(args.seed)
    config = WireCheckConfig(
        image_size=args.image_size,
        d_model=args.d_model,
        n_heads=args.n_heads,
        encoder_layers=args.encoder_layers,
        decoder_layers=args.decoder_layers,
    )
    device = select_device(args.device)
    dataset = SyntheticWiringDataset(args.samples, config, seed=args.seed)
    train_size = int(len(dataset) * 0.9)
    valid_size = len(dataset) - train_size
    train_dataset, valid_dataset = random_split(dataset, [train_size, valid_size], generator=torch.Generator().manual_seed(args.seed))

    train_loader = DataLoader(
        train_dataset,
        batch_size=args.batch_size,
        shuffle=True,
        num_workers=args.num_workers,
        pin_memory=device.type == "cuda",
    )
    valid_loader = DataLoader(
        valid_dataset,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=args.num_workers,
        pin_memory=device.type == "cuda",
    )

    model = WireCheckNet(config).to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)
    scaler = torch.cuda.amp.GradScaler(enabled=args.amp and device.type == "cuda")

    output_dir = Path(args.output)
    output_dir.mkdir(parents=True, exist_ok=True)
    config.save(output_dir / "config.json")
    print(json.dumps({
        "device": str(device),
        "cuda_available": torch.cuda.is_available(),
        "cuda_name": torch.cuda.get_device_name(0) if torch.cuda.is_available() else None,
        "samples": len(dataset),
        "parameters": count_parameters(model),
        "image_size": config.image_size,
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
    model: WireCheckNet,
    loader: DataLoader,
    optimizer: torch.optim.Optimizer,
    scaler: torch.cuda.amp.GradScaler,
    device: torch.device,
    train: bool,
    amp: bool,
) -> dict[str, float]:
    model.train(train)
    total_loss = 0.0
    total_part_acc = 0.0
    total_wire_acc = 0.0
    total_risk_acc = 0.0
    steps = 0
    for batch in loader:
        images = batch["image"].to(device, non_blocking=True)
        part_classes = batch["part_classes"].to(device, non_blocking=True)
        part_boxes = batch["part_boxes"].to(device, non_blocking=True)
        wire_classes = batch["wire_classes"].to(device, non_blocking=True)
        wire_endpoints = batch["wire_endpoints"].to(device, non_blocking=True)
        risk_class = batch["risk_class"].to(device, non_blocking=True)

        with torch.set_grad_enabled(train):
            with torch.cuda.amp.autocast(enabled=amp and device.type == "cuda"):
                outputs = model(images)
                loss, parts = wirecheck_loss(outputs, part_classes, part_boxes, wire_classes, wire_endpoints, risk_class)
            if train:
                optimizer.zero_grad(set_to_none=True)
                scaler.scale(loss).backward()
                scaler.unscale_(optimizer)
                torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
                scaler.step(optimizer)
                scaler.update()

        total_loss += float(loss.detach().cpu())
        total_part_acc += slot_accuracy(outputs["part_logits"].detach(), part_classes)
        total_wire_acc += slot_accuracy(outputs["wire_logits"].detach(), wire_classes)
        total_risk_acc += class_accuracy(outputs["risk_logits"].detach(), risk_class)
        steps += 1
    return {
        "loss": total_loss / max(1, steps),
        "part_acc": total_part_acc / max(1, steps),
        "wire_acc": total_wire_acc / max(1, steps),
        "risk_acc": total_risk_acc / max(1, steps),
    }


def wirecheck_loss(
    outputs: dict[str, torch.Tensor],
    part_classes: torch.Tensor,
    part_boxes: torch.Tensor,
    wire_classes: torch.Tensor,
    wire_endpoints: torch.Tensor,
    risk_class: torch.Tensor,
) -> tuple[torch.Tensor, dict[str, torch.Tensor]]:
    part_class_loss = nn.functional.cross_entropy(outputs["part_logits"].transpose(1, 2), part_classes)
    wire_class_loss = nn.functional.cross_entropy(outputs["wire_logits"].transpose(1, 2), wire_classes)
    risk_loss = nn.functional.cross_entropy(outputs["risk_logits"], risk_class)

    part_mask = part_classes.ne(0).unsqueeze(-1)
    wire_mask = wire_classes.ne(0).unsqueeze(-1)
    part_box_loss = masked_l1(outputs["part_boxes"], part_boxes, part_mask)
    wire_endpoint_loss = masked_l1(outputs["wire_endpoints"], wire_endpoints, wire_mask)
    loss = part_class_loss + wire_class_loss + 0.8 * risk_loss + 4.0 * part_box_loss + 3.0 * wire_endpoint_loss
    return loss, {
        "part_class": part_class_loss,
        "wire_class": wire_class_loss,
        "risk": risk_loss,
        "part_box": part_box_loss,
        "wire_endpoint": wire_endpoint_loss,
    }


def masked_l1(predicted: torch.Tensor, target: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
    if mask.float().sum() < 1:
        return predicted.sum() * 0
    return (torch.abs(predicted - target) * mask).sum() / mask.float().sum().clamp(min=1)


@torch.no_grad()
def slot_accuracy(logits: torch.Tensor, labels: torch.Tensor) -> float:
    predictions = logits.argmax(dim=-1)
    return float((predictions == labels).float().mean().detach().cpu())


@torch.no_grad()
def class_accuracy(logits: torch.Tensor, labels: torch.Tensor) -> float:
    predictions = logits.argmax(dim=-1)
    return float((predictions == labels).float().mean().detach().cpu())


def save_checkpoint(path: Path, model: WireCheckNet, config: WireCheckConfig, epoch: int, metrics: dict[str, float]) -> None:
    torch.save(
        {
            "model_type": "wirechecknet",
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
