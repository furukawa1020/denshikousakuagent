from __future__ import annotations

import argparse
import json
import random
import time
from pathlib import Path

import torch
from torch import nn
from torch.nn import functional as F
from torch.utils.data import DataLoader, random_split

from ai_models.makergraph.tokenizer import MakerTokenizer

from .config import TutorialResponseConfig
from .data import TutorialResponseDataset, collect_texts, collate_response_batch, generate_records, load_jsonl, write_jsonl
from .model import TutorialResponseTransformer, count_parameters


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Train neural free-response tutorial generator.")
    parser.add_argument("--data", default="data/tutorial_response_training.jsonl")
    parser.add_argument("--extra-data", action="append", default=[], help="Additional JSONL records, such as exported real interaction logs.")
    parser.add_argument("--extra-weight", type=int, default=2, help="Repeat each extra-data record this many times during training.")
    parser.add_argument("--preference-data", action="append", default=[], help="JSONL records with source/chosen/rejected tutorial responses.")
    parser.add_argument("--preference-loss-weight", type=float, default=0.2)
    parser.add_argument("--preference-beta", type=float, default=0.2)
    parser.add_argument("--preference-batch-size", type=int, default=24)
    parser.add_argument("--keep-negative-extra", action="store_true", help="Keep negatively rated extra records instead of skipping them.")
    parser.add_argument("--feedback-up-multiplier", type=int, default=2)
    parser.add_argument("--feedback-fix-multiplier", type=int, default=5)
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
    extra_records = load_extra_records(
        args.extra_data,
        max(1, args.extra_weight),
        keep_negative=args.keep_negative_extra,
        up_multiplier=max(1, args.feedback_up_multiplier),
        fix_multiplier=max(1, args.feedback_fix_multiplier),
    )
    records.extend(extra_records)
    preference_records = load_preference_records(args.preference_data)
    print(json.dumps({"event": "data_loaded", "records": len(records), "elapsed_sec": round(time.perf_counter() - started, 2)}, ensure_ascii=False), flush=True)
    tokenizer = MakerTokenizer(domain_terms=[])
    tokenizer.build_vocab(collect_texts(records) + collect_preference_texts(preference_records), min_freq=1, max_vocab_size=22000)
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
    preference_loader = None
    if preference_records:
        preference_dataset = TutorialPreferenceDataset(preference_records, tokenizer, config.max_source_length, config.max_target_length)
        preference_loader = DataLoader(preference_dataset, batch_size=args.preference_batch_size, shuffle=True, collate_fn=collate_preference_batch)
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
        "preference_records": len(preference_records),
        "vocab_size": len(tokenizer.vocab),
        "parameters": count_parameters(model),
        "elapsed_sec": round(time.perf_counter() - started, 2),
    }, ensure_ascii=False, indent=2), flush=True)

    best_valid = float("inf")
    history: list[dict[str, float]] = []
    for epoch in range(1, args.epochs + 1):
        train_metrics = run_epoch(
            model,
            train_loader,
            optimizer,
            scaler,
            device,
            train=True,
            amp=args.amp,
            phase="train",
            epoch=epoch,
            preference_loader=preference_loader,
            preference_loss_weight=args.preference_loss_weight,
            preference_beta=args.preference_beta,
        )
        valid_metrics = run_epoch(model, valid_loader, optimizer, scaler, device, train=False, amp=False, phase="valid", epoch=epoch)
        row = {"epoch": epoch, **prefix("train", train_metrics), **prefix("valid", valid_metrics)}
        history.append(row)
        print(json.dumps({"event": "epoch_complete", **row}, ensure_ascii=False), flush=True)
        if valid_metrics["loss"] <= best_valid:
            best_valid = valid_metrics["loss"]
            save_checkpoint(output_dir / "best.pt", model, config, epoch, valid_metrics)
    save_checkpoint(output_dir / "last.pt", model, config, args.epochs, history[-1])
    (output_dir / "metrics.json").write_text(json.dumps(history, ensure_ascii=False, indent=2), encoding="utf-8")


def load_extra_records(paths: list[str], weight: int, keep_negative: bool = False, up_multiplier: int = 2, fix_multiplier: int = 5) -> list[dict[str, str]]:
    records: list[dict[str, str]] = []
    seen: set[tuple[str, str]] = set()
    feedback_counts = {"up": 0, "down": 0, "fix": 0, "none": 0}
    for raw_path in paths:
        path = Path(raw_path)
        if not path.exists():
            print(json.dumps({"event": "extra_data_missing", "path": str(path)}, ensure_ascii=False), flush=True)
            continue
        loaded = load_jsonl(path)
        for record in loaded:
            source = str(record.get("source") or "")
            target = str(record.get("target") or "")
            if not source or not target:
                continue
            feedback = str(record.get("feedback") or "none")
            if feedback == "down" and not keep_negative:
                feedback_counts["down"] += 1
                continue
            key = (source, target)
            if key in seen:
                continue
            seen.add(key)
            repeat = feedback_repeat(feedback, weight, up_multiplier, fix_multiplier)
            feedback_counts[feedback if feedback in feedback_counts else "none"] += 1
            for _ in range(repeat):
                records.append({"source": source, "target": target})
        print(json.dumps({"event": "extra_data_loaded", "path": str(path), "unique_records": len(seen), "weighted_records": len(records), "feedback": feedback_counts}, ensure_ascii=False), flush=True)
    return records


def load_preference_records(paths: list[str]) -> list[dict[str, str]]:
    records: list[dict[str, str]] = []
    seen: set[tuple[str, str, str]] = set()
    for raw_path in paths:
        path = Path(raw_path)
        if not path.exists():
            print(json.dumps({"event": "preference_data_missing", "path": str(path)}, ensure_ascii=False), flush=True)
            continue
        loaded = load_jsonl(path)
        for record in loaded:
            source = str(record.get("source") or "")
            chosen = str(record.get("chosen") or "")
            rejected = str(record.get("rejected") or "")
            if not source or not chosen or not rejected or chosen == rejected:
                continue
            key = (source, chosen, rejected)
            if key in seen:
                continue
            seen.add(key)
            records.append({"source": source, "chosen": chosen, "rejected": rejected})
        print(json.dumps({"event": "preference_data_loaded", "path": str(path), "records": len(records)}, ensure_ascii=False), flush=True)
    return records


def collect_preference_texts(records: list[dict[str, str]]) -> list[str]:
    texts: list[str] = []
    for record in records:
        texts.extend([record["source"], record["chosen"], record["rejected"]])
    return texts


class TutorialPreferenceDataset(torch.utils.data.Dataset):
    def __init__(self, records: list[dict[str, str]], tokenizer: MakerTokenizer, max_source_length: int, max_target_length: int) -> None:
        self.records = records
        self.tokenizer = tokenizer
        self.max_source_length = max_source_length
        self.max_target_length = max_target_length

    def __len__(self) -> int:
        return len(self.records)

    def __getitem__(self, index: int) -> dict[str, torch.Tensor]:
        record = self.records[index]
        source_ids, source_mask = self.tokenizer.encode(record["source"], self.max_source_length, "<query>")
        chosen_ids, chosen_mask = self.tokenizer.encode(record["chosen"], self.max_target_length, "<log>")
        rejected_ids, rejected_mask = self.tokenizer.encode(record["rejected"], self.max_target_length, "<log>")
        return {
            "source_ids": torch.tensor(source_ids, dtype=torch.long),
            "source_mask": torch.tensor(source_mask, dtype=torch.bool),
            "chosen_decoder_input_ids": torch.tensor(chosen_ids[:-1], dtype=torch.long),
            "chosen_labels": torch.tensor(chosen_ids[1:], dtype=torch.long),
            "chosen_label_mask": torch.tensor(chosen_mask[1:], dtype=torch.bool),
            "rejected_decoder_input_ids": torch.tensor(rejected_ids[:-1], dtype=torch.long),
            "rejected_labels": torch.tensor(rejected_ids[1:], dtype=torch.long),
            "rejected_label_mask": torch.tensor(rejected_mask[1:], dtype=torch.bool),
        }


def collate_preference_batch(items: list[dict[str, torch.Tensor]]) -> dict[str, torch.Tensor]:
    return {key: torch.stack([item[key] for item in items]) for key in items[0]}


def feedback_repeat(feedback: str, weight: int, up_multiplier: int, fix_multiplier: int) -> int:
    if feedback == "fix":
        return weight * fix_multiplier
    if feedback == "up":
        return weight * up_multiplier
    return weight


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
    preference_loader: DataLoader | None = None,
    preference_loss_weight: float = 0.0,
    preference_beta: float = 0.2,
) -> dict[str, float]:
    model.train(train)
    criterion = nn.CrossEntropyLoss(ignore_index=model.config.pad_token_id)
    total_loss = 0.0
    total_tokens = 0
    total_correct = 0
    total_preference_loss = 0.0
    total_preference_correct = 0
    total_preference_pairs = 0
    steps = 0
    preference_iter = iter(preference_loader) if train and preference_loader is not None and preference_loss_weight > 0 else None
    for batch in loader:
        batch = {key: value.to(device) for key, value in batch.items()}
        with torch.set_grad_enabled(train):
            with torch.cuda.amp.autocast(enabled=amp and device.type == "cuda"):
                logits = model(batch["source_ids"], batch["source_mask"], batch["decoder_input_ids"], batch["label_mask"])
                loss = criterion(logits.reshape(-1, logits.shape[-1]), batch["labels"].reshape(-1))
                preference_parts: dict[str, float] = {}
                if preference_iter is not None:
                    preference_batch, preference_iter = next_preference_batch(preference_iter, preference_loader, device)
                    preference_loss, preference_parts = preference_loss_for_batch(model, preference_batch, preference_beta)
                    loss = loss + preference_loss_weight * preference_loss
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
        if preference_parts:
            total_preference_loss += preference_parts["loss"]
            total_preference_correct += int(preference_parts["correct"])
            total_preference_pairs += int(preference_parts["pairs"])
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
                        "preference_loss": total_preference_loss / max(1, steps) if total_preference_pairs else 0.0,
                        "preference_acc": total_preference_correct / max(1, total_preference_pairs),
                    },
                    ensure_ascii=False,
                ),
                flush=True,
            )
    return {
        "loss": total_loss / max(1, steps),
        "token_acc": total_correct / max(1, total_tokens),
        "preference_loss": total_preference_loss / max(1, steps) if total_preference_pairs else 0.0,
        "preference_acc": total_preference_correct / max(1, total_preference_pairs),
    }


def next_preference_batch(preference_iter: Any, preference_loader: DataLoader, device: torch.device) -> tuple[dict[str, torch.Tensor], Any]:
    try:
        batch = next(preference_iter)
    except StopIteration:
        preference_iter = iter(preference_loader)
        batch = next(preference_iter)
    return {key: value.to(device) for key, value in batch.items()}, preference_iter


def preference_loss_for_batch(model: TutorialResponseTransformer, batch: dict[str, torch.Tensor], beta: float) -> tuple[torch.Tensor, dict[str, float]]:
    chosen_logp = response_log_probability(
        model,
        batch["source_ids"],
        batch["source_mask"],
        batch["chosen_decoder_input_ids"],
        batch["chosen_labels"],
        batch["chosen_label_mask"],
    )
    rejected_logp = response_log_probability(
        model,
        batch["source_ids"],
        batch["source_mask"],
        batch["rejected_decoder_input_ids"],
        batch["rejected_labels"],
        batch["rejected_label_mask"],
    )
    margin = chosen_logp - rejected_logp
    loss = -F.logsigmoid(beta * margin).mean()
    return loss, {
        "loss": float(loss.detach().cpu()),
        "correct": int(margin.gt(0).sum().detach().cpu()),
        "pairs": int(margin.numel()),
    }


def response_log_probability(
    model: TutorialResponseTransformer,
    source_ids: torch.Tensor,
    source_mask: torch.Tensor,
    decoder_input_ids: torch.Tensor,
    labels: torch.Tensor,
    label_mask: torch.Tensor,
) -> torch.Tensor:
    logits = model(source_ids, source_mask, decoder_input_ids, label_mask)
    log_probs = F.log_softmax(logits, dim=-1)
    token_log_probs = log_probs.gather(-1, labels.unsqueeze(-1)).squeeze(-1)
    token_log_probs = token_log_probs * label_mask.float()
    lengths = label_mask.float().sum(dim=1).clamp_min(1.0)
    return token_log_probs.sum(dim=1) / lengths


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
