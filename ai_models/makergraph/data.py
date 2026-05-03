from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import torch
from torch.utils.data import Dataset

from .sample_data import PROJECT_CATALOG, TRAINING_EXAMPLES, project_by_id, project_to_document
from .tokenizer import MakerTokenizer


PROFILE_KEYS = ["difficulty_tolerance", "budget_sensitivity", "novelty_preference"]


def write_seed_jsonl(path: str | Path) -> None:
    target = Path(path)
    target.parent.mkdir(parents=True, exist_ok=True)
    with target.open("w", encoding="utf-8") as handle:
        for example in TRAINING_EXAMPLES:
            project = project_by_id(example["positive_project_id"])
            record = {
                "query": example["query"],
                "positive_project": project,
                "positive_project_text": project_to_document(project),
                "targets": example["targets"],
            }
            handle.write(json.dumps(record, ensure_ascii=False) + "\n")


def load_jsonl(path: str | Path) -> list[dict[str, Any]]:
    records: list[dict[str, Any]] = []
    with Path(path).open("r", encoding="utf-8") as handle:
        for line in handle:
            line = line.strip()
            if line:
                records.append(json.loads(line))
    return records


def collect_training_texts(records: list[dict[str, Any]]) -> list[str]:
    texts = []
    for record in records:
        texts.append(record["query"])
        texts.append(record.get("positive_project_text") or project_to_document(record["positive_project"]))
        texts.append(project_to_graph_sequence(record["positive_project"]))
    for project in PROJECT_CATALOG:
        texts.append(project_to_document(project))
        texts.append(project_to_graph_sequence(project))
    return texts


class IntentProjectDataset(Dataset):
    def __init__(self, records: list[dict[str, Any]], tokenizer: MakerTokenizer, max_length: int) -> None:
        self.records = records
        self.tokenizer = tokenizer
        self.max_length = max_length

    def __len__(self) -> int:
        return len(self.records)

    def __getitem__(self, index: int) -> dict[str, Any]:
        record = self.records[index]
        project_text = record.get("positive_project_text") or project_to_document(record["positive_project"])
        query_ids, query_mask = self.tokenizer.encode(record["query"], self.max_length, "<query>")
        project_ids, project_mask = self.tokenizer.encode(project_text, self.max_length, "<project>")
        targets = record.get("targets", {})
        profile = [float(targets.get(key, 0.5)) for key in PROFILE_KEYS]
        return {
            "query_ids": torch.tensor(query_ids, dtype=torch.long),
            "query_mask": torch.tensor(query_mask, dtype=torch.bool),
            "project_ids": torch.tensor(project_ids, dtype=torch.long),
            "project_mask": torch.tensor(project_mask, dtype=torch.bool),
            "profile": torch.tensor(profile, dtype=torch.float32),
            "query_text": record["query"],
            "project_id": record["positive_project"]["id"],
        }


def collate_batch(items: list[dict[str, Any]]) -> dict[str, Any]:
    return {
        "query_ids": torch.stack([item["query_ids"] for item in items]),
        "query_mask": torch.stack([item["query_mask"] for item in items]),
        "project_ids": torch.stack([item["project_ids"] for item in items]),
        "project_mask": torch.stack([item["project_mask"] for item in items]),
        "profile": torch.stack([item["profile"] for item in items]),
        "query_text": [item["query_text"] for item in items],
        "project_id": [item["project_id"] for item in items],
    }


def project_to_graph_sequence(project: dict[str, Any]) -> str:
    graph = list(project.get("graph", []))
    skills = [f"skill:{skill}" for skill in project.get("skills", [])]
    budget = [f"cost:{project.get('budget', 'unknown')}"]
    risk = [f"risk:difficulty_{project.get('difficulty', 1)}"]
    failures = [f"failure:{failure}" for failure in project.get("failure_logs", [])]
    return " <sep> ".join(graph + skills + budget + risk + failures)


class ProjectGraphDataset(Dataset):
    def __init__(self, records: list[dict[str, Any]], tokenizer: MakerTokenizer, max_length: int) -> None:
        self.records = records
        self.tokenizer = tokenizer
        self.max_length = max_length

    def __len__(self) -> int:
        return len(self.records)

    def __getitem__(self, index: int) -> dict[str, Any]:
        record = self.records[index]
        source_ids, source_mask = self.tokenizer.encode(record["query"], self.max_length, "<query>")
        graph_sequence = project_to_graph_sequence(record["positive_project"])
        target_ids, target_mask = self.tokenizer.encode(graph_sequence, self.max_length, "<graph>")
        decoder_input = target_ids[:-1]
        decoder_mask = target_mask[:-1]
        labels = target_ids[1:]
        label_mask = target_mask[1:]
        return {
            "source_ids": torch.tensor(source_ids, dtype=torch.long),
            "source_mask": torch.tensor(source_mask, dtype=torch.bool),
            "decoder_input_ids": torch.tensor(decoder_input, dtype=torch.long),
            "decoder_input_mask": torch.tensor(decoder_mask, dtype=torch.bool),
            "labels": torch.tensor(labels, dtype=torch.long),
            "label_mask": torch.tensor(label_mask, dtype=torch.bool),
            "query_text": record["query"],
            "graph_text": graph_sequence,
            "project_id": record["positive_project"]["id"],
        }


def collate_graph_batch(items: list[dict[str, Any]]) -> dict[str, Any]:
    return {
        "source_ids": torch.stack([item["source_ids"] for item in items]),
        "source_mask": torch.stack([item["source_mask"] for item in items]),
        "decoder_input_ids": torch.stack([item["decoder_input_ids"] for item in items]),
        "decoder_input_mask": torch.stack([item["decoder_input_mask"] for item in items]),
        "labels": torch.stack([item["labels"] for item in items]),
        "label_mask": torch.stack([item["label_mask"] for item in items]),
        "query_text": [item["query_text"] for item in items],
        "graph_text": [item["graph_text"] for item in items],
        "project_id": [item["project_id"] for item in items],
    }
