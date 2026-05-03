from __future__ import annotations

import json
import re
from collections import Counter
from pathlib import Path

from .config import DEFAULT_DOMAIN_TERMS


SPECIAL_TOKENS = [
    "<pad>",
    "<unk>",
    "<bos>",
    "<eos>",
    "<sep>",
    "<query>",
    "<project>",
    "<component>",
    "<graph>",
    "<log>",
    "<budget>",
    "<skill>",
    "<safety>",
]


class MakerTokenizer:
    """Small domain tokenizer for Japanese maker text.

    This intentionally has no third-party dependency so the repository stays
    portable. It greedily preserves electronics terms, groups ASCII words and
    numbers, and falls back to character tokens for Japanese text.
    """

    def __init__(self, vocab: dict[str, int] | None = None, domain_terms: list[str] | None = None) -> None:
        self.domain_terms = sorted(domain_terms or DEFAULT_DOMAIN_TERMS, key=len, reverse=True)
        self.vocab = vocab or {token: index for index, token in enumerate(SPECIAL_TOKENS)}
        self.id_to_token = {index: token for token, index in self.vocab.items()}

    @property
    def pad_id(self) -> int:
        return self.vocab["<pad>"]

    @property
    def unk_id(self) -> int:
        return self.vocab["<unk>"]

    def tokenize(self, text: str) -> list[str]:
        text = normalize_text(text)
        tokens: list[str] = []
        index = 0
        while index < len(text):
            char = text[index]
            if char.isspace():
                index += 1
                continue

            matched = None
            lowered = text[index:].lower()
            for term in self.domain_terms:
                if lowered.startswith(term.lower()):
                    matched = term
                    break
            if matched:
                tokens.append(matched)
                index += len(matched)
                continue

            ascii_match = re.match(r"[a-zA-Z_][a-zA-Z0-9_+\-]*", text[index:])
            if ascii_match:
                tokens.append(ascii_match.group(0).lower())
                index += len(ascii_match.group(0))
                continue

            number_match = re.match(r"\d+(?:,\d{3})*(?:\.\d+)?", text[index:])
            if number_match:
                tokens.append("<num>")
                index += len(number_match.group(0))
                continue

            tokens.append(char)
            index += 1
        return tokens

    def build_vocab(self, texts: list[str], min_freq: int = 1, max_vocab_size: int = 12000) -> None:
        counter: Counter[str] = Counter()
        for text in texts:
            counter.update(self.tokenize(text))

        vocab = {token: index for index, token in enumerate(SPECIAL_TOKENS)}
        for token, count in counter.most_common():
            if count < min_freq:
                continue
            if token in vocab:
                continue
            if len(vocab) >= max_vocab_size:
                break
            vocab[token] = len(vocab)
        self.vocab = vocab
        self.id_to_token = {index: token for token, index in vocab.items()}

    def encode(
        self,
        text: str,
        max_length: int,
        prefix_token: str | None = None,
    ) -> tuple[list[int], list[int]]:
        tokens = ["<bos>"]
        if prefix_token:
            tokens.append(prefix_token)
        tokens.extend(self.tokenize(text))
        tokens.append("<eos>")
        ids = [self.vocab.get(token, self.unk_id) for token in tokens[:max_length]]
        mask = [1] * len(ids)
        while len(ids) < max_length:
            ids.append(self.pad_id)
            mask.append(0)
        return ids, mask

    def decode(self, token_ids: list[int]) -> str:
        tokens = []
        for token_id in token_ids:
            token = self.id_to_token.get(int(token_id), "<unk>")
            if token in SPECIAL_TOKENS:
                continue
            tokens.append(token)
        return "".join(tokens)

    def save(self, path: str | Path) -> None:
        payload = {"vocab": self.vocab, "domain_terms": self.domain_terms}
        Path(path).write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")

    @classmethod
    def load(cls, path: str | Path) -> "MakerTokenizer":
        payload = json.loads(Path(path).read_text(encoding="utf-8"))
        return cls(vocab={key: int(value) for key, value in payload["vocab"].items()}, domain_terms=payload["domain_terms"])


def normalize_text(text: str) -> str:
    return (
        str(text)
        .replace("〜", "-")
        .replace("～", "-")
        .replace("　", " ")
        .strip()
    )
