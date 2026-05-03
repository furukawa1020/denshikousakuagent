from __future__ import annotations

import argparse
import json
from pathlib import Path

import torch

from .config import MakerIntentConfig
from .model import ProjectGraphSequenceTransformer
from .tokenizer import MakerTokenizer


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Generate project graph tokens from intent text.")
    parser.add_argument("--model-dir", default="runs/project_graph_transformer")
    parser.add_argument("--text", required=True)
    parser.add_argument("--device", default="auto")
    parser.add_argument("--max-new-tokens", type=int, default=96)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    generator = ProjectGraphGenerator(args.model_dir, args.device)
    print(json.dumps(generator.generate(args.text, args.max_new_tokens), ensure_ascii=False, indent=2))


class ProjectGraphGenerator:
    def __init__(self, model_dir: str | Path, device: str = "auto") -> None:
        self.model_dir = Path(model_dir)
        selected_device = "cuda" if device == "auto" and torch.cuda.is_available() else device
        if selected_device == "auto":
            selected_device = "cpu"
        self.device = torch.device(selected_device)
        self.tokenizer = MakerTokenizer.load(self.model_dir / "tokenizer.json")
        payload = torch.load(self.model_dir / "best.pt", map_location=self.device)
        self.config = MakerIntentConfig.from_json(payload["config"])
        self.model = ProjectGraphSequenceTransformer(self.config).to(self.device)
        self.model.load_state_dict(payload["model_state"])
        self.model.eval()

    @torch.no_grad()
    def generate(self, text: str, max_new_tokens: int = 96) -> dict[str, object]:
        source_ids, source_mask = self.tokenizer.encode(text, self.config.max_length, "<query>")
        source = torch.tensor([source_ids], dtype=torch.long, device=self.device)
        source_attention = torch.tensor([source_mask], dtype=torch.bool, device=self.device)

        bos_id = self.tokenizer.vocab["<bos>"]
        graph_id = self.tokenizer.vocab["<graph>"]
        eos_id = self.tokenizer.vocab["<eos>"]
        pad_id = self.tokenizer.pad_id
        generated = [bos_id, graph_id]

        for _ in range(max_new_tokens):
            decoder = generated[-(self.config.max_length - 1):]
            decoder_mask = [1] * len(decoder)
            while len(decoder) < self.config.max_length - 1:
                decoder.append(pad_id)
                decoder_mask.append(0)
            decoder_ids = torch.tensor([decoder], dtype=torch.long, device=self.device)
            decoder_attention = torch.tensor([decoder_mask], dtype=torch.bool, device=self.device)
            logits = self.model(source, source_attention, decoder_ids, decoder_attention)
            next_index = min(len(generated) - 1, logits.size(1) - 1)
            next_id = int(logits[0, next_index].argmax().detach().cpu())
            generated.append(next_id)
            if next_id == eos_id:
                break

        tokens = [self.tokenizer.id_to_token.get(token_id, "<unk>") for token_id in generated]
        graph_tokens = [token for token in tokens if token not in {"<bos>", "<eos>", "<pad>", "<graph>"}]
        return {
            "text": text,
            "device": str(self.device),
            "tokens": graph_tokens,
            "graphSequence": " ".join(graph_tokens),
        }


if __name__ == "__main__":
    main()
