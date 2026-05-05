from __future__ import annotations

import argparse
import base64
import io
import json
from pathlib import Path
from typing import Any

import torch
import torch.nn.functional as F

from .config import WireCheckConfig
from .model import WireCheckNet, decode_outputs, load_checkpoint
from .synthetic import SyntheticWiringDataset


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run WireCheckNet inference.")
    parser.add_argument("--model-dir", default="runs/wirechecknet")
    parser.add_argument("--image", default=None)
    parser.add_argument("--synthetic", action="store_true")
    parser.add_argument("--device", default="auto")
    parser.add_argument("--threshold", type=float, default=0.35)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    service = WireCheckInference(args.model_dir, args.device)
    if args.synthetic:
        result = service.predict_synthetic(score_threshold=args.threshold)
    elif args.image:
        result = service.predict_image_path(args.image, score_threshold=args.threshold)
    else:
        raise SystemExit("Use --synthetic or --image path")
    print(json.dumps(result, ensure_ascii=False, indent=2))


class WireCheckInference:
    def __init__(self, model_dir: str | Path, device: str = "auto") -> None:
        self.model_dir = Path(model_dir)
        selected_device = "cuda" if device == "auto" and torch.cuda.is_available() else device
        if selected_device == "auto":
            selected_device = "cpu"
        self.device = torch.device(selected_device)
        self.model, self.config, self.payload = load_checkpoint(str(self.model_dir / "best.pt"), map_location=self.device)
        self.model.to(self.device)

    @torch.no_grad()
    def predict_tensor(self, image: torch.Tensor, score_threshold: float = 0.35) -> dict[str, Any]:
        if image.ndim == 3:
            image = image.unsqueeze(0)
        image = resize_image_tensor(image.to(self.device), self.config.image_size)
        outputs = self.model(image)
        decoded = decode_outputs(outputs, self.config, score_threshold=score_threshold)
        decoded["device"] = str(self.device)
        decoded["modelDir"] = str(self.model_dir)
        decoded["expectedCircuitDiff"] = estimate_diff(decoded)
        return decoded

    def predict_synthetic(self, score_threshold: float = 0.35) -> dict[str, Any]:
        dataset = SyntheticWiringDataset(1, self.config, seed=999)
        sample = dataset[0]
        result = self.predict_tensor(sample["image"], score_threshold=score_threshold)
        result["syntheticLabel"] = {
            "risk": self.config.risk_classes[int(sample["risk_class"])],
            "circuitName": sample["circuit_name"],
        }
        return result

    def predict_image_path(self, path: str | Path, score_threshold: float = 0.35) -> dict[str, Any]:
        return self.predict_tensor(load_image_tensor(path), score_threshold=score_threshold)

    def predict_base64(self, image_base64: str, score_threshold: float = 0.35) -> dict[str, Any]:
        raw = base64.b64decode(image_base64)
        return self.predict_tensor(load_image_bytes(raw), score_threshold=score_threshold)


def load_image_path_with_pillow(path: str | Path) -> torch.Tensor:
    from PIL import Image

    image = Image.open(path).convert("RGB")
    return pil_to_tensor(image)


def load_image_tensor(path: str | Path) -> torch.Tensor:
    return load_image_path_with_pillow(path)


def load_image_bytes(raw: bytes) -> torch.Tensor:
    from PIL import Image

    image = Image.open(io.BytesIO(raw)).convert("RGB")
    return pil_to_tensor(image)


def pil_to_tensor(image: Any) -> torch.Tensor:
    width, height = image.size
    data = torch.ByteTensor(torch.ByteStorage.from_buffer(image.tobytes()))
    data = data.view(height, width, 3).permute(2, 0, 1).float() / 255.0
    return data


def resize_image_tensor(image: torch.Tensor, image_size: int) -> torch.Tensor:
    return F.interpolate(image, size=(image_size, image_size), mode="bilinear", align_corners=False).clamp(0, 1)


def estimate_diff(decoded: dict[str, Any]) -> list[dict[str, str]]:
    part_names = {part["class"] for part in decoded["parts"]}
    wire_names = {wire["class"] for wire in decoded["wires"]}
    differences = []
    if "gnd" not in wire_names:
        differences.append({"target": "GND", "status": "suspicious", "message": "GND線が認識されていません。共通GNDを確認してください。"})
    if "led" in part_names and "resistor" not in part_names:
        differences.append({"target": "LED resistor", "status": "danger", "message": "LED用の直列抵抗が認識されていません。GPIO直結の可能性があります。"})
    if decoded["risk"]["class"] in {"voltage_mismatch", "unsafe_power"}:
        differences.append({"target": "power", "status": "danger", "message": "電源または電圧不一致の危険候補があります。USBを抜いて確認してください。"})
    if not differences:
        differences.append({"target": "overall", "status": "ok", "message": "合っていそうです。ピン番号とコード定数を最後に照合してください。"})
    return differences


if __name__ == "__main__":
    main()
