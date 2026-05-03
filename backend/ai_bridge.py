from __future__ import annotations

from pathlib import Path
from typing import Any


ROOT = Path(__file__).resolve().parents[1]
DEFAULT_MODEL_DIR = ROOT / "runs" / "maker_intent_transformer"


def transformer_intent(payload: dict[str, Any]) -> dict[str, Any]:
    """Run the trained Transformer if a checkpoint exists.

    The API server itself has no hard PyTorch dependency. This keeps the
    prototype runnable on small machines while still letting Colab/CUDA-trained
    checkpoints be dropped into runs/maker_intent_transformer.
    """

    text = " ".join(
        str(payload.get(key, ""))
        for key in ["text", "mood", "problem", "selectedOption", "inventory", "budget"]
        if payload.get(key)
    ).strip()
    if not text:
        text = "作りたいものは分からない。Lチカの次に進みたい。"

    model_dir = Path(payload.get("modelDir") or DEFAULT_MODEL_DIR)
    checkpoint = model_dir / "best.pt"
    tokenizer = model_dir / "tokenizer.json"
    if not checkpoint.exists() or not tokenizer.exists():
        return {
            "available": False,
            "reason": f"checkpoint not found: {checkpoint}",
            "expectedCommand": "python -m ai_models.makergraph.train_intent_encoder --device cuda --amp",
        }

    try:
        from ai_models.makergraph.infer import MakerIntentInference
    except Exception as exc:  # pragma: no cover - depends on optional torch install
        return {
            "available": False,
            "reason": f"PyTorch inference import failed: {exc}",
            "expectedInstall": "pip install -r requirements-ml.txt",
        }

    try:
        service = MakerIntentInference(model_dir=model_dir, device=str(payload.get("device") or "auto"))
        return {
            "available": True,
            "modelDir": str(model_dir),
            "device": str(service.device),
            "profile": service.profile(text),
            "recommendations": service.rank_projects(text, top_k=int(payload.get("topK") or 3)),
        }
    except Exception as exc:  # pragma: no cover - runtime model loading branch
        return {
            "available": False,
            "reason": f"Transformer inference failed: {exc}",
            "modelDir": str(model_dir),
        }
