from __future__ import annotations

from pathlib import Path
import sys
from typing import Any


ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))
DEFAULT_MODEL_DIR = ROOT / "runs" / "maker_intent_transformer"
DEFAULT_GRAPH_MODEL_DIR = ROOT / "runs" / "project_graph_transformer"
DEFAULT_WIRECHECK_MODEL_DIR = ROOT / "runs" / "wirechecknet"
DEFAULT_BOM_MODEL_DIR = ROOT / "runs" / "bom_estimator"


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


def transformer_project_graph(payload: dict[str, Any]) -> dict[str, Any]:
    text = " ".join(
        str(payload.get(key, ""))
        for key in ["text", "mood", "problem", "selectedOption", "inventory", "budget"]
        if payload.get(key)
    ).strip()
    if not text:
        text = "作りたいものは分からない。Lチカの次に進みたい。"

    model_dir = Path(payload.get("modelDir") or DEFAULT_GRAPH_MODEL_DIR)
    checkpoint = model_dir / "best.pt"
    tokenizer = model_dir / "tokenizer.json"
    if not checkpoint.exists() or not tokenizer.exists():
        return {
            "available": False,
            "reason": f"checkpoint not found: {checkpoint}",
            "expectedCommand": "python -m ai_models.makergraph.train_project_graph_generator --device cuda --amp",
        }

    try:
        from ai_models.makergraph.generate_project_graph import ProjectGraphGenerator
    except Exception as exc:  # pragma: no cover
        return {
            "available": False,
            "reason": f"PyTorch graph generator import failed: {exc}",
            "expectedInstall": "pip install -r requirements-ml.txt",
        }

    try:
        generator = ProjectGraphGenerator(model_dir=model_dir, device=str(payload.get("device") or "auto"))
        result = generator.generate(text, max_new_tokens=int(payload.get("maxNewTokens") or 96))
        return {
            "available": True,
            "modelDir": str(model_dir),
            **result,
        }
    except Exception as exc:  # pragma: no cover
        return {
            "available": False,
            "reason": f"Project graph Transformer inference failed: {exc}",
            "modelDir": str(model_dir),
        }


def wirechecknet_inference(payload: dict[str, Any]) -> dict[str, Any]:
    model_dir = Path(payload.get("modelDir") or DEFAULT_WIRECHECK_MODEL_DIR)
    checkpoint = model_dir / "best.pt"
    if not checkpoint.exists():
        return {
            "available": False,
            "reason": f"checkpoint not found: {checkpoint}",
            "expectedCommand": "python -m ai_models.wirecheck.train_wirecheck --device cuda --amp",
        }

    try:
        from ai_models.wirecheck.infer_wirecheck import WireCheckInference
    except Exception as exc:  # pragma: no cover
        return {
            "available": False,
            "reason": f"WireCheckNet import failed: {exc}",
            "expectedInstall": "pip install -r requirements-ml.txt",
        }

    try:
        service = WireCheckInference(model_dir=model_dir, device=str(payload.get("device") or "auto"))
        threshold = float(payload.get("threshold") or 0.35)
        if payload.get("imageBase64"):
            result = service.predict_base64(str(payload["imageBase64"]), score_threshold=threshold)
        else:
            result = service.predict_synthetic(score_threshold=threshold)
        return {
            "available": True,
            **result,
        }
    except Exception as exc:  # pragma: no cover
        return {
            "available": False,
            "reason": f"WireCheckNet inference failed: {exc}",
            "modelDir": str(model_dir),
        }


def neural_bom_inference(payload: dict[str, Any]) -> dict[str, Any]:
    text = " ".join(
        str(payload.get(key, ""))
        for key in ["text", "project", "projectGraph", "inventory", "budget", "selectedOption"]
        if payload.get(key)
    ).strip()
    if not text:
        text = "水やり通知を作りたい。ESP32とLEDと抵抗は持っている。予算5000円。"

    model_dir = Path(payload.get("modelDir") or DEFAULT_BOM_MODEL_DIR)
    checkpoint = model_dir / "best.pt"
    if not checkpoint.exists():
        return {
            "available": False,
            "reason": f"checkpoint not found: {checkpoint}",
            "expectedCommand": "python -m ai_models.bom.train_bom_estimator --device cuda --amp",
        }

    try:
        from ai_models.bom.infer_bom import BOMInference
    except Exception as exc:  # pragma: no cover
        return {
            "available": False,
            "reason": f"BOM estimator import failed: {exc}",
            "expectedInstall": "pip install -r requirements-ml.txt",
        }

    try:
        service = BOMInference(model_dir=model_dir, device=str(payload.get("device") or "auto"))
        return service.predict(text, threshold=float(payload.get("threshold") or 0.42))
    except Exception as exc:  # pragma: no cover
        return {
            "available": False,
            "reason": f"BOM estimator inference failed: {exc}",
            "modelDir": str(model_dir),
        }
