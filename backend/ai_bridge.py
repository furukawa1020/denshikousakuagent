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
DEFAULT_SKILLREC_MODEL_DIR = ROOT / "runs" / "skillrec_transformer"
DEFAULT_NEURAL_AGENT_MODEL_DIR = ROOT / "runs" / "neural_agents"
DEFAULT_CIRCUIT_VALIDATOR_MODEL_DIR = ROOT / "runs" / "circuit_validator"
DEFAULT_INVENTORY_MATCHER_MODEL_DIR = ROOT / "runs" / "inventory_matcher"
DEFAULT_TUTORIAL_AGENT_MODEL_DIR = ROOT / "runs" / "tutorial_agent"
_SERVICE_CACHE: dict[tuple[str, str, str], Any] = {}


def cached_service(kind: str, model_dir: Path, device: str, factory: Any) -> Any:
    key = (kind, str(model_dir.resolve()), device)
    service = _SERVICE_CACHE.get(key)
    if service is None:
        service = factory()
        _SERVICE_CACHE[key] = service
    return service


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
        device = str(payload.get("device") or "auto")
        service = cached_service("intent", model_dir, device, lambda: MakerIntentInference(model_dir=model_dir, device=device))
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
        device = str(payload.get("device") or "auto")
        generator = cached_service("project_graph", model_dir, device, lambda: ProjectGraphGenerator(model_dir=model_dir, device=device))
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
        device = str(payload.get("device") or "auto")
        service = cached_service("wirecheck", model_dir, device, lambda: WireCheckInference(model_dir=model_dir, device=device))
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
        device = str(payload.get("device") or "auto")
        service = cached_service("bom", model_dir, device, lambda: BOMInference(model_dir=model_dir, device=device))
        return service.predict(text, threshold=float(payload.get("threshold") or 0.42))
    except Exception as exc:  # pragma: no cover
        return {
            "available": False,
            "reason": f"BOM estimator inference failed: {exc}",
            "modelDir": str(model_dir),
        }


def skillrec_inference(payload: dict[str, Any]) -> dict[str, Any]:
    model_dir = Path(payload.get("modelDir") or DEFAULT_SKILLREC_MODEL_DIR)
    checkpoint = model_dir / "best.pt"
    if not checkpoint.exists():
        return {
            "available": False,
            "reason": f"checkpoint not found: {checkpoint}",
            "expectedCommand": "python -m ai_models.skillrec.train_skillrec --device cuda --amp",
        }

    try:
        from ai_models.skillrec.infer_skillrec import SkillRecInference, default_events
    except Exception as exc:  # pragma: no cover
        return {
            "available": False,
            "reason": f"SkillRec import failed: {exc}",
            "expectedInstall": "pip install -r requirements-ml.txt",
        }

    try:
        events = payload.get("events")
        if not isinstance(events, list):
            events = default_events()
        device = str(payload.get("device") or "auto")
        service = cached_service("skillrec", model_dir, device, lambda: SkillRecInference(model_dir=model_dir, device=device))
        return service.predict(events)
    except Exception as exc:  # pragma: no cover
        return {
            "available": False,
            "reason": f"SkillRec inference failed: {exc}",
            "modelDir": str(model_dir),
        }


def neural_agent_inference(payload: dict[str, Any]) -> dict[str, Any]:
    model_dir = Path(payload.get("modelDir") or DEFAULT_NEURAL_AGENT_MODEL_DIR)
    checkpoint = model_dir / "best.pt"
    tokenizer = model_dir / "tokenizer.json"
    if not checkpoint.exists() or not tokenizer.exists():
        return {
            "available": False,
            "reason": f"checkpoint not found: {checkpoint}",
            "expectedCommand": "python -m ai_models.neural_agents.train_neural_agents --device cuda --amp",
        }

    try:
        from ai_models.neural_agents.infer_neural_agents import NeuralAgentInference
    except Exception as exc:  # pragma: no cover
        return {
            "available": False,
            "reason": f"NeuralAgent import failed: {exc}",
            "expectedInstall": "pip install -r requirements-ml.txt",
        }

    try:
        device = str(payload.get("device") or "auto")
        service = cached_service("neural_agent", model_dir, device, lambda: NeuralAgentInference(model_dir=model_dir, device=device))
        return service.predict(payload)
    except Exception as exc:  # pragma: no cover
        return {
            "available": False,
            "reason": f"NeuralAgent inference failed: {exc}",
            "modelDir": str(model_dir),
        }


def circuit_validator_inference(payload: dict[str, Any]) -> dict[str, Any]:
    model_dir = Path(payload.get("modelDir") or DEFAULT_CIRCUIT_VALIDATOR_MODEL_DIR)
    checkpoint = model_dir / "best.pt"
    tokenizer = model_dir / "tokenizer.json"
    if not checkpoint.exists() or not tokenizer.exists():
        return {
            "available": False,
            "reason": f"checkpoint not found: {checkpoint}",
            "expectedCommand": "python -m ai_models.circuit_validator.train_circuit_validator --device cuda --amp",
        }

    try:
        from ai_models.circuit_validator.infer_circuit_validator import CircuitValidatorInference
    except Exception as exc:  # pragma: no cover
        return {
            "available": False,
            "reason": f"CircuitValidator import failed: {exc}",
            "expectedInstall": "pip install -r requirements-ml.txt",
        }

    try:
        device = str(payload.get("device") or "auto")
        service = cached_service("circuit_validator", model_dir, device, lambda: CircuitValidatorInference(model_dir=model_dir, device=device))
        return service.predict(payload)
    except Exception as exc:  # pragma: no cover
        return {
            "available": False,
            "reason": f"CircuitValidator inference failed: {exc}",
            "modelDir": str(model_dir),
        }


def inventory_matcher_inference(payload: dict[str, Any]) -> dict[str, Any]:
    model_dir = Path(payload.get("modelDir") or DEFAULT_INVENTORY_MATCHER_MODEL_DIR)
    checkpoint = model_dir / "best.pt"
    tokenizer = model_dir / "tokenizer.json"
    if not checkpoint.exists() or not tokenizer.exists():
        return {
            "available": False,
            "reason": f"checkpoint not found: {checkpoint}",
            "expectedCommand": "python -m ai_models.inventory_matcher.train_inventory_matcher --device cuda --amp",
        }

    try:
        from ai_models.inventory_matcher.infer_inventory_matcher import InventoryMatcherInference
    except Exception as exc:  # pragma: no cover
        return {
            "available": False,
            "reason": f"InventoryMatcher import failed: {exc}",
            "expectedInstall": "pip install -r requirements-ml.txt",
        }

    try:
        device = str(payload.get("device") or "auto")
        service = cached_service("inventory_matcher", model_dir, device, lambda: InventoryMatcherInference(model_dir=model_dir, device=device))
        return service.predict(payload)
    except Exception as exc:  # pragma: no cover
        return {
            "available": False,
            "reason": f"InventoryMatcher inference failed: {exc}",
            "modelDir": str(model_dir),
        }


def tutorial_agent_inference(payload: dict[str, Any]) -> dict[str, Any]:
    model_dir = Path(payload.get("modelDir") or DEFAULT_TUTORIAL_AGENT_MODEL_DIR)
    checkpoint = model_dir / "best.pt"
    tokenizer = model_dir / "tokenizer.json"
    if not checkpoint.exists() or not tokenizer.exists():
        return {
            "available": False,
            "reason": f"checkpoint not found: {checkpoint}",
            "expectedCommand": "python -m ai_models.tutorial_agent.train_tutorial_agent --device cuda --amp",
        }

    try:
        from ai_models.tutorial_agent.infer_tutorial_agent import TutorialAgentInference
    except Exception as exc:  # pragma: no cover
        return {
            "available": False,
            "reason": f"TutorialAgent import failed: {exc}",
            "expectedInstall": "pip install -r requirements-ml.txt",
        }

    try:
        device = str(payload.get("device") or "auto")
        service = cached_service("tutorial_agent", model_dir, device, lambda: TutorialAgentInference(model_dir=model_dir, device=device))
        return service.predict(payload)
    except Exception as exc:  # pragma: no cover
        return {
            "available": False,
            "reason": f"TutorialAgent inference failed: {exc}",
            "modelDir": str(model_dir),
        }
