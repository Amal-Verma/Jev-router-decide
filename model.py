import json
from pathlib import Path
from typing import Any, Dict, List, Tuple

QUEUES: List[str] = ["billing", "technical", "account", "shipping", "general"]
APPROVE_THRESHOLD = 0.75


class DecisionModel:
    def __init__(self, artifacts_dir: str = "artifacts") -> None:
        directory = Path(artifacts_dir)
        if not directory.is_absolute():
            directory = Path(__file__).resolve().parent / directory
        directory = directory.resolve()
        self.artifacts_dir = str(directory)
        onnx_path = directory / "decide.onnx"
        if not onnx_path.is_file():
            raise FileNotFoundError(f"Missing trained model: {onnx_path}. Select a bundled artifacts directory.")
        ensemble = (directory / "ensemble.json").is_file()
        if ensemble:
            from decide.runtime import EnsembleBackend
            self.backend = EnsembleBackend(self.artifacts_dir)
        else:
            from decide.runtime import EncoderBackend
            self.backend = EncoderBackend(self.artifacts_dir)
        self.policy = {
            "route_min_conf": 0.5,
            "approve_min_conf": 0.65,
            "grounded_min_conf": 0.65,
            "approve_threshold": APPROVE_THRESHOLD,
        }
        policy_path = directory / "policy.json"
        if policy_path.is_file():
            self.policy.update(json.loads(policy_path.read_text(encoding="utf-8")))
        model_name = "ensemble" if ensemble else directory.name if directory.name in {"minilm", "ettin"} else "encoder"
        self.metadata = {
            "model_name": model_name,
            "name": {"ensemble": "MiniLM + Ettin CPU ensemble", "minilm": "MiniLM + MoE", "ettin": "Ettin-17m + MoE"}.get(model_name, "Shared encoder"),
            "architecture": "One ONNX graph; shared encoders and experts across all three decisions",
            "artifact_bytes": onnx_path.stat().st_size,
            "thresholds": self.policy.copy(),
        }

    @staticmethod
    def has_ticket(state: Dict[str, Any]) -> bool:
        return bool(str(state.get("subject") or "").strip() or str(state.get("body") or "").strip())

    def route(self, state: Dict[str, Any]) -> Tuple[str, Dict[str, float], float, bool]:
        if not self.has_ticket(state):
            probs = {queue: 1.0 / len(QUEUES) for queue in QUEUES}
            return QUEUES[0], probs, probs[QUEUES[0]], True
        p = self.backend.route_probs(state)
        probs = {queue: float(p[i]) for i, queue in enumerate(QUEUES)}
        value = max(probs, key=probs.get)
        confidence = probs[value]
        return value, probs, confidence, confidence < self.policy["route_min_conf"]

    def approve(self, state: Dict[str, Any]) -> Tuple[bool, float, bool]:
        if not self.has_ticket(state) or float(state.get("amount") or 0) < 0:
            return False, 0.5, True
        probability = float(self.backend.approve_prob(state))
        value = probability > self.policy["approve_threshold"]
        lower, upper = self.policy.get("approve_abstain_band", (None, None))
        if lower is None:
            return value, probability, max(probability, 1.0 - probability) < self.policy["approve_min_conf"]
        return value, probability, lower < probability < upper

    def grounded(self, state: Dict[str, Any]) -> Tuple[bool, float, bool]:
        if not str(state.get("draft_reply") or "").strip() or not str(state.get("kb_article") or "").strip():
            return False, 0.5, True
        probability = float(self.backend.grounded_prob(state))
        return probability > 0.5, probability, max(probability, 1.0 - probability) < self.policy["grounded_min_conf"]
