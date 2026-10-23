"""
DecisionModel: one shared model, three typed decisions.

Loads artifacts/decide.onnx (the shared multi-task encoder, see README) when
present, otherwise falls back to the classical baseline in artifacts/baseline.pkl.
The three methods keep the starter's signatures and return shapes.
"""

import json
import os
from typing import Any, Dict, List, Tuple

QUEUES: List[str] = ["billing", "technical", "account", "shipping", "general"]

# Wrongly approving costs 3x wrongly holding, so approve only when p > 3/(3+1).
APPROVE_THRESHOLD = 0.75


class DecisionModel:
    def __init__(self, artifacts_dir: str = "artifacts") -> None:
        self.artifacts_dir = artifacts_dir
        onnx_path = os.path.join(artifacts_dir, "decide.onnx")
        if os.path.exists(onnx_path):
            from decide.runtime import EncoderBackend
            self.backend = EncoderBackend(artifacts_dir)
        else:
            from decide.baseline_runtime import BaselineBackend
            self.backend = BaselineBackend(artifacts_dir)
        policy_path = os.path.join(artifacts_dir, "policy.json")
        self.policy = {"route_min_conf": 0.5, "approve_min_conf": 0.65, "grounded_min_conf": 0.65,
                       "approve_threshold": APPROVE_THRESHOLD}
        if os.path.exists(policy_path):
            with open(policy_path, encoding="utf-8") as fh:
                self.policy.update(json.load(fh))

    # -- routing ------------------------------------------------------------
    def route(self, state: Dict[str, Any]) -> Tuple[str, Dict[str, float], float, bool]:
        p = self.backend.route_probs(state)
        probs = {q: float(p[i]) for i, q in enumerate(QUEUES)}
        value = max(probs, key=probs.get)
        conf = probs[value]
        return value, probs, conf, conf < self.policy["route_min_conf"]

    # -- approval -----------------------------------------------------------
    def approve(self, state: Dict[str, Any]) -> Tuple[bool, float, bool]:
        p = float(self.backend.approve_prob(state))
        value = p > self.policy["approve_threshold"]
        return value, p, max(p, 1.0 - p) < self.policy["approve_min_conf"]

    # -- grounding ----------------------------------------------------------
    def grounded(self, state: Dict[str, Any]) -> Tuple[bool, float, bool]:
        p = float(self.backend.grounded_prob(state))
        return p > 0.5, p, max(p, 1.0 - p) < self.policy["grounded_min_conf"]
