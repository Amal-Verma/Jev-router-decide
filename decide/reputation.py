# decide/reputation.py
import math
import time
from dataclasses import dataclass, field
from typing import Dict, Any, Tuple, Optional

TIERS = {
    "CHAMPION": {"min_score": 85.0, "name": "Trusted Champion", "rate_limit": 600, "tarpit_delay_sec": 0.0},
    "STANDARD": {"min_score": 60.0, "name": "Good Standing",    "rate_limit": 180, "tarpit_delay_sec": 0.0},
    "PROBATION": {"min_score": 40.0, "name": "Probationary",      "rate_limit": 60,  "tarpit_delay_sec": 0.0},
    "QUARANTINE":{"min_score": 15.0, "name": "Quarantine",        "rate_limit": 15,  "tarpit_delay_sec": 1.5},
    "SUSPENDED": {"min_score": 0.0,  "name": "Suspended",         "rate_limit": 0,   "tarpit_delay_sec": 0.0},
}

@dataclass
class UserProfile:
    user_id: str
    reputation: float = 50.0
    tier: str = "Probationary"
    total_requests: int = 0
    truthful_count: int = 0
    violation_count: int = 0
    quota_remaining: int = 60
    last_reset: float = field(default_factory=time.time)

class ReputationLedger:
    def __init__(self, eta_reward: float = 2.0, eta_penalty: float = 5.0):
        self.eta_reward = eta_reward
        self.eta_penalty = eta_penalty
        self.users: Dict[str, UserProfile] = {}

    def get_or_create(self, user_id: str) -> UserProfile:
        if user_id not in self.users:
            self.users[user_id] = UserProfile(user_id=user_id)
        return self.users[user_id]

    def calculate_signal(self, decision: str, output: Dict[str, Any], state: Dict[str, Any]) -> Tuple[float, str]:
        """Derive normalized integrity signal in [-1.0, 1.0] from model output."""
        if output.get("abstain", False):
            return 0.0, "ABSTAIN_NEUTRAL"  # Safe abstention is never penalized

        if decision == "grounded":
            p_true = float(output.get("p_true", 0.5))
            signal = 2.0 * (p_true - 0.5)
            evaluation = "GROUNDED_TRUTHFUL" if signal > 0.3 else ("UNGROUNDED_FABRICATION" if signal < -0.3 else "BORDERLINE")
            return max(-1.0, min(1.0, signal)), evaluation

        elif decision == "approve":
            p_true = float(output.get("p_true", 0.5))
            amount = float(state.get("amount", 0.0))
            if p_true >= 0.75:
                signal = (p_true - 0.75) / 0.25
                evaluation = "LEGITIMATE_APPROVAL"
            else:
                risk_multiplier = 1.0 + (0.2 * math.log10(1 + max(0.0, amount)))
                signal = -1.0 * ((0.75 - p_true) / 0.75) * risk_multiplier
                evaluation = "QUESTIONABLE_FINANCIAL_CLAIM"
            return max(-1.0, min(1.0, signal)), evaluation

        elif decision == "route":
            conf = float(output.get("confidence", 0.5))
            signal = (conf - 0.5) / 0.5 if conf > 0.5 else -0.5
            evaluation = "HIGH_CONFIDENCE_ROUTING" if signal > 0 else "AMBIGUOUS_NOISY_INPUT"
            return max(-1.0, min(1.0, signal)), evaluation

        return 0.0, "UNKNOWN"

    def record_decision(self, user_id: str, decision: str, output: Dict[str, Any], state: Dict[str, Any]) -> Dict[str, Any]:
        user = self.get_or_create(user_id)
        signal, evaluation = self.calculate_signal(decision, output, state)

        # Asymmetric update
        if signal > 0:
            delta = self.eta_reward * signal * (1.0 - user.reputation / 100.0)
            user.truthful_count += 1
        elif signal < 0:
            delta = self.eta_penalty * signal * (user.reputation / 100.0)
            user.violation_count += 1
        else:
            delta = 0.0

        user.reputation = max(0.0, min(100.0, user.reputation + delta))
        user.total_requests += 1

        # Determine Tier
        for key, conf in TIERS.items():
            if user.reputation >= conf["min_score"]:
                user.tier = conf["name"]
                break

        return {
            "signal": round(signal, 3),
            "evaluation": evaluation,
            "reputation_delta": round(delta, 2),
            "current_reputation": round(user.reputation, 2),
            "user_tier": user.tier,
            "truthful_count": user.truthful_count,
            "violation_count": user.violation_count,
        }