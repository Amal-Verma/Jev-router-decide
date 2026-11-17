# Integrity Incentive & Reputation Engine (Decide v2)

> **Feature Specification & Implementation Guide**  
> Capitalizing on calibrated deep learning confidence and grounding scores to reward truthful submissions and penalize malicious or hallucinated activity.

---

## 1. Executive Summary & Motivation

In automated decision-support and workflow automation platforms, compute and human verification resources are finite. When users (or autonomous agents) submit tickets, drafted responses, or refund requests, platform abuse takes several forms:
1. **Hallucinated or Fabricated Replies:** Agents or support personnel submitting drafted responses containing false promises not grounded in official documentation (`grounded` decision failure).
2. **Fraudulent / Speculative Refund Inquiries:** Submitting inflated amounts or spam approval requests hoping to bypass review thresholds (`approve` decision failure).
3. **Gibberish / Malicious Spam:** Submitting noisy or unclassifiable text that wastes CPU inference cycles and pollutes support queues (`route` decision failure).

The **Decide** engine already computes **rigorously calibrated probabilities** (via temperature-scaled Multi-Task MiniLM + Ettin ensemble) and deliberate **abstention policies**. This feature turns those passive predictive outputs into an **active incentive and gatekeeping mechanism**:
- **Truthful Submissions** boost the user's **Trust Reputation Score ($R_u$)**, unlocking higher API quotas, faster queue priority, and automated fast-track approvals.
- **Deceptive or Hallucinated Submissions** incur immediate score penalties, dynamic rate-limiting, compute throttling, and mandatory manual escrow quarantine.

---

## 2. Theoretical Formulation & Integrity Metrics

The system derives an **Integrity Signal ($I_t \in [-1.0, +1.0]$)** directly from the decision type and the model's calibrated confidence.

```
       +---------------------------------------------------------+
       |   Inference Output: (value, p_true / probs, abstain)   |
       +---------------------------------------------------------+
                                    |
            +-----------------------+-----------------------+
            |                       |                       |
     [route decision]      [approve decision]     [grounded decision]
            |                       |                       |
     Confidence c           P(true) vs Amount        P(true) Groundedness
            |                       |                       |
            +-----------------------+-----------------------+
                                    |
                                    v
                     Normalized Integrity Signal:
                          I_t in [-1.0, +1.0]
                                    |
                                    v
                 Bayesian Reputation Update: R_u in [0, 100]
```

### 2.1. Decision-Specific Integrity Signals

#### A. Grounding Integrity (`grounded`)
Grounding measures factual alignment between a draft reply and knowledge-base article. The model outputs calibrated $p_{\text{true}} \in [0, 1]$.
$$\Delta_{\text{grounded}} = 2 \cdot (p_{\text{true}} - 0.5) \in [-1.0, +1.0]$$
- If $p_{\text{true}} \ge 0.90$ (strongly supported): $I_t \approx +0.80$ to $+1.0$ (High Reward).
- If $0.40 \le p_{\text{true}} \le 0.60$ (abstain / indeterminate): $I_t = 0.0$ (Neutral, deferred to human without penalty).
- If $p_{\text{true}} \le 0.15$ (verifiably false / hallucinated claim): $I_t \approx -0.70$ to $-1.0$ (Severe Penalty).

#### B. Approval Integrity (`approve`)
Approval evaluates whether a financial or policy claim is legitimate. A submission is evaluated against both the calibrated probability $p_{\text{true}}$ and the risk magnitude ($\text{amount}$):
$$I_{\text{approve}} = \begin{cases} 
+(p_{\text{true}} - \tau_{\text{approve}}) & \text{if } p_{\text{true}} \ge \tau_{\text{approve}} \\
-\lambda_{\text{risk}} \cdot (\tau_{\text{approve}} - p_{\text{true}}) \cdot \log_{10}(1 + \text{amount}) & \text{if } p_{\text{true}} < \tau_{\text{approve}}
\end{cases}$$
*Where $\tau_{\text{approve}} = 0.75$ and $\lambda_{\text{risk}} = 1.2$. Frivolous high-value attempts that fail verification result in harsher penalties.*

#### C. Routing Integrity (`route`)
Routing integrity evaluates text coherence and predictability over the 5 ticket queues. Let $c = \max_k P(y = k \mid x)$ be the top-1 confidence:
$$I_{\text{route}} = \begin{cases}
+1.0 \cdot (c - 0.5) / 0.5 & \text{if } c > 0.60 \text{ and not abstain} \\
0.0 & \text{if abstain (ambiguous ticket)} \\
-0.80 & \text{if unclassifiable gibberish / out-of-distribution noise}
\end{cases}$$

---

## 3. Reputation Dynamics & User Tiers

### 3.1. Exponential Moving Average with Asymmetric Slopes
Trust is slow to build but quick to lose. We update user score $R_u \in [0, 100]$ (initial default $R_0 = 50.0$):

$$R_u^{(t)} = R_u^{(t-1)} + \Delta R_t$$

$$\Delta R_t = \begin{cases} 
\eta_{\text{reward}} \cdot I_t \cdot \left(1 - \frac{R_u^{(t-1)}}{100}\right) & \text{if } I_t > 0 \\
\eta_{\text{penalty}} \cdot I_t \cdot \left(\frac{R_u^{(t-1)}}{100}\right) & \text{if } I_t < 0
\end{cases}$$

- **Learning rates:** $\eta_{\text{reward}} = 2.0$, $\eta_{\text{penalty}} = 5.0$ (Asymmetric penalty factor $2.5\times$).
- **Boundary caps:** Clamped strictly to $[0.0, 100.0]$.

### 3.2. Platform Reputation Tiers

| Tier Name | Score Range ($R_u$) | Platform Treatment & Privileges | Rate Limit (req/min) | Human Review |
|---|---|---|---|---|
| **Trusted Champion** | $85.0 - 100.0$ | **Fast-track lane:** Instant automated routing & execution, bypass cold caches, maximum quota multiplier ($3\times$). | 600 | 2% random audit |
| **Good Standing** | $60.0 - 84.9$ | **Standard lane:** Normal inference quotas, standard batching. | 180 | 10% spot check |
| **Probationary** | $40.0 - 59.9$ | **Monitored lane:** Standard latency, no priority access, warning banners in UI. | 60 | 40% verification |
| **Quarantine (Tarpit)** | $15.0 - 39.9$ | **Throttled lane:** Forced 1.5s delay to disincentivize bot spam, all approvals held in escrow. | 15 | 100% human escrow |
| **Suspended / Barred** | $0.0 - 14.9$ | **Blocked lane:** API access locked; submissions rejected with HTTP 403 `INTEGRITY_VIOLATION`. | 0 | Blocked |

---

## 4. API Specification & Request Lifecycle

### 4.1. Request Header Additions
Clients must provide a user/agent identity token in `POST /v1/decide`:
```http
POST /v1/decide HTTP/1.1
Host: 127.0.0.1:8000
Content-Type: application/json
X-User-ID: agent_usr_98124
X-API-Key: sec_live_abc123
```

### 4.2. Response Payload with Reputation Feedback
The engine returns the decision along with real-time reputation updates and resource quotas:

```json
{
  "request_id": "req-98234-x",
  "decision": "grounded",
  "type": "binary",
  "value": true,
  "p_true": 0.9842,
  "abstain": false,
  "latency_ms": 11.23,
  "integrity": {
    "signal": "+0.968",
    "evaluation": "HIGH_CONFIDENCE_GROUNDED",
    "reputation_delta": "+1.84",
    "current_reputation": 78.50,
    "user_tier": "Good Standing",
    "quota_remaining": 174,
    "rate_limit_reset_sec": 48
  }
}
```

If a penalized user submits an ungrounded or fraudulent claim:
```json
{
  "request_id": "req-98235-y",
  "decision": "grounded",
  "type": "binary",
  "value": false,
  "p_true": 0.0412,
  "abstain": false,
  "latency_ms": 10.95,
  "integrity": {
    "signal": "-0.918",
    "evaluation": "UNGROUNDED_FABRICATION_DETECTED",
    "reputation_delta": "-4.12",
    "current_reputation": 42.10,
    "user_tier": "Probationary",
    "warning": "Draft claims contradict source KB. Continued violations will lock account.",
    "quota_remaining": 52
  }
}
```

---

## 5. Software Architecture & Implementation

### 5.1. Module Layout
The incentive engine is cleanly decoupled from inference:
```
decide/
├── runtime.py                 # ONNX multi-task inference
├── inputs.py                  # Tokenization & framing
├── reputation/                # [NEW MODULE]
│   ├── __init__.py
│   ├── engine.py              # Mathematical integrity score calculator
│   ├── ledger.py              # SQLite / Redis persistent reputation store
│   ├── policy.py              # Rate-limit rules & tier thresholds
│   └── middleware.py          # FastAPI request interceptor & throttler
```

### 5.2. Core Implementation Blueprint

```python
# decide/reputation/engine.py
from dataclasses import dataclass
from typing import Dict, Any, Tuple

@dataclass
class IntegrityResult:
    signal: float
    reputation_delta: float
    new_reputation: float
    tier: str
    action: str

class ReputationEngine:
    def __init__(self, eta_reward=2.0, eta_penalty=5.0):
        self.eta_reward = eta_reward
        self.eta_penalty = eta_penalty

    def calculate_signal(self, decision: str, output: Dict[str, Any], state: Dict[str, Any]) -> float:
        if output.get("abstain", False):
            return 0.0  # Safe abstention is never penalized

        if decision == "grounded":
            p_true = output["p_true"]
            return float(2.0 * (p_true - 0.5))

        elif decision == "approve":
            p_true = output["p_true"]
            amount = float(state.get("amount", 0.0))
            if p_true >= 0.75:
                return float((p_true - 0.75) / 0.25)
            else:
                risk_factor = 1.0 + (0.2 * (amount / 100.0) if amount > 0 else 0.0)
                return float(-1.0 * (0.75 - p_true) / 0.75 * min(risk_factor, 2.0))

        elif decision == "route":
            conf = output["confidence"]
            return float((conf - 0.5) / 0.5) if conf > 0.5 else -0.5

        return 0.0

    def update_reputation(self, current_rep: float, signal: float) -> Tuple[float, float, str]:
        if signal > 0:
            delta = self.eta_reward * signal * (1.0 - current_rep / 100.0)
        else:
            delta = self.eta_penalty * signal * (current_rep / 100.0)

        new_rep = max(0.0, min(100.0, current_rep + delta))

        if new_rep >= 85.0:
            tier = "Trusted Champion"
        elif new_rep >= 60.0:
            tier = "Good Standing"
        elif new_rep >= 40.0:
            tier = "Probationary"
        elif new_rep >= 15.0:
            tier = "Quarantine"
        else:
            tier = "Suspended"

        return round(delta, 2), round(new_rep, 2), tier
```

---

## 6. Security, Gaming Resistance & Auditing

1. **Sybil & New Account Cold-Start:** New accounts start with $R_0 = 50.0$ and are restricted to 30 requests/hour until at least 50 validated decisions are logged.
2. **Adversarial Noise Resistance:** Temperature calibration guarantees that extreme overconfident errors are minimized (ECE $\approx 0.016$).
3. **Dual-Key Human Auditing Loop:** If a quarantined user disputes an integrity penalty, an auditor review (`POST /v1/reputation/dispute`) can reverse penalties with compensation credits if the model erred.
4. **Anti-Collusion Decay:** Inactivity causes a gradual regression toward median ($50.0$) at $0.5$ points/day to prevent stale bot hoarding.

---

## 7. Educational & Evaluation Value for DL Viva

When presenting this feature to professors and examiners, highlight these key deep learning connections:
- **Direct Utilization of Calibration:** Unlike vanilla models where raw softmax outputs are uncalibrated and cannot be trusted, our Temperature-Scaled and Holdout-Calibrated outputs represent true empirical accuracy, making mathematical reward calculations justifiable.
- **Selective Classification Integration:** The abstention boundary is treated as a neutral safety zone—users are rewarded for certainty and never penalized for genuine ambiguity.
- **Closed-Loop Alignment:** Aligning user incentives with model objective functions ensures platform resource longevity and clean operational data.
