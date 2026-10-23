# API

`POST /v1/decide`. One decision per call, the whole state object every time.

## Request

```json
{ "request_id": "abc-123",
  "decision": "route",
  "state": { "subject": "...", "body": "...", "customer_tier": "pro",
             "amount": 34.44, "draft_reply": "...", "kb_article": "..." } }
```

| decision | fields it reads |
|---|---|
| `route` | subject, body |
| `approve` | subject, body, customer_tier, amount |
| `grounded` | draft_reply, kb_article |

Missing or `null` fields are treated as empty / zero, never rejected.

## Responses

Choice (`route`): `probs` covers all five queues and sums to 1;
`confidence == probs[value]`.

```json
{ "request_id": "abc-123", "decision": "route", "type": "choice",
  "value": "billing",
  "probs": {"billing": 0.91, "technical": 0.03, "account": 0.03, "shipping": 0.01, "general": 0.02},
  "confidence": 0.91, "abstain": false, "latency_ms": 11.2 }
```

Binary (`approve`, `grounded`): `p_true` is the calibrated probability of "yes".

```json
{ "request_id": "abc-123", "decision": "approve", "type": "binary",
  "value": false, "p_true": 0.31, "abstain": true, "latency_ms": 12.9 }
```

- `approve.value` is `true` only when `p_true > 0.75` (a wrong approval costs 3x a wrong hold).
- `grounded.value` is `p_true > 0.5`.
- `abstain: true` means "send to a human". The probabilities are still honest.

`GET /health` returns `{"status": "ok", "loaded": true}` once the model is in memory.
