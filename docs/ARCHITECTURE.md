# Architecture diagrams

## 1. One model, three decisions

The served model is an ensemble of two members in one ONNX graph: MiniLM-L6-H384
(int8) and Ettin-17m (fp32). Each member has the structure below, and one
`session.run` computes both. Inside each member, every request runs through the
same encoder and the same experts. The decision type only changes how the input
is framed, which task embedding the gate sees, and which output head's logit is
read. The two members' temperature-scaled logits are averaged, then a final
temperature is applied (`decide/runtime.py` `EnsembleBackend`).

```mermaid
flowchart LR
    subgraph IN["Input framing (decide/inputs.py)"]
        direction TB
        A1["route<br/>(subject, body)"]
        A2["approve<br/>(tier + amount + subject, body)<br/>+ tabular features"]
        A3["grounded<br/>(kb_article, draft_reply)"]
    end
    subgraph SH["Shared, inside each member"]
        direction TB
        ENC["encoder<br/>(MiniLM-L6-H384 or Ettin-17m)"] --> POOL["mean pool"]
        POOL --> GATE["gate"]
        TASK["task embedding"] --> GATE
        POOL --> EXP["experts 1..4"]
        TAB["tab MLP"] --> EXP
        TASK --> EXP
        GATE --> MIX["weighted sum"]
        EXP --> MIX
    end
    subgraph HD["Per decision"]
        direction TB
        H1["route: Linear 256 to 5"]
        H2["approve: Linear 256 to 1"]
        H3["grounded: Linear 256 to 1"]
    end
    IN --> ENC
    IN --> TAB
    MIX --> H1 & H2 & H3
```

Adding a fourth decision = one task-embedding row + one Linear head + its
training examples. Nothing in the blue path changes, and latency stays the same.

## 2. Request lifecycle

```mermaid
sequenceDiagram
    participant C as Caller
    participant A as app.py (FastAPI)
    participant M as model.py (policy)
    participant R as runtime.py (ONNX Runtime, CPU)
    C->>A: POST /v1/decide {decision, state}
    A->>M: route / approve / grounded(state)
    M->>R: frame input, tokenize (Rust tokenizers)
    R->>R: one session.run: both members, 4 threads
    R-->>M: each member's logit for the requested head
    M->>M: member temperatures, average, final temperature, softmax / sigmoid
    M->>M: commit value, set abstain flag
    M-->>A: (value, probs or p_true, abstain)
    A-->>C: typed JSON + latency_ms (about 14 ms p50 on a busy laptop)
```

## 3. Approve and abstention logic

```mermaid
flowchart TD
    S["calibrated p_true"] --> T{"p_true > 0.75?<br/>(3:1 cost)"}
    T -->|yes| Y["value = true (auto-approve)"]
    T -->|no| N["value = false (hold)"]
    S --> C{"0.15 < p < 0.79?<br/>(lopsided: wrong approvals cost 3x)"}
    C -->|yes| AB["abstain = true<br/>(send to a human)"]
    C -->|no| OK["abstain = false"]
```

The thresholds come from `artifacts/policy.json`, set on the training
holdout so about 85% of requests are answered (target 80%). `route` abstains
when its top-queue probability is under 0.892; `grounded` when
max(p, 1-p) is under 0.9957.

What the approve head learned from the labels:

```mermaid
flowchart LR
    T["tier"] --> CUT{"amount above tier cutoff?<br/>free ~$30 / pro ~$120 / enterprise ~$500"}
    CUT -->|yes| HOLD["hold"]
    CUT -->|no| FLAG{"red flag in text?<br/>repeat claim, new account,<br/>prior dispute, asks for exception"}
    FLAG -->|yes| HOLD
    FLAG -->|no| LIKELY["likely approve (~80%)<br/>rest: hidden account history"]
```

## 4. Training pipeline

```mermaid
flowchart LR
    D["train.jsonl<br/>6,500"] --> SP{"split"}
    SP -->|5,500| TR["multi-task fine-tune, per member<br/>Kaggle T4, 5 epochs, seed 13,<br/>one task per batch"]
    SP -->|1,000 holdout| CAL["fit_ensemble.py: member + final<br/>temperatures, abstention thresholds"]
    TR --> CAL
    TR --> EX["export ONNX: MiniLM int8,<br/>Ettin fp32; merge into one graph"]
    CAL --> ART["artifacts/<br/>decide.onnx 92 MB<br/>ensemble.json, calibration.json, policy.json"]
    EX --> ART
    CV["4-fold CV<br/>(evaluation only)"] -.->|"chose epochs, approve band"| TR
    DEV["dev.jsonl 1,500<br/>(never trained on)"] -.->|"reporting"| SC["selfcheck.py<br/>85.44"]
    ART --> SC
```
