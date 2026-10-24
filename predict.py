"""Answer a ticket from the command line, without starting a server.

    python predict.py --model ensemble --decision approve --ticket ticket.json
    python predict.py --model minilm --ticket ticket.json          # all three decisions
    echo '{"subject": "...", "body": "...", "customer_tier": "pro", "amount": 12}' | python predict.py --model ettin

The ticket is the request's "state" object (subject, body, customer_tier,
amount, draft_reply, kb_article); missing fields are treated as empty.
Output: one JSON line per decision, in the same shape as the API response.
"""
import argparse
import json
import os
import sys
import time

from model import QUEUES, DecisionModel

MODELS = {"ensemble": "artifacts", "minilm": "artifacts/minilm", "ettin": "artifacts/ettin"}
DEFAULTS = {"subject": "", "body": "", "customer_tier": "free", "amount": 0.0, "draft_reply": "", "kb_article": ""}


def answer(m, decision, state):
    t0 = time.perf_counter()
    if decision == "route":
        value, probs, conf, ab = m.route(state)
        out = {"decision": "route", "type": "choice", "value": value,
               "probs": {q: round(float(probs[q]), 4) for q in QUEUES}, "confidence": round(float(conf), 4), "abstain": bool(ab)}
    else:
        value, p, ab = (m.approve if decision == "approve" else m.grounded)(state)
        out = {"decision": decision, "type": "binary", "value": bool(value), "p_true": round(float(p), 4), "abstain": bool(ab)}
    out["latency_ms"] = round((time.perf_counter() - t0) * 1000, 2)
    return out


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--model", choices=MODELS, default="ensemble")
    ap.add_argument("--decision", choices=["route", "approve", "grounded", "all"], default="all")
    ap.add_argument("--ticket", help="JSON file with the ticket state (default: read stdin)")
    args = ap.parse_args()
    raw = json.load(open(args.ticket)) if args.ticket else json.load(sys.stdin)
    state = dict(DEFAULTS, **{k: v for k, v in (raw.get("state", raw)).items() if v is not None})
    m = DecisionModel(os.path.join(os.path.dirname(os.path.abspath(__file__)), MODELS[args.model]))
    for d in (["route", "approve", "grounded"] if args.decision == "all" else [args.decision]):
        print(json.dumps(answer(m, d, state)))


if __name__ == "__main__":
    main()
