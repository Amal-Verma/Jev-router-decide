"""How each decision type is presented to the shared encoder.

The three questions compare different things, so each gets its own text pair:
    route     (subject, body)                        what is the ticket about
    approve   ("<tier> plan, $<amount>. subject", body) + tier/amount/flag features
    grounded  (kb_article, draft_reply)              NLI-style premise/hypothesis
All three go through the same encoder; `tab` carries the structured features
(zero-filled where a decision does not use them).
"""
import numpy as np

from .tabular import N_APPROVE_FEATS, approve_features, grounded_features

QUEUES = ["billing", "technical", "account", "shipping", "general"]
N_TAB = N_APPROVE_FEATS + 3


def encode_inputs(task, s):
    subject, body = s.get("subject") or "", s.get("body") or ""
    tab = np.zeros(N_TAB, dtype=np.float32)
    if task == "route":
        return subject, body, tab
    if task == "approve":
        tab[:N_APPROVE_FEATS] = approve_features(s)
        tier = (s.get("customer_tier") or "free").lower()
        return f"{tier} plan, amount ${float(s.get('amount') or 0):.2f}. {subject}", body, tab
    tab[N_APPROVE_FEATS:] = grounded_features(s)
    return s.get("kb_article") or "", s.get("draft_reply") or "", tab
