"""How each decision type is presented to the shared encoder.

The three questions compare different things, so each gets its own text pair:
    route     (subject, body)                        what is the ticket about
    approve   ("<tier> plan, $<amount>. subject", body) + tier/amount/flag features
    grounded  (kb_article, draft_reply)              NLI-style premise/hypothesis
All three go through the same encoder; `tab` carries the structured features
(zero-filled where a decision does not use them).
"""
import os

import numpy as np

from .features import NUM_WORDS, QTY_RE, UNIT_DAYS
from .tabular import N_APPROVE_FEATS, approve_features, grounded_features

# Optional preprocessing (experiment): rewrite every duration in the draft and
# the article into one canonical form, so "two weeks" and "14 days" read alike.
CANON = os.environ.get("DECIDE_CANON") == "1"


FRACTIONS = [(r"\ba quarter of an hour\b", "15 minutes"), (r"\bhalf an hour\b", "30 minutes"),
             (r"\bthree quarters of an hour\b", "45 minutes")]


def canonical_durations(text):
    import re
    for pat, rep in FRACTIONS:
        text = re.sub(pat, rep, text, flags=re.I)

    def rep(m):
        whole = m.group(0)
        if re.search(r"\d\s*-\s*\d", whole) or re.search(r"\d\s+\d", whole):
            return whole                       # ranges like "3-5 business days" stay as written
        n, u = m.group(1), m.group(2)
        n = NUM_WORDS.get(n.lower()) if not n[0].isdigit() else float(n)
        if n is None:
            return whole
        u = u.lower()
        u = u.rstrip("s") if u != "business days" else "business day"
        days = float(n) * UNIT_DAYS.get(u, 1)
        if days < 1:
            return f"{days * 1440:g} minutes"
        return f"{days:g} {'business ' if u == 'business day' else ''}days"
    return QTY_RE.sub(rep, text)


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
    kb, draft = s.get("kb_article") or "", s.get("draft_reply") or ""
    if CANON:
        kb, draft = canonical_durations(kb), canonical_durations(draft)
    tab[N_APPROVE_FEATS:] = grounded_features(dict(s, kb_article=kb, draft_reply=draft))
    return kb, draft, tab
