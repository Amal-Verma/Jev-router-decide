"""Structured features for the approve decision (tier x amount x red flags).

The approval policy is mostly an amount threshold that depends on the tier,
overridden by risk signals in the ticket text. These features let a small head
learn sharp thresholds that a text encoder alone handles poorly.
"""
import math

import numpy as np

from .features import quantities

TIERS = ["free", "pro", "enterprise"]
# Thresholds spanning the region where the label flips for each tier.
CUTS = [5, 10, 15, 20, 25, 30, 40, 50, 75, 100, 110, 120, 150, 200, 300, 400, 500, 750, 1000]
RED_FLAGS = [
    "raised something similar", "signed up recently", "dispute", "outside your usual policy",
    "exception", "again", "second time", "chargeback", "not notified", "don't recall",
]
PLAN_MENTION = ["free plan", "pro plan", "enterprise plan"]


def approve_features(state):
    tier = (state.get("customer_tier") or "free").lower()
    amt = float(state.get("amount") or 0.0)
    body = ((state.get("subject") or "") + " " + (state.get("body") or "")).lower()
    f = [1.0 if tier == t else 0.0 for t in TIERS]
    f += [1.0 if amt == 0 else 0.0, math.log1p(amt) / 7.0]
    for t in TIERS:
        on = 1.0 if tier == t else 0.0
        f += [on * (1.0 if amt > c else 0.0) for c in CUTS]
    f += [1.0 if k in body else 0.0 for k in RED_FLAGS]
    f += [1.0 if k in body else 0.0 for k in PLAN_MENTION]
    return np.asarray(f, dtype=np.float32)


N_APPROVE_FEATS = len(approve_features({}))


def grounded_features(state):
    d = quantities(state.get("draft_reply") or "")
    k = quantities(state.get("kb_article") or "")
    return np.asarray([float(len(d - k)), float(len(d & k)), float(d <= k)], dtype=np.float32)
