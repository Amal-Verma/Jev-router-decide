"""Classical baseline (gate 1): TF-IDF + logistic regression / LightGBM.

Not the submission model; kept as the reference the transformer must beat.
    python train_baseline.py   -> baseline/baseline.pkl
"""
import json
import pickle
import sys

import lightgbm as lgb
import numpy as np
from scipy.sparse import csr_matrix, hstack
from sklearn.calibration import CalibratedClassifierCV
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression

from decide.features import tokens
from decide.tabular import approve_features, grounded_features

QUEUES = ["billing", "technical", "account", "shipping", "general"]


def load(p):
    return [json.loads(l) for l in open(p, encoding="utf-8") if l.strip()]


def ticket(s):
    return (s.get("subject") or "") + " . " + (s.get("body") or "")


def diff_text(s):
    """Draft tokens absent from the article, plus article tokens absent from the draft."""
    d, k = tokens(s.get("draft_reply") or ""), tokens(s.get("kb_article") or "")
    ks, ds = set(k), set(d)
    return " ".join(["D_" + t for t in d if t not in ks] + ["K_" + t for t in k if t not in ds])


def main(out="baseline/baseline.pkl"):
    tr = load("data/train.jsonl")
    S = [r["state"] for r in tr]

    route_vec = TfidfVectorizer(ngram_range=(1, 2), min_df=2, sublinear_tf=True)
    Xr = route_vec.fit_transform([ticket(s) for s in S])
    yr = [QUEUES.index(r["labels"]["route"]) for r in tr]
    route = CalibratedClassifierCV(LogisticRegression(C=4, max_iter=2000), cv=5, method="sigmoid").fit(Xr, yr)

    Xa = hstack([Xr, csr_matrix(np.stack([approve_features(s) for s in S]))]).tocsr()
    ya = [int(r["labels"]["approve"]) for r in tr]
    approve = CalibratedClassifierCV(
        lgb.LGBMClassifier(n_estimators=400, learning_rate=0.05, num_leaves=31, verbose=-1),
        cv=5, method="isotonic").fit(Xa, ya)

    g_vec = TfidfVectorizer(ngram_range=(1, 2), min_df=2, token_pattern=r"\S+")
    Xg = hstack([g_vec.fit_transform([diff_text(s) for s in S]),
                 csr_matrix(np.stack([grounded_features(s) for s in S]))]).tocsr()
    yg = [int(r["labels"]["grounded"]) for r in tr]
    grounded = CalibratedClassifierCV(LogisticRegression(C=4, max_iter=2000), cv=5, method="sigmoid").fit(Xg, yg)

    with open(out, "wb") as fh:
        pickle.dump({"route_vec": route_vec, "route": route, "approve": approve,
                     "g_vec": g_vec, "grounded": grounded}, fh)
    print("wrote", out)


if __name__ == "__main__":
    main(*sys.argv[1:])
