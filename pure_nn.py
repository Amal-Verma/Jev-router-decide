"""A pure neural network, trained from scratch (no pretrained transformer).

Preprocessing and transformations turn every request into one feature vector:
  * text normalisation: lower-case, durations canonicalised ("two weeks" -> "14 days"),
    money and ids masked, ticket numbers removed
  * word 1-2 gram and character 3-5 gram TF-IDF of the ticket
  * draft-vs-article difference tokens (TF-IDF) and duration-match features
  * approve tabular features (tier x amount cuts, red flags), log-amount
Then ONE network: shared input projection -> shared mixture of experts (gate sees
the task embedding) -> one small head per decision. Calibrated by temperature on
the 1,000-row holdout, scored with the self-check metric code.

    python pure_nn.py            # trains on train.jsonl only, scores dev
"""
import json
import random
import re
import sys

import numpy as np
import scipy.sparse as sp
import torch
import torch.nn as nn
import torch.nn.functional as F
from sklearn.feature_extraction.text import TfidfVectorizer

from decide.features import QTY_RE, NUM_WORDS, UNIT_DAYS, quantities, tokens
from decide.tabular import approve_features
from offline_score import responses, score
from cv_analyze import policy_from
from train_encoder import fit_temperature

QUEUES = ["billing", "technical", "account", "shipping", "general"]
TASKS = ["route", "approve", "grounded"]


def _code(v):
    """Digit-free token for a duration in days, so number masking leaves it intact."""
    return "dur" + "".join("abcdefghij"[int(c)] if c.isdigit() else "p" for c in f"{v:g}")


def normalise(text):
    t = text.lower()
    durs = []

    def dur(m):
        n, u = m.group(1), m.group(2)
        n = NUM_WORDS.get(n) if not n[0].isdigit() else float(n)
        if n is None:
            return m.group(0)
        u = u.rstrip("s") if u != "business days" else "business day"
        durs.append(_code(round(float(n) * UNIT_DAYS.get(u, 1), 3)))
        return f" qqdur{len(durs) - 1}qq "
    t = QTY_RE.sub(dur, t)
    t = re.sub(r"\$\s?[\d,]+(\.\d+)?", " moneyamt ", t)
    t = re.sub(r"#\d+|tkt-\d+|inv-\d+", " refid ", t)
    t = re.sub(r"(?<!qqdur)\d+", " num ", t)
    t = re.sub(r"qqdur(\d+)qq", lambda m: durs[int(m.group(1))], t)
    return re.sub(r"\s+", " ", t).strip()


def ticket(s):
    return normalise((s.get("subject") or "") + " . " + (s.get("body") or ""))


def diff_text(s):
    d, k = tokens(normalise(s.get("draft_reply") or "")), tokens(normalise(s.get("kb_article") or ""))
    ks, ds = set(k), set(d)
    return " ".join(["D_" + t for t in d if t not in ks] + ["K_" + t for t in k if t not in ds])


def ground_feats(s):
    d, k = quantities(s.get("draft_reply") or ""), quantities(s.get("kb_article") or "")
    return [len(d - k), len(k - d), len(d & k), float(d <= k), float(bool(d))]


class Featuriser:
    def fit(self, states):
        self.w = TfidfVectorizer(ngram_range=(1, 2), min_df=2, max_features=20000, sublinear_tf=True)
        self.c = TfidfVectorizer(analyzer="char_wb", ngram_range=(3, 5), min_df=3, max_features=20000, sublinear_tf=True)
        self.g = TfidfVectorizer(token_pattern=r"\S+", ngram_range=(1, 2), min_df=2, max_features=10000, sublinear_tf=True)
        t = [ticket(s) for s in states]
        self.w.fit(t); self.c.fit(t); self.g.fit([diff_text(s) for s in states])
        return self

    def transform(self, states):
        t = [ticket(s) for s in states]
        tab = np.stack([approve_features(s) for s in states])
        gf = np.array([ground_feats(s) for s in states], dtype=np.float32)
        return sp.hstack([self.w.transform(t), self.c.transform(t), self.g.transform([diff_text(s) for s in states]),
                          sp.csr_matrix(tab), sp.csr_matrix(gf)]).tocsr().astype(np.float32)


class PureMoE(nn.Module):
    def __init__(self, d_in, d=384, n_experts=4, d_expert=256):
        super().__init__()
        self.proj = nn.Sequential(nn.Dropout(0.2), nn.Linear(d_in, d), nn.GELU(), nn.Dropout(0.2))
        self.task_emb = nn.Embedding(3, 32)
        self.experts = nn.ModuleList([nn.Sequential(nn.Linear(d + 32, d_expert), nn.GELU(), nn.Dropout(0.1))
                                      for _ in range(n_experts)])
        self.gate = nn.Linear(d + 32, n_experts)
        self.heads = nn.ModuleList([nn.Linear(d_expert, 5), nn.Linear(d_expert, 1), nn.Linear(d_expert, 1)])

    def forward(self, x, task):
        h = self.proj(x)
        te = self.task_emb(task)
        hx = torch.cat([h, te], -1)
        g = torch.softmax(self.gate(hx), -1)
        z = (g.unsqueeze(-1) * torch.stack([e(hx) for e in self.experts], 1)).sum(1)
        return [hd(z) for hd in self.heads]


def labels(rows, task):
    if task == "route":
        return np.array([QUEUES.index(r["labels"]["route"]) for r in rows])
    return np.array([float(r["labels"][task]) for r in rows])


@torch.no_grad()
def predict(model, X):
    model.eval()
    out = {}
    for ti, t in enumerate(TASKS):
        zs = []
        for i in range(0, X.shape[0], 512):
            xb = torch.tensor(X[i:i + 512].toarray())
            z = model(xb, torch.full((xb.shape[0],), ti, dtype=torch.long))[ti]
            zs.append(z if t == "route" else z.squeeze(-1))
        out[t] = torch.cat(zs).numpy()
    return out


def main(seed=13, epochs=12):
    seed = int(seed)
    random.seed(seed); np.random.seed(seed); torch.manual_seed(seed)
    rows = [json.loads(l) for l in open("data/train.jsonl") if l.strip()]
    random.shuffle(rows)
    cal, train = rows[:1000], rows[1000:]
    dev = [json.loads(l) for l in open("data/dev.jsonl") if l.strip()]
    fz = Featuriser().fit([r["state"] for r in train])
    Xtr, Xc, Xd = (fz.transform([r["state"] for r in rs]) for rs in (train, cal, dev))
    print("features", Xtr.shape[1], flush=True)
    model = PureMoE(Xtr.shape[1])
    opt = torch.optim.AdamW(model.parameters(), lr=1e-3, weight_decay=1e-4)
    Y = {t: labels(train, t) for t in TASKS}
    n = Xtr.shape[0]; bs = 64
    best, best_state = 1e9, None
    for ep in range(epochs):
        model.train()
        order = [(ti, idx) for ti in range(3) for idx in np.array_split(np.random.permutation(n), n // bs)]
        random.shuffle(order)
        for ti, idx in order:
            xb = torch.tensor(Xtr[idx].toarray())
            z = model(xb, torch.full((len(idx),), ti, dtype=torch.long))[ti]
            y = torch.tensor(Y[TASKS[ti]][idx])
            loss = F.cross_entropy(z, y.long()) if ti == 0 else F.binary_cross_entropy_with_logits(z.squeeze(-1), y.float())
            opt.zero_grad(); loss.backward(); opt.step()
        lc = predict(model, Xc)
        nll = (F.cross_entropy(torch.tensor(lc["route"]), torch.tensor(labels(cal, "route"))).item()
               + sum(F.binary_cross_entropy_with_logits(torch.tensor(lc[t]), torch.tensor(labels(cal, t)).float()).item()
                     for t in ("approve", "grounded")))
        print(f"epoch {ep + 1}: holdout NLL {nll:.4f}", flush=True)
        if nll < best:
            best, best_state = nll, {k: v.clone() for k, v in model.state_dict().items()}
    model.load_state_dict(best_state)
    lc, ld = predict(model, Xc), predict(model, Xd)
    T = {"route": fit_temperature(lc["route"], labels(cal, "route"), True),
         "approve": fit_temperature(lc["approve"], labels(cal, "approve"), False),
         "grounded": fit_temperature(lc["grounded"], labels(cal, "grounded"), False)}
    s = score(dev, responses(ld, T, policy_from(lc, T, 0.85)))
    print(json.dumps({k: (round(v, 4) if isinstance(v, float) else v) for k, v in s.items()}))
    return model, fz


if __name__ == "__main__":
    main(*sys.argv[1:])
