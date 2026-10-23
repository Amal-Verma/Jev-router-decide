"""Cross-validation report from the Kaggle CV run (kaggle/decide_cv.ipynb).

For every fold and epoch: fit temperatures and the abstention policy on that
fold's own 1,000-row holdout, then score the fold's 2,000 unseen rows with the
self-check metric code (offline_score.py). Variants share the same logits, so
differences come from the choice being tested, not from retraining noise.

    python cv_analyze.py kaggle/cv_logits kaggle/cv_bundle
"""
import json
import os
import sys

import numpy as np

from offline_score import responses, score
from train_encoder import QUEUES, fit_temperature

FOLDS = 4


def load(p):
    return [json.loads(l) for l in open(p) if l.strip()]


def sig(z):
    return 1 / (1 + np.exp(-z))


def policy_from(cal, temps, coverage, approve_threshold=0.75):
    z = cal["route"] / temps["route"]
    p = np.exp(z - z.max(1, keepdims=True))
    p /= p.sum(1, keepdims=True)
    pa, pg = sig(cal["approve"] / temps["approve"]), sig(cal["grounded"] / temps["grounded"])
    q = 1 - coverage
    return {"route_min_conf": float(np.quantile(p.max(1), q)),
            "approve_min_conf": float(np.quantile(np.maximum(pa, 1 - pa), q)),
            "grounded_min_conf": float(np.quantile(np.maximum(pg, 1 - pg), q)),
            "approve_threshold": approve_threshold}


def best_approve_threshold(cal, temps, ys):
    """Threshold minimising 3:1 cost on the holdout."""
    pa = sig(cal["approve"] / temps["approve"])
    ys = np.asarray(ys, dtype=bool)
    best = min(np.arange(0.40, 0.95, 0.01),
               key=lambda t: (3 * np.sum((pa > t) & ~ys) + np.sum((pa <= t) & ys)))
    return float(best)


def main(cv_dir="kaggle/cv_logits", bundle="kaggle/cv_bundle"):
    rows_all = {}
    results = {}
    for k in range(FOLDS):
        fold = os.path.join(cv_dir, f"fold{k}")
        train = {r["request_id"]: r for r in load(os.path.join(bundle, f"fold{k}_train.jsonl"))}
        ev = load(os.path.join(bundle, f"fold{k}_eval.jsonl"))
        cal_rows = [train[i] for i in json.load(open(os.path.join(fold, "cal_ids.json")))]
        y_route = [QUEUES.index(r["labels"]["route"]) for r in cal_rows]
        y_appr = [float(r["labels"]["approve"]) for r in cal_rows]
        y_grnd = [float(r["labels"]["grounded"]) for r in cal_rows]
        for e in range(1, 6):
            cal = dict(np.load(os.path.join(fold, f"e{e}_cal.npz")))
            dev = dict(np.load(os.path.join(fold, f"e{e}_dev.npz")))
            temps = {"route": fit_temperature(cal["route"], y_route, True),
                     "approve": fit_temperature(cal["approve"], y_appr, False),
                     "grounded": fit_temperature(cal["grounded"], y_grnd, False)}
            variants = {f"cov{c}": policy_from(cal, temps, c) for c in (0.82, 0.85, 0.88)}
            variants["cov0.85+tuned_thr"] = policy_from(
                cal, temps, 0.85, best_approve_threshold(cal, temps, y_appr))
            for name, pol in variants.items():
                s = score(ev, responses(dev, temps, pol))
                results.setdefault((e, name), []).append(s)

    keys = ["total", "quality", "calibration", "abstention", "route_f1", "approve_cost", "grounded_f1", "mean_ece"]
    print(f"{'epoch':>5} {'variant':<18}" + "".join(f"{k:>13}" for k in keys) + "   min cov")
    for (e, name), ss in sorted(results.items()):
        line = f"{e:>5} {name:<18}"
        for k in keys:
            v = np.array([s[k] for s in ss])
            line += f"  {v.mean():6.3f}±{v.std():.3f}" if k != "total" else f"  {v.mean():6.2f}±{v.std():.2f}"
        line += f"   {min(min(s['coverage']) for s in ss):.3f}"
        print(line)
    folds_total = {k: round(results[(5, 'cov0.85')][k]['total'], 2) for k in range(FOLDS)}
    print("\nper-fold total, served config (5 epochs, cov0.85):", folds_total)


if __name__ == "__main__":
    main(*sys.argv[1:])
