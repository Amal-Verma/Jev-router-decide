"""Bayesian variants (MC dropout, last-layer Laplace) against the plain model.

Paired: every variant is computed from the same trained fold model, so the
comparison is not swamped by seed noise. Calibration and the abstention policy
are fitted on each fold's own 1,000-row holdout; scoring uses the self-check
metric code on the fold's 2,000 unseen rows.

    python bayes_analyze.py kaggle/bayes_logits kaggle/bayes_bundle
"""
import json
import sys

import numpy as np
from sklearn.isotonic import IsotonicRegression

from cv_analyze import policy_from
from offline_score import responses, score
from posthoc_cal import as_logits, sig, softmax
from train_encoder import QUEUES, fit_temperature


def aug(z):
    return np.hstack([z, np.ones((len(z), 1))])


def laplace_var(z_train, p_train, z_eval, lam):
    """Predictive logit variance under a Laplace posterior over one head's
    weights (GGN Hessian, isotropic prior with precision lam)."""
    A = aug(z_train)
    w = p_train * (1 - p_train)
    H = (A * w[:, None]).T @ A + lam * np.eye(A.shape[1])
    S = np.linalg.inv(H)
    E = aug(z_eval)
    return np.einsum("ij,jk,ik->i", E, S, E)


def probit_binary(f, var):
    return sig(f / np.sqrt(1 + np.pi * var / 8))


def laplace_head(tr, cal, dev, heads, task, y_cal, lams=np.logspace(-2, 3, 11)):
    """Returns (cal_probs, dev_probs) with lam chosen by holdout NLL."""
    W = heads[f"{task}_head_weight"]; b = heads[f"{task}_head_bias"]
    f = lambda z: z @ W.T + b
    ftr, fc, fd = f(tr), f(cal), f(dev)
    best = None
    for lam in lams:
        if task == "route":
            ptr = softmax(ftr)
            vc = np.stack([laplace_var(tr, ptr[:, k], cal, lam) for k in range(len(QUEUES))], 1)
            pc = softmax(fc / np.sqrt(1 + np.pi * vc / 8))
            nll = -np.mean(np.log(pc[np.arange(len(y_cal)), y_cal] + 1e-12))
        else:
            ptr = sig(ftr[:, 0])
            vc = laplace_var(tr, ptr, cal, lam)
            pc = probit_binary(fc[:, 0], vc)
            nll = -np.mean(y_cal * np.log(pc + 1e-12) + (1 - y_cal) * np.log(1 - pc + 1e-12))
        if best is None or nll < best[0]:
            best = (nll, lam)
    lam = best[1]
    if task == "route":
        ptr = softmax(ftr)
        vc = np.stack([laplace_var(tr, ptr[:, k], cal, lam) for k in range(len(QUEUES))], 1)
        vd = np.stack([laplace_var(tr, ptr[:, k], dev, lam) for k in range(len(QUEUES))], 1)
        return softmax(fc / np.sqrt(1 + np.pi * vc / 8)), softmax(fd / np.sqrt(1 + np.pi * vd / 8)), lam
    ptr = sig(ftr[:, 0])
    return (probit_binary(fc[:, 0], laplace_var(tr, ptr, cal, lam)),
            probit_binary(fd[:, 0], laplace_var(tr, ptr, dev, lam)), lam)


def temp_probs(p_cal, p_dev, y, multiclass):
    """Temperature-recalibrate given probabilities (treated as logits)."""
    if multiclass:
        lc, ld = np.log(p_cal + 1e-12), np.log(p_dev + 1e-12)
        t = fit_temperature(lc, y, True)
        return softmax(lc / t), softmax(ld / t)
    lc = np.log(p_cal + 1e-12) - np.log(1 - p_cal + 1e-12)
    ld = np.log(p_dev + 1e-12) - np.log(1 - p_dev + 1e-12)
    t = fit_temperature(lc, y, False)
    return sig(lc / t), sig(ld / t)


def iso(p_cal, p_dev, y):
    m = IsotonicRegression(out_of_bounds="clip", y_min=1e-3, y_max=1 - 1e-3).fit(p_cal, y)
    return m.predict(p_cal), m.predict(p_dev)


def main(logdir="kaggle/bayes_logits", bundle="kaggle/bayes_bundle"):
    out = {}
    for k in range(4):
        d = f"{logdir}/fold{k}"
        ev = [json.loads(l) for l in open(f"{bundle}/fold{k}_eval.jsonl")]
        train = {r["request_id"]: r for r in (json.loads(l) for l in open(f"{bundle}/fold{k}_train.jsonl"))}
        cal = [train[i] for i in json.load(open(f"{d}/cal_ids.json"))]
        yr = np.array([QUEUES.index(r["labels"]["route"]) for r in cal])
        ya = np.array([r["labels"]["approve"] for r in cal], float)
        yg = np.array([r["labels"]["grounded"] for r in cal], float)
        lc, ld = dict(np.load(f"{d}/cal_logits.npz")), dict(np.load(f"{d}/dev_logits.npz"))
        bt, bc, bd = (dict(np.load(f"{d}/bayes_{s}.npz")) for s in ("train", "cal", "dev"))
        heads = dict(np.load(f"{d}/heads.npz"))

        T = {"route": fit_temperature(lc["route"], yr, True), "approve": fit_temperature(lc["approve"], ya, False),
             "grounded": fit_temperature(lc["grounded"], yg, False)}
        plain_c = (softmax(lc["route"] / T["route"]), sig(lc["approve"] / T["approve"]), sig(lc["grounded"] / T["grounded"]))
        plain_d = (softmax(ld["route"] / T["route"]), sig(ld["approve"] / T["approve"]), sig(ld["grounded"] / T["grounded"]))

        mc = [temp_probs(bc["route_mc"], bd["route_mc"], yr, True),
              temp_probs(bc["approve_mc"], bd["approve_mc"], ya, False),
              temp_probs(bc["grounded_mc"], bd["grounded_mc"], yg, False)]
        lap = []
        for task, y in (("route", yr), ("approve", ya), ("grounded", yg)):
            pc, pd, _ = laplace_head(bt[f"{task}_z"], bc[f"{task}_z"], bd[f"{task}_z"], heads, task,
                                     y.astype(int) if task == "route" else y)
            lap.append(temp_probs(pc, pd, y, task == "route"))

        variants = {
            "plain (temperature)": (plain_c, plain_d),
            "MC dropout": (tuple(x[0] for x in mc), tuple(x[1] for x in mc)),
            "Laplace last layer": (tuple(x[0] for x in lap), tuple(x[1] for x in lap)),
        }
        for name in list(variants):
            c, dd = variants[name]
            ia, ig = iso(c[1], dd[1], ya), iso(c[2], dd[2], yg)
            variants[name + " + isotonic"] = ((c[0], ia[0], ig[0]), (dd[0], ia[1], ig[1]))
        for name, (c, dd) in variants.items():
            lgc, tt = as_logits(*c); lgd, _ = as_logits(*dd)
            pol = policy_from(lgc, tt, 0.85, band=[0.15, 0.79])
            out.setdefault(name, []).append(score(ev, responses(lgd, tt, pol)))

    base = np.array([s["total"] for s in out["plain (temperature)"]])
    for name, ss in out.items():
        t = np.array([s["total"] for s in ss])
        print(f"{name:32s} total {t.mean():.2f} ± {t.std():.2f}  vs plain {t.mean() - base.mean():+.2f} "
              f"(better on {int((t > base + 1e-9).sum())}/4)  C {np.mean([s['calibration'] for s in ss]):.3f}  "
              f"A {np.mean([s['abstention'] for s in ss]):.3f}  ECE {np.mean([s['mean_ece'] for s in ss]):.4f}")


if __name__ == "__main__":
    main(*sys.argv[1:])
