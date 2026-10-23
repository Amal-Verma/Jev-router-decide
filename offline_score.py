"""Score saved logits with the self-check's own metric code, without a server.

Rebuilds the response dicts the service would return (same calibration, same
policy as model.py), then reuses selfcheck.py's metric functions and its
aggregation rules. Latency is not measured here; it is reported as 1.0 (the
served model already gets full latency marks) so totals stay comparable.

    python offline_score.py <artifact dir> <eval.jsonl> [<logits.npz>]
"""
import json
import os
import sys
from collections import Counter

import numpy as np

import selfcheck as sc

QUEUES = sc.QUEUES


def responses(logits, temps, policy):
    z = logits["route"] / temps["route"]
    p = np.exp(z - z.max(1, keepdims=True))
    p /= p.sum(1, keepdims=True)
    pa = 1 / (1 + np.exp(-logits["approve"] / temps["approve"]))
    pg = 1 / (1 + np.exp(-logits["grounded"] / temps["grounded"]))
    out = {"route": [], "approve": [], "grounded": []}
    for i in range(len(pa)):
        probs = {q: float(p[i, j]) for j, q in enumerate(QUEUES)}
        v = max(probs, key=probs.get)
        out["route"].append({"value": v, "probs": probs, "confidence": probs[v],
                             "abstain": probs[v] < policy["route_min_conf"]})
        out["approve"].append({"value": bool(pa[i] > policy["approve_threshold"]), "p_true": float(pa[i]),
                               "abstain": max(pa[i], 1 - pa[i]) < policy["approve_min_conf"]})
        out["grounded"].append({"value": bool(pg[i] > 0.5), "p_true": float(pg[i]),
                                "abstain": max(pg[i], 1 - pg[i]) < policy["grounded_min_conf"]})
    return out


def score(rows, resp):
    """Mirror of selfcheck.main()'s quality / calibration / abstention block."""
    res = {d: [(r["labels"][d], resp[d][i]) for i, r in enumerate(rows)] for d in sc.DECISIONS}
    ans = lambda r: not r["abstain"]

    pairs, confs, corr, a_r, prob_rows, truths = [], [], [], [], [], []
    for t, r in res["route"]:
        top = max(r["probs"], key=r["probs"].get)
        confs.append(r["probs"][top]); corr.append(1.0 if top == t else 0.0)
        prob_rows.append(r["probs"]); truths.append(t); a_r.append(ans(r))
        if ans(r):
            pairs.append((t, r["value"]))
    route_cov = sum(a_r) / len(a_r)
    route_f1, route_acc = sc.macro_f1(pairs, QUEUES), sum(corr) / len(corr)
    route_ece, route_risk = sc.ece(confs, corr), sc.selective_risk(confs, corr)

    def binary(dec):
        ys = [1.0 if t else 0.0 for t, _ in res[dec]]
        probs = [r["p_true"] for _, r in res[dec]]
        preds = [r["value"] for _, r in res[dec]]
        aa = [ans(r) for _, r in res[dec]]
        bconf = [max(p, 1 - p) for p in probs]
        bcorr = [1.0 if (p > 0.5) == (y > 0.5) else 0.0 for p, y in zip(probs, ys)]
        return ys, probs, preds, aa, sc.ece(probs, ys), sc.brier(probs, ys), sc.selective_risk(bconf, bcorr), bcorr

    ys, ap, apred, aa, a_ece, a_brier, a_risk, a_corr = binary("approve")
    gys, gp, gpred, ga, g_ece, g_brier, g_risk, g_corr = binary("grounded")
    appr_cov, gr_cov = sum(aa) / len(aa), sum(ga) / len(ga)
    appr_cost = sc.cost_error([bool(y) for y, a in zip(ys, aa) if a], [p for p, a in zip(apred, aa) if a])
    gr_f1 = sc.macro_f1([(bool(y), p) for y, p, a in zip(gys, gpred, ga) if a], [True, False])

    ans_route = [t for t, a in zip(truths, a_r) if a]
    maj = Counter(ans_route).most_common(1)[0][0]
    triv_route = sc.macro_f1([(t, maj) for t in ans_route], QUEUES)
    ans_ys = [y for y, a in zip(ys, aa) if a]
    hold_cost = sum(ans_ys) / len(ans_ys)
    ans_g = [y for y, a in zip(gys, ga) if a]
    triv_g = max(sc.macro_f1([(bool(y), True) for y in ans_g], [True, False]),
                 sc.macro_f1([(bool(y), False) for y in ans_g], [True, False]))

    cov_scale = lambda c: min(1.0, c / sc.TARGET_COVERAGE)
    quality = (sc.skill(route_f1, triv_route) * cov_scale(route_cov)
               + sc.cost_score(appr_cost, hold_cost) * cov_scale(appr_cov)
               + sc.skill(gr_f1, triv_g) * cov_scale(gr_cov)) / 3

    prior = {c: truths.count(c) / len(truths) for c in QUEUES}
    rb = sc.multiclass_brier(prob_rows, truths, QUEUES)
    rref = sc.multiclass_brier([prior] * len(truths), truths, QUEUES)
    ar, gr = sum(ys) / len(ys), sum(gys) / len(gys)
    skill = (sc.brier_skill(rb, rref) + sc.brier_skill(a_brier, ar * (1 - ar)) + sc.brier_skill(g_brier, gr * (1 - gr))) / 3
    mean_ece = (route_ece + a_ece + g_ece) / 3
    calibration = 0.5 * max(0.0, 1 - mean_ece / 0.20) + 0.5 * skill

    full_err = [1 - route_acc, 1 - sum(a_corr) / len(a_corr), 1 - sum(g_corr) / len(g_corr)]
    abst = sum(max(0.0, 1 - r / e) if e > 1e-9 else 1.0
               for r, e in zip([route_risk, a_risk, g_risk], full_err)) / 3
    parts = {"quality": quality, "calibration": calibration, "latency": 1.0, "abstention": abst}
    total = sum(sc.WEIGHTS[k] * parts[k] for k in sc.WEIGHTS) * 100
    return {"total": total, **parts, "route_f1": route_f1, "approve_cost": appr_cost, "grounded_f1": gr_f1,
            "mean_ece": mean_ece, "coverage": (route_cov, appr_cov, gr_cov)}


def main(art, eval_path, logits_path=None):
    rows = [json.loads(l) for l in open(eval_path) if l.strip()]
    temps = json.load(open(os.path.join(art, "calibration.json")))
    policy = json.load(open(os.path.join(art, "policy.json")))
    lg = np.load(logits_path or os.path.join(art, "dev_logits.npz"))
    s = score(rows, responses(lg, temps, policy))
    print(json.dumps({k: (round(v, 4) if isinstance(v, float) else v) for k, v in s.items()}))


if __name__ == "__main__":
    main(*sys.argv[1:])
