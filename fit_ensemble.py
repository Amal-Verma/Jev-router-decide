"""Calibrate the merged ensemble graph and write its policy (holdout only).

Uses the merged int8 ONNX graph's own outputs on the 1,000-row training holdout
both members share (cal_ids.json), never the dev split:
  1. one temperature per member and decision,
  2. average of the members' scaled logits, then a final temperature,
  3. abstention thresholds at ~85% coverage; the approve band keeps the served
     lower edge (0.15) and moves the upper edge down only as far as needed for
     holdout approve coverage >= 0.84.

    python fit_ensemble.py kaggle/art_ensemble data/train.jsonl
"""
import json
import os
import sys

import numpy as np

from decide.runtime import EnsembleBackend
from train_encoder import QUEUES, fit_temperature

TASKS = ["route", "approve", "grounded"]


def sig(z):
    return 1 / (1 + np.exp(-z))


def main(art, train_path):
    with open(os.path.join(art, "ensemble.json"), "w") as fh:     # members first, temps filled below
        json.dump({"members": [{"prefix": "m_", "tok": "tok_m", "name": "MiniLM-L6-H384 + MoE"},
                               {"prefix": "e_", "tok": "tok_e", "name": "Ettin-17m + MoE"}],
                   "member_temp": [{t: 1.0 for t in TASKS}, {t: 1.0 for t in TASKS}]}, fh)
    with open(os.path.join(art, "calibration.json"), "w") as fh:
        json.dump({t: 1.0 for t in TASKS}, fh)
    rows = {r["request_id"]: r for r in (json.loads(l) for l in open(train_path) if l.strip())}
    cal = [rows[i] for i in json.load(open(os.path.join(art, "cal_ids.json")))]
    y = {"route": np.array([QUEUES.index(r["labels"]["route"]) for r in cal]),
         "approve": np.array([float(r["labels"]["approve"]) for r in cal]),
         "grounded": np.array([float(r["labels"]["grounded"]) for r in cal])}
    b = EnsembleBackend(art)
    raw = {t: [np.array(z) for z in zip(*(b.member_logits(t, r["state"]) for r in cal))] for t in TASKS}

    member_temp = [{}, {}]
    combined = {}
    for t in TASKS:
        scaled = []
        for i in range(2):
            member_temp[i][t] = fit_temperature(raw[t][i], y[t], t == "route")
            scaled.append(raw[t][i] / member_temp[i][t])
        combined[t] = np.mean(scaled, 0)
    final = {t: fit_temperature(combined[t], y[t], t == "route") for t in TASKS}

    z = combined["route"] / final["route"]
    pr = np.exp(z - z.max(1, keepdims=True)); pr /= pr.sum(1, keepdims=True)
    pa, pg = sig(combined["approve"] / final["approve"]), sig(combined["grounded"] / final["grounded"])
    hi = 0.79
    while hi > 0.6 and np.mean(~((pa > 0.15) & (pa < hi))) < 0.84:
        hi -= 0.01
    policy = {"route_min_conf": float(np.quantile(pr.max(1), 0.15)),
              "approve_min_conf": float(np.quantile(np.maximum(pa, 1 - pa), 0.15)),
              "grounded_min_conf": float(np.quantile(np.maximum(pg, 1 - pg), 0.15)),
              "approve_threshold": 0.75, "approve_abstain_band": [0.15, round(hi, 2)],
              "coverage_target": 0.85}
    cov = {"route": float(np.mean(pr.max(1) >= policy["route_min_conf"])),
           "approve": float(np.mean(~((pa > 0.15) & (pa < hi)))),
           "grounded": float(np.mean(np.maximum(pg, 1 - pg) >= policy["grounded_min_conf"]))}

    cfg = json.load(open(os.path.join(art, "ensemble.json")))
    cfg["member_temp"] = member_temp
    json.dump(cfg, open(os.path.join(art, "ensemble.json"), "w"), indent=2)
    json.dump(final, open(os.path.join(art, "calibration.json"), "w"), indent=2)
    json.dump(policy, open(os.path.join(art, "policy.json"), "w"), indent=2)
    print("member temperatures", member_temp)
    print("final temperatures", final)
    print("policy", policy)
    print("holdout coverage", {k: round(v, 3) for k, v in cov.items()})


if __name__ == "__main__":
    main(*sys.argv[1:])
