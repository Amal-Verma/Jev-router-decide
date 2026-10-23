"""Pick abstention thresholds from the calibration holdout (never from dev).

Quality is scaled down below 80% coverage and abstaining above it is free, so
each decision abstains on its least-confident ~15% (coverage target 0.85, a
margin over 0.80: on dev, 0.82 landed at 0.799 for route).

    python tune_policy.py artifacts
"""
import json
import os
import sys

import numpy as np

COVERAGE = 0.85


def sigmoid(z):
    return 1.0 / (1.0 + np.exp(-z))


def main(art="artifacts"):
    temps = json.load(open(os.path.join(art, "calibration.json")))
    lc = np.load(os.path.join(art, "cal_logits.npz"))
    z = lc["route"] / temps["route"]
    p = np.exp(z - z.max(1, keepdims=True))
    p /= p.sum(1, keepdims=True)
    route_conf = p.max(1)
    pa = sigmoid(lc["approve"] / temps["approve"])
    pg = sigmoid(lc["grounded"] / temps["grounded"])
    q = 1.0 - COVERAGE
    policy = {
        "route_min_conf": float(np.quantile(route_conf, q)),
        "approve_min_conf": float(np.quantile(np.maximum(pa, 1 - pa), q)),
        "grounded_min_conf": float(np.quantile(np.maximum(pg, 1 - pg), q)),
        "approve_threshold": 0.75,
        # Asymmetric approve band, chosen on 8,000 out-of-fold predictions
        # (cv_analyze / docs/report): cuts expected 3:1 cost on answered cases.
        "approve_abstain_band": [0.15, 0.79],
        "coverage_target": COVERAGE,
    }
    with open(os.path.join(art, "policy.json"), "w") as fh:
        json.dump(policy, fh, indent=2)
    print(json.dumps(policy, indent=2))


if __name__ == "__main__":
    main(*sys.argv[1:])
