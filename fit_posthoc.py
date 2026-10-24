"""Fit post-hoc calibration maps for the served model (writes posthoc.json).

Uses the *served int8 model's own* predictions on its 1,000-row training
holdout (cal_ids.json), so the maps match exactly what serves. The maps are
the CV winner in docs/EXPERIMENTS.md: isotonic curves for approve and grounded,
and an isotonic map from the route top-2 logit margin to the top probability.

    python fit_posthoc.py <artifact dir with decide.onnx, cal_ids.json> data/train.jsonl
"""
import json
import os
import sys

import numpy as np
from sklearn.isotonic import IsotonicRegression

from decide.runtime import EncoderBackend

QUEUES = ["billing", "technical", "account", "shipping", "general"]


def curve(x, y, lo, hi):
    m = IsotonicRegression(out_of_bounds="clip", y_min=lo, y_max=hi).fit(x, y)
    return {"x": [float(v) for v in m.X_thresholds_], "y": [float(v) for v in m.y_thresholds_]}


def main(art, train_path):
    rows = {r["request_id"]: r for r in (json.loads(l) for l in open(train_path) if l.strip())}
    cal = [rows[i] for i in json.load(open(os.path.join(art, "cal_ids.json")))]
    b = EncoderBackend(art)
    b.maps = {}                                      # fit on raw temperature outputs
    pa = np.array([b.approve_prob(r["state"]) for r in cal])
    pg = np.array([b.grounded_prob(r["state"]) for r in cal])
    raw = np.array([b._logits("route", r["state"]) for r in cal])
    s = np.sort(raw, 1)
    margin = s[:, -1] - s[:, -2]
    correct = (raw.argmax(1) == [QUEUES.index(r["labels"]["route"]) for r in cal]).astype(float)
    maps = {"approve": curve(pa, [float(r["labels"]["approve"]) for r in cal], 1e-3, 1 - 1e-3),
            "grounded": curve(pg, [float(r["labels"]["grounded"]) for r in cal], 1e-3, 1 - 1e-3),
            "route_margin": curve(margin, correct, 0.2, 0.999)}
    with open(os.path.join(art, "posthoc.json"), "w") as fh:
        json.dump(maps, fh)
    print({k: len(v["x"]) for k, v in maps.items()}, "points; wrote", os.path.join(art, "posthoc.json"))


if __name__ == "__main__":
    main(*sys.argv[1:])
