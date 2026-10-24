"""Cross-validated test of the MiniLM + Ettin-17m ensemble (4 folds, 8,000 rows).

Both members were trained on the same fold files with the same seeds, so they
share each fold's calibration holdout. Each member is temperature-calibrated on
that holdout, the calibrated logits are averaged, a final temperature and the
abstention policy are fitted on the holdout, and the fold's 2,000 unseen rows
are scored with the self-check metric code.

    python ens_cv.py
"""
import json

import numpy as np

from cv_analyze import policy_from
from offline_score import responses, score
from train_encoder import QUEUES, fit_temperature


def member(lc, ld, ys):
    out_c, out_d = {}, {}
    for t, y in ys.items():
        T = fit_temperature(lc[t], y, t == "route")
        out_c[t], out_d[t] = lc[t] / T, ld[t] / T
    return out_c, out_d


def run(lcs, lds, ys, ev, cov, band):
    lc = {t: np.mean([c[t] for c in lcs], 0) for t in ys}
    ld = {t: np.mean([d[t] for d in lds], 0) for t in ys}
    T = {t: fit_temperature(lc[t], ys[t], t == "route") for t in ys}
    return score(ev, responses(ld, T, policy_from(lc, T, cov, band=band)))


def main():
    res = {}
    for k in range(4):
        ev = [json.loads(l) for l in open(f"kaggle/cv_bundle/fold{k}_eval.jsonl")]
        train = {r["request_id"]: r for r in (json.loads(l) for l in open(f"kaggle/cv_bundle/fold{k}_train.jsonl"))}
        a, b = f"kaggle/cv_logits/fold{k}", f"kaggle/ens_logits/ettin17_fold{k}"
        ids = json.load(open(f"{a}/cal_ids.json"))
        assert ids == json.load(open(f"{b}/cal_ids.json")), "holdouts differ"
        cal = [train[i] for i in ids]
        ys = {"route": np.array([QUEUES.index(r["labels"]["route"]) for r in cal]),
              "approve": np.array([r["labels"]["approve"] for r in cal], float),
              "grounded": np.array([r["labels"]["grounded"] for r in cal], float)}
        mc, md = member(dict(np.load(f"{a}/e5_cal.npz")), dict(np.load(f"{a}/e5_dev.npz")), ys)
        ec, ed = member(dict(np.load(f"{b}/cal_logits.npz")), dict(np.load(f"{b}/dev_logits.npz")), ys)
        for name, lcs, lds in (("MiniLM (served setup)", [mc], [md]), ("Ettin-17m", [ec], [ed]),
                               ("MiniLM + Ettin-17m", [mc, ec], [md, ed])):
            for cov in (0.85, 0.82):
                res.setdefault((name, cov), []).append(run(lcs, lds, ys, ev, cov, [0.15, 0.79]))
    base = np.array([s["total"] for s in res[("MiniLM (served setup)", 0.85)]])
    for (name, cov), ss in res.items():
        t = np.array([s["total"] for s in ss])
        print(f"{name:24s} cov {cov}: total {t.mean():.2f} ± {t.std():.2f}  vs served {t.mean() - base.mean():+.2f} "
              f"(better on {int((t > base).sum())}/4)  folds {', '.join('%.2f' % x for x in t)}  "
              f"C {np.mean([s['calibration'] for s in ss]):.3f} A {np.mean([s['abstention'] for s in ss]):.3f} "
              f"min cov {min(min(s['coverage']) for s in ss):.3f}")


if __name__ == "__main__":
    main()
