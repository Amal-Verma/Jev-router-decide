"""Post-hoc calibration / confidence variants, scored on the 4 CV folds.

For each fold: fit the calibrator on that fold's 1,000-row holdout, apply it to
the fold's 2,000 unseen rows, set the policy on the holdout, score with the
self-check metric code. Only the probability mapping changes between variants.
"""
import json
import numpy as np
from scipy.optimize import minimize
from sklearn.isotonic import IsotonicRegression

from cv_analyze import policy_from
from offline_score import responses, score
from train_encoder import QUEUES, fit_temperature


def softmax(z):
    e = np.exp(z - z.max(1, keepdims=True))
    return e / e.sum(1, keepdims=True)


def sig(z):
    return 1 / (1 + np.exp(-z))


def nll_multi(params, z, y, k):
    W = params[:k]; b = params[k:]
    p = softmax(z * W + b)
    return -np.mean(np.log(p[np.arange(len(y)), y] + 1e-12))


def fit_vector(z, y):
    k = z.shape[1]
    r = minimize(nll_multi, np.r_[np.ones(k), np.zeros(k)], args=(z, y, k), method="L-BFGS-B")
    return r.x[:k], r.x[k:]


def fit_platt(z, y):
    f = lambda p: -np.mean(y * np.log(sig(p[0] * z + p[1]) + 1e-12) + (1 - y) * np.log(1 - sig(p[0] * z + p[1]) + 1e-12))
    return minimize(f, [1.0, 0.0], method="L-BFGS-B").x


def as_logits(probs_route, pa, pg):
    """Back to logits with temperature 1 so offline_score.responses reproduces them."""
    eps = 1e-7
    return ({"route": np.log(np.clip(probs_route, eps, 1)),
             "approve": np.log(np.clip(pa, eps, 1 - eps) / np.clip(1 - pa, eps, 1)),
             "grounded": np.log(np.clip(pg, eps, 1 - eps) / np.clip(1 - pg, eps, 1))},
            {"route": 1.0, "approve": 1.0, "grounded": 1.0})


def margin_reshape(zc, zd, yc, temp):
    """Keep the argmax and the calibrated distribution shape, but set the top
    probability from an isotonic map of the top-2 logit margin (fit on holdout)."""
    def marg(z):
        s = np.sort(z, 1); return s[:, -1] - s[:, -2]
    pc = softmax(zc / temp)
    iso = IsotonicRegression(out_of_bounds="clip", y_min=0.2, y_max=0.999).fit(marg(zc), (pc.argmax(1) == yc).astype(float))
    pd = softmax(zd / temp)
    top = iso.predict(marg(zd))
    i = pd.argmax(1)
    rest = pd.copy(); rest[np.arange(len(i)), i] = 0
    rest = rest / rest.sum(1, keepdims=True) * (1 - top)[:, None]
    rest[np.arange(len(i)), i] = top
    return rest


def main():
    out = {}
    for k in range(4):
        fold = f"kaggle/cv_logits/fold{k}"
        ev = [json.loads(l) for l in open(f"kaggle/cv_bundle/fold{k}_eval.jsonl")]
        train = {r["request_id"]: r for r in (json.loads(l) for l in open(f"kaggle/cv_bundle/fold{k}_train.jsonl"))}
        cal = [train[i] for i in json.load(open(f"{fold}/cal_ids.json"))]
        yr = np.array([QUEUES.index(r["labels"]["route"]) for r in cal])
        ya = np.array([r["labels"]["approve"] for r in cal], float)
        yg = np.array([r["labels"]["grounded"] for r in cal], float)
        lc = dict(np.load(f"{fold}/e5_cal.npz")); ld = dict(np.load(f"{fold}/e5_dev.npz"))
        T = {"route": fit_temperature(lc["route"], yr, True), "approve": fit_temperature(lc["approve"], ya, False),
             "grounded": fit_temperature(lc["grounded"], yg, False)}
        base_pr_c, base_pr_d = softmax(lc["route"] / T["route"]), softmax(ld["route"] / T["route"])
        pa_c, pa_d = sig(lc["approve"] / T["approve"]), sig(ld["approve"] / T["approve"])
        pg_c, pg_d = sig(lc["grounded"] / T["grounded"]), sig(ld["grounded"] / T["grounded"])
        variants = {}
        variants["temperature (served)"] = ((base_pr_c, pa_c, pg_c), (base_pr_d, pa_d, pg_d))
        W, b = fit_vector(lc["route"], yr)
        aa, ab = fit_platt(lc["approve"], ya); ga, gb = fit_platt(lc["grounded"], yg)
        vs = lambda z: softmax(z * W + b)
        variants["vector + Platt"] = ((vs(lc["route"]), sig(aa * lc["approve"] + ab), sig(ga * lc["grounded"] + gb)),
                                      (vs(ld["route"]), sig(aa * ld["approve"] + ab), sig(ga * ld["grounded"] + gb)))
        isa = IsotonicRegression(out_of_bounds="clip", y_min=1e-3, y_max=1 - 1e-3).fit(pa_c, ya)
        isg = IsotonicRegression(out_of_bounds="clip", y_min=1e-3, y_max=1 - 1e-3).fit(pg_c, yg)
        variants["temp route + isotonic binary"] = ((base_pr_c, isa.predict(pa_c), isg.predict(pg_c)),
                                                    (base_pr_d, isa.predict(pa_d), isg.predict(pg_d)))
        variants["vector route + isotonic binary"] = ((vs(lc["route"]), isa.predict(pa_c), isg.predict(pg_c)),
                                                      (vs(ld["route"]), isa.predict(pa_d), isg.predict(pg_d)))
        isa2 = IsotonicRegression(out_of_bounds="clip", y_min=1e-3, y_max=1 - 1e-3).fit(sig(aa * lc["approve"] + ab), ya)
        isg2 = IsotonicRegression(out_of_bounds="clip", y_min=1e-3, y_max=1 - 1e-3).fit(sig(ga * lc["grounded"] + gb), yg)
        variants["vector route + Platt->isotonic"] = ((vs(lc["route"]), isa2.predict(sig(aa * lc["approve"] + ab)), isg2.predict(sig(ga * lc["grounded"] + gb))),
                                                      (vs(ld["route"]), isa2.predict(sig(aa * ld["approve"] + ab)), isg2.predict(sig(ga * ld["grounded"] + gb))))
        mr_c = margin_reshape(lc["route"], lc["route"], yr, T["route"]); mr_d = margin_reshape(lc["route"], ld["route"], yr, T["route"])
        variants["margin-shaped route"] = ((mr_c, pa_c, pg_c), (mr_d, pa_d, pg_d))
        variants["margin route + isotonic binary"] = ((mr_c, isa.predict(pa_c), isg.predict(pg_c)), (mr_d, isa.predict(pa_d), isg.predict(pg_d)))
        for name, (c, d) in variants.items():
            lgc, tt = as_logits(*c); lgd, _ = as_logits(*d)
            pol = policy_from(lgc, tt, 0.85, band=[0.15, 0.79])
            out.setdefault(name, []).append(score(ev, responses(lgd, tt, pol)))
    for name, ss in out.items():
        t = np.array([s["total"] for s in ss])
        print(f"{name:30s} total {t.mean():.2f} ± {t.std():.2f}   Q {np.mean([s['quality'] for s in ss]):.3f}  "
              f"C {np.mean([s['calibration'] for s in ss]):.3f}  A {np.mean([s['abstention'] for s in ss]):.3f}  "
              f"ECE {np.mean([s['mean_ece'] for s in ss]):.4f}  per-fold {', '.join('%.2f' % x for x in t)}")


if __name__ == "__main__":
    main()
