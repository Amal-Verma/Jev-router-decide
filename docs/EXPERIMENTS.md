# Experiments: score against approach

Every number here was measured in this project with the judges' own scoring code
(`selfcheck.py`, or `offline_score.py`, which reuses its metric functions).
Totals are out of 100. Weights: quality 40%, calibration 25%, latency 20%,
abstention 15%.

## 1. Full self-check on the public dev split (served int8 model, CPU)

These are end-to-end runs through the real web service.

| # | Approach | Trained on | Total | Quality | Calibration | Latency | Abstention | Mean p95 | Verdict |
|---|---|---|---|---|---|---|---|---|---|
| 1 | Classical baseline: TF-IDF + logistic regression (route, grounded), LightGBM + tabular features (approve) | train | 78.45 | 0.783 | 0.760 | 1.000 | 0.542 | 22.5 ms | gate-1 baseline |
| 2 | H384 MiniLM + mixture of experts, symmetric abstention | train | 83.76 | 0.850 | 0.856 | 1.000 | 0.558 | ~24 ms (busy machine) | superseded by #3 |
| 3 | **H384 MiniLM + mixture of experts, lopsided approve band (served)** | **train** | **83.75** | 0.849 | 0.856 | 1.000 | 0.558 | **10.6 ms** (quiet) | **served** |
| 4 | H768 NLI MiniLM + mixture of experts | train | 84.52 | 0.846 | 0.849 | 1.000 | 0.632 | 21.3 ms (quiet) | +0.77, but 2x slower; grounded p95 24.7 ms sits at the 25 ms line |
| 5 | Same as #4, machine busy | train | 73.65 | 0.828 | 0.849 | 0.492 | 0.632 | 63 ms | shows the latency risk |
| 6 | H384 MiniLM + mixture of experts, retrained on train + dev | train + dev | 87.55 | 0.876 | 0.883 | 1.000 | 0.695 | 10.8 ms | **rejected**: inflated, dev was in training |

The same served model (#3) measured 78.26 while the machine was busy (load 11):
quality, calibration and abstention were identical, and only latency dropped (p95 46 ms).

## 2. Offline score on dev (fp32 predictions, latency held at full marks)

| Approach | Total | Note |
|---|---|---|
| H384 + mixture of experts | 83.38 | int8 served version: 83.75 (they agree within 0.4) |
| H768 + mixture of experts | 85.02 | better abstention 0.627 against 0.525 |

## 3. Four-fold cross-validation (8,000 rows, each fold scored on 2,000 unseen rows)

Evaluation-only models, never served. fp32, H384 + mixture of experts.

### Training length (coverage target 0.85)

| Epochs | Total |
|---|---|
| 1 | 68.53 ± 4.21 |
| 2 | 80.05 ± 0.79 |
| 3 | 81.74 ± 0.69 |
| 4 | 82.17 ± 1.24 |
| **5** | **82.44 ± 1.30** |

### Abstention policy (5 epochs)

| Policy | Total | Lowest coverage on any fold |
|---|---|---|
| coverage target 0.82 | 82.50 ± 1.25 | 0.799 (below 80%, risky) |
| coverage target 0.85 | 82.44 ± 1.30 | 0.832 |
| coverage target 0.88 | 82.32 ± 1.31 | 0.866 |
| approve threshold tuned on holdout | 82.44 ± 1.30 | no gain over 0.75 |
| **lopsided approve band 0.15 to 0.79 (served)** | **82.56 ± 1.28** | 0.816 |
| served + grounded coverage 0.92 | 82.49 ± 1.27 | 0.816 |

### Article-aware routing (test of a found signal)

On vague tickets the KB article predicts the route label 68% of the time, against
30% for the ticket-only model. Using it was still a loss, because those tickets are
already abstained on:

| Variant | Total |
|---|---|
| served (no article signal) | **82.56 ± 1.27** |
| article model blended into every ticket | 81.36 ± 1.28 |
| article model on vague tickets only (best gate) | 81.82 ± 1.40 |

### Second split (different shuffle, accuracy only, stopped early)

| Model | Route | Approve | Grounded |
|---|---|---|---|
| H384 (4 folds) | 0.885 to 0.895 | 0.894 to 0.909 | 0.973 to 0.977 |
| H768 (2 folds) | 0.894 to 0.899 | 0.891 to 0.905 | 0.990 to 0.995 |

## 4. Signal hunt (what limits the score)

| Question | Test | Finding |
|---|---|---|
| Are route "wrong" labels predictable? | flip rate among confident cases, split by every field | random: about 5.7% everywhere |
| Is approve's hidden history recoverable? | hold rate among "looks fine" cases, split by every field; repeat customers | random: about 15.5% everywhere; customers barely repeat |
| Do article or draft explain approve? | logistic regression on article and draft text | no: AUC 0.50 |
| Do article or draft explain route? | logistic regression on article and draft text | yes on vague tickets (68%), but no score gain (section 3) |

Estimated honest ceiling on unseen data: about 85 to 86.

## 5. Architecture comparison (pending)

Running on Kaggle: mixture of experts against a shared MLP trunk, plain linear
heads, and distillation from the H768 teacher, 2 seeds each, trained on train only.
Results will be added here.
