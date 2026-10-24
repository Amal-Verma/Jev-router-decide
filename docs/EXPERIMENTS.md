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
| 3 | **H384 MiniLM + mixture of experts, lopsided approve band (served)** | **train** | **83.75** | 0.849 | 0.856 | 1.000 | 0.558 | **10.7 ms** (quiet) | **served** |
| 4 | H768 NLI MiniLM + mixture of experts | train | 84.52 | 0.846 | 0.849 | 1.000 | 0.632 | 21.3 ms (quiet) | +0.77, but 2x slower; grounded p95 24.7 ms sits at the 25 ms line |
| 5 | Same as #4, machine busy | train | 73.65 | 0.828 | 0.849 | 0.492 | 0.632 | 63 ms | shows the latency risk |
| 6 | H384 MiniLM + mixture of experts, retrained on train + dev | train + dev | 87.55 | 0.876 | 0.883 | 1.000 | 0.695 | 10.8 ms | **rejected**: inflated, dev was in training |
| 7 | Ensemble: H384 MiniLM + MoE and Ettin-17m + MoE, merged into one ONNX graph | train | 82.78 | 0.846 | 0.821 | 1.000 | 0.562 | 16.8 ms | not served: -0.97 against #3 in the same back-to-back run, mostly calibration |
| 8 | Same as #7, machine busy (load 5.7 to 7.5) | train | 82.20 | 0.846 | 0.821 | 0.971 | 0.562 | 27.2 ms | latency drops below full marks |

The same served model (#3) measured 78.26 while the machine was busy (load 11):
quality, calibration and abstention were identical, and only latency dropped (p95 46 ms).

Rows #3 and #7 were measured back to back on the same machine (load 1.6 to 4 during the runs):
per-decision p95 was route 9.6 / approve 10.5 / grounded 11.8 ms for #3 and 15.0 / 16.6 / 18.7 ms
for #7. Both returned 4,500 of 4,500 valid responses. The ensemble had slightly better route
macro-F1 (0.945 against 0.941) but worse approve cost (0.196 against 0.190) and mean ECE
(0.028 against 0.016).

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

### Ensemble of two encoders (MiniLM + Ettin-17m, one merged ONNX graph)

| Model | Total |
|---|---|
| served setup (H384 MiniLM + MoE) | 82.57 ± 1.29 |
| ensemble, MiniLM + Ettin-17m | **83.13 ± 0.53** |

Cross-validation favoured the ensemble by +0.56, but on the dev split the served model
scored 83.75 against 82.78 (section 1, rows #3 and #7) and runs about 1.6x faster. The
decision rule fixed in advance was to serve the ensemble only if its dev total was within
0.2 of the served model; it was not, so the single MiniLM model stays served.

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

## 5. Architecture, loss and Bayesian variants

All trained on `train.jsonl` only (H384, 5 epochs). Dev rows use 2 seeds (13, 14),
paired against the served MoE with the same seeds. CV rows are paired on the same
4 fold models.

### Architecture and loss (dev, offline, latency at full marks)

| Variant | Seed 13 | Seed 14 | Mean | vs MoE |
|---|---|---|---|---|
| **MoE + temperature (served)** | 83.16 | 84.05 | **83.61** | |
| entropy regularisation 0.05 (Feng et al. 2022) | 83.41 | 83.88 | 83.64 | +0.04 |
| distillation from H768 teacher | 83.03 | 84.23 | 83.63 | +0.02 |
| focal loss, gamma 2 (Mukhoti et al. 2020) | 83.10 | 83.20 | 83.15 | -0.46 |
| label smoothing 0.1 | 82.90 | 83.09 | 83.00 | -0.61 |
| shared MLP trunk (no experts, no gate) | 82.62 | 82.62 | 82.62 | -0.99 |
| linear heads (no middle layer) | 82.89 | 82.26 | 82.58 | -1.03 |
| H768 teacher (1 seed) | 83.12 | | 83.12 | |
| MoE + isotonic calibration | 82.71 | 83.40 | 83.06 | -0.55 |

### Bayesian and post-hoc calibration (4-fold CV, same fold models)

| Variant | Total | vs plain | Folds better |
|---|---|---|---|
| plain (temperature) | 82.54 | | |
| MC dropout, 20 samples on experts + head | 82.63 | +0.09 | 2 of 4 |
| last-layer Laplace (GGN, prior tuned on holdout) | 82.31 | -0.23 | 0 of 4 |
| plain + isotonic binaries | 82.69 | +0.14 | 3 of 4 |
| MC dropout + isotonic | 82.72 | +0.18 | 3 of 4 |
| Laplace + isotonic | 82.68 | +0.14 | 2 of 4 |

On the first CV models, post-hoc variants scored: vector scaling + Platt +0.29,
isotonic binaries +0.41, margin-shaped route + isotonic +0.51. Isotonic is mixed
across tests (+0.14 to +0.51 on CV, -0.55 on dev), because it improves calibration
but flattens the confidence ranking that the abstention score uses. It is not shipped.
The runtime can load such maps (`fit_posthoc.py`, `posthoc.json`), but the served
artifacts contain none.

### Newer encoders, from-scratch network, preprocessing, cleaning, augmentation (dev, paired)

| Variant | Seed 13 | Seed 14 | Mean | vs served MoE |
|---|---|---|---|---|
| Ettin-17m (2025) + vector/Platt calibration | 84.83 | 82.55 | 83.69 | +0.08 |
| Ettin-32m (2025) + vector/Platt calibration | 83.50 | 82.78 | 83.14 | -0.47 |
| DeBERTa-v3-xsmall NLI | | | | dropped: 25 to 31 ms p95 |
| canonical durations ("two weeks" -> "14 days") | 83.21 | 83.20 | 83.20 | -0.40 |
| label cleaning, relabel 563 rows (confident learning) | 82.84 | 82.34 | 82.59 | -1.02 |
| label cleaning, drop 563 rows | 82.47 | 82.28 | 82.38 | -1.23 |
| data augmentation 30% (label-safe) | 82.36 | 82.41 | 82.39 | -1.22 |
| data augmentation 60% (1 seed) | 83.20 | | | +0.04 |
| pure NN from scratch + preprocessing + MoE | 77.39 | 77.97 | 77.68 | -5.9 |

Cleaning and augmentation both make training data cleaner than the test labels, so the
model grows overconfident on noisy-looking cases and the abstention score drops.

H768 speed: graph fusion (before or after int8) gave no reliable speed-up, because the
attention blocks do not match ONNX Runtime's fusable pattern, and fusing the int8 graph
changed outputs. H768 cannot be made meaningfully faster without changing the model.

### Conclusion

No method beats the served model by more than the noise. Bayesian methods model
weight uncertainty, but the remaining errors come from random labels (section 4),
which no training method can learn. The served model stays H384 + MoE + temperature.

Literature consulted: Geifman and El-Yaniv 2017 and the selective classification
benchmark (arXiv 2401.12708); Mukhoti et al. 2020 (arXiv 2002.09437); Feng et al.
2022; Zhu et al. 2022/2023 on BERT and label noise (arXiv 2305.16337); MC dropout
and ensembles on pretrained transformers (arXiv 2601.22068); multi-task loss
weighting survey (arXiv 2404.18961).
