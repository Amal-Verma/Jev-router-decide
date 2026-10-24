# Experiments: score against approach

Every number here was measured in this project with the judges' own scoring code
(`selfcheck.py`, or `offline_score.py`, which reuses its metric functions).
Totals are out of 100. Weights: quality 40%, calibration 25%, latency 20%,
abstention 15%.

## 1. Full self-check on the public dev split (CPU)

These are end-to-end runs through the real web service.

| # | Approach | Trained on | Total | Quality | Calibration | Latency | Abstention | Mean p95 | Verdict |
|---|---|---|---|---|---|---|---|---|---|
| 1 | Classical baseline: TF-IDF + logistic regression (route, grounded), LightGBM + tabular features (approve) | train | 78.45 | 0.783 | 0.760 | 1.000 | 0.542 | 22.5 ms | gate-1 baseline |
| 2 | H384 MiniLM + mixture of experts, symmetric abstention | train | 83.76 | 0.850 | 0.856 | 1.000 | 0.558 | ~24 ms (busy machine) | superseded by #3 |
| 3 | H384 MiniLM + mixture of experts, lopsided approve band | train | 83.75 | 0.849 | 0.856 | 1.000 | 0.558 | 10.7 ms (quiet) | served until the final ensemble (#9) |
| 4 | H768 NLI MiniLM + mixture of experts | train | 84.52 | 0.846 | 0.849 | 1.000 | 0.632 | 21.3 ms (quiet) | +0.77, but 2x slower; grounded p95 24.7 ms sits at the 25 ms line |
| 5 | Same as #4, machine busy | train | 73.65 | 0.828 | 0.849 | 0.492 | 0.632 | 63 ms | shows the latency risk |
| 6 | H384 MiniLM + mixture of experts, retrained on train + dev | train + dev | 87.55 | 0.876 | 0.883 | 1.000 | 0.695 | 10.8 ms | **rejected**: inflated, dev was in training |
| 7 | Ensemble: H384 MiniLM + MoE and Ettin-17m + MoE, merged into one ONNX graph | train | 82.78 | 0.846 | 0.821 | 1.000 | 0.562 | 16.8 ms | not served: -0.97 against #3 in the same back-to-back run, mostly calibration |
| 8 | Same as #7, machine busy (load 5.7 to 7.5) | train | 82.20 | 0.846 | 0.821 | 0.971 | 0.562 | 27.2 ms | latency drops below full marks |
| 9 | **Ensemble: MiniLM + MoE (int8) and Ettin-17m + MoE (fp32), one merged graph** | **train** | **85.44** | 0.858 | 0.869 | 1.000 | 0.627 | **20.4 ms** (busy, load 3.5 to 4.5) | **served** (section 6) |

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

Cross-validation favoured the ensemble by +0.56, with less spread. At first, on the dev
split, the single MiniLM model scored 83.75 against 82.78 for the ensemble (section 1,
rows #3 and #7), so the single model stayed served. Section 6 explains why that ensemble
was weak (its Ettin member was int8) and how the fixed ensemble became the served model.

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

No training or calibration method beats H384 + MoE + temperature by more than the
noise. Bayesian methods model weight uncertainty, but the remaining errors come from
random labels (section 4), which no training method can learn. The one change that
did help is the ensemble with Ettin-17m, once Ettin is kept at full precision
(section 6).

## 6. The int8 discovery and the final ensemble

### Ettin does not survive int8

The ensemble in section 1 (#7) used both members in int8, as every model here had been
served. Comparing each member's int8 graph with its fp32 graph on the same inputs showed
the problem:

| Member | Share of cases where int8 and fp32 give the same answer |
|---|---|
| MiniLM-L6-H384 + MoE | 99.7 to 100% across the three decisions |
| Ettin-17m + MoE | route 91%, approve 92%, grounded 81% |

Dynamic int8 quantisation leaves MiniLM essentially unchanged but changes up to a fifth
of Ettin's answers. So the earlier ensembles were judged with a damaged Ettin member.

### Full self-check on dev (same selfcheck, train-only models)

| Model | Total | Note |
|---|---|---|
| MiniLM + MoE alone (int8, previously served) | 83.75 | mean p95 10.7 ms |
| Ettin-17m + MoE alone, fp32 | 84.68 | mean p95 12.7 ms, but approve coverage 0.790 (under 80%) |
| MiniLM + Ettin-17m, Ettin int8 | 82.24 | int8 breaks Ettin |
| MiniLM + Ettin-17m, earlier ensemble with a different Ettin run | 82.78 | row #7 |
| **MiniLM (int8) + Ettin-17m (fp32), one merged graph (served)** | **85.44** | mean p95 20.4 ms on a busy laptop |

Served ensemble in detail: quality 0.858, calibration 0.869, latency 1.000, abstention
0.627; route macro-F1 0.9471, approve cost 0.1786, grounded macro-F1 0.9983, mean ECE
0.0141, Brier skill 0.8086; coverage route 0.838, approve 0.829, grounded 0.837;
4,500 of 4,500 valid. p95 per decision: route 18.1, approve 19.8, grounded 23.3 ms,
measured with load average 3.5 to 4.5. A fresh-venv `scripts/verify_submission.sh` run
gave PASS, 85.44, mean p95 23.0 ms under similar load.

How it is built: both members are trained on `train.jsonl` only (seed 13) and calibrated
on the same 1,000-row train holdout. Each member is a full shared-backbone model that
answers all three decisions. One `session.run` computes both; the decision's logit is the
mean of the two members' temperature-scaled logits, then a final temperature
(`decide/runtime.py` `EnsembleBackend`, `artifacts/ensemble.json`, `fit_ensemble.py`).
Dev is never used for training or fitting. The Ettin member is a deterministic rerun of an
earlier Ettin training run (same code, seed 13); three reruns reproduced it exactly.

### How much of the dev score is luck

| 4-fold cross-validation (fresh fp32 models per fold) | Total |
|---|---|
| MiniLM alone | 82.57 ± 1.29 |
| Ettin-17m alone | 81.85 (mean) |
| **MiniLM + Ettin-17m ensemble** | **83.13 ± 0.53** |

The ensemble is the best and steadiest option in cross-validation, but its +1.69 on dev
is larger than its +0.56 in cross-validation, so the dev 85.44 is partly a favourable
Ettin run. An honest hidden-test expectation is roughly 83 to 85.

### Latency risk

Mean p95 is about 20 to 23 ms on a busy laptop, under the 25 ms full-marks line but with
less headroom than the single MiniLM model (10.7 ms). The judging host should be kept
otherwise idle.

Literature consulted: Geifman and El-Yaniv 2017 and the selective classification
benchmark (arXiv 2401.12708); Mukhoti et al. 2020 (arXiv 2002.09437); Feng et al.
2022; Zhu et al. 2022/2023 on BERT and label noise (arXiv 2305.16337); MC dropout
and ensembles on pretrained transformers (arXiv 2601.22068); multi-task loss
weighting survey (arXiv 2404.18961).
