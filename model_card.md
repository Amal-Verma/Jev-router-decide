# Model card: Decide

**What it is.** One multi-task model that answers three typed questions about a
customer-support request: `route` (5 queues), `approve` (can it be
auto-approved), `grounded` (is the draft reply supported by the KB article). It
serves on CPU behind `POST /v1/decide` as one ONNX graph (`artifacts/decide.onnx`,
92 MB; `artifacts/` holds only the served ensemble, about 93 MB), computed with one `session.run` per request.

**Architecture.** An ensemble of two complete shared models, merged into that
one graph. Each member answers all three decisions with a shared encoder, a
shared mixture of 4 expert MLPs (a soft gate, conditioned on the pooled text
and a task embedding, mixes them) and one small Linear head per decision.
- Member 1: MiniLM-L6-H384, from `cross-encoder/ms-marco-MiniLM-L6-v2`
  (Apache-2.0), int8.
- Member 2: Ettin-17m, from `jhu-clsp/ettin-encoder-17m` (MIT), kept at fp32,
  because int8 dynamic quantisation broke it (int8 and fp32 gave the same answer
  on only 91% of route, 92% of approve, 81% of grounded cases; MiniLM int8
  agrees with fp32 on 99.7 to 100%).

The decision's logit is the mean of the two members' temperature-scaled logits,
then a final temperature (`decide/runtime.py` `EnsembleBackend`,
`artifacts/ensemble.json`, `fit_ensemble.py`). Each decision has its own input
framing: route reads (subject, body); approve reads ("tier plan, $amount.
subject", body) plus tier/amount/red-flag features; grounded reads
(kb_article, draft_reply) as a cross-encoded pair.

**Size.** About 40.2 M parameters (counted from `artifacts/decide.onnx`):
23.07 M in the MiniLM member and 17.17 M in the Ettin member. Only the final
heads are decision-specific (1,799 weights per member), so over 99.99% is
shared by all three decisions. That is about 10x smaller than the 421 M Laya model.

**Training data.** Only `train.jsonl` (6,500 rows): 5,500 for training and
1,000 held out for calibration and the abstention thresholds, the same rows for
both members. **Dev was never used for training or fitting,** so the dev scores
below are honest. Each member trained for 5 epochs on a Kaggle T4 (seed 13),
multi-task, one task per batch. The Ettin member is a deterministic rerun of an
earlier run with the same code and seed; three reruns reproduced it exactly.

**Calibration.** Temperature scaling by NLL on the holdout, in two steps: one
temperature per member and decision (MiniLM route 1.17, approve 1.03, grounded
1.22; Ettin 1.27, 1.36, 1.15), then one final temperature on the averaged logit
(route 1.00, approve 0.96, grounded 0.84).

**Public dev split** (`results.json`, full self-check, 1,500 records x 3
decisions): **85.44 / 100**. Quality 0.858, calibration 0.869, latency 1.000,
abstention 0.627. Route macro-F1 0.9471, approve cost 0.1786 (always-hold
0.366), grounded macro-F1 0.9983, mean ECE 0.0141, Brier skill 0.8086. Coverage
route 0.838, approve 0.829, grounded 0.837. 4,500 / 4,500 valid.

| Same self-check, other models | Total |
|---|---|
| MiniLM member alone (previously served) | 83.75 |
| Ettin-17m alone, fp32 | 84.68 (approve coverage 0.790) |
| ensemble with Ettin in int8 | 82.24 |

**What to expect on unseen data.** 4-fold cross-validation of the ensemble idea
(fresh fp32 models per fold, each scored on rows it never saw) gave
**83.13 ± 0.53**, against 82.57 ± 1.29 for MiniLM alone and 81.85 for Ettin
alone. The dev 85.44 is partly a favourable Ettin run; an honest hidden-test
expectation is roughly 83 to 85.

**Latency** (self-check, one request at a time, ONNX Runtime with 4 threads, on
a busy laptop with load average 3.5 to 4.5): p95 route 18.1, approve 19.8,
grounded 23.3 ms (mean 20.4 ms). A fresh-venv `scripts/verify_submission.sh`
run gave mean p95 23.0 ms under similar load, and a fresh-clone run 23.8 ms. This is under the 25 ms
full-marks line, with less headroom than the single MiniLM model (10.7 ms).

**When it abstains.** Abstained requests still return calibrated probabilities.
Thresholds live in `artifacts/policy.json`, set on the holdout to answer about
85% of requests (the target is 80%):
- route: when the top-queue probability is under 0.892;
- grounded: when max(p, 1-p) is under 0.9957;
- approve: when 0.15 < p < 0.79. The band is lopsided on purpose, because a
  wrong approval costs 3x a wrong hold.

It commits to "approve" only when `p_true > 0.75` (the 3:1 Bayes threshold).

**Where it fails.**
- *Routing:* about 11% of labels disagree with the ticket's evident topic,
  mostly multi-issue tickets where the label follows the secondary issue.
- *Approval:* the learned policy is a per-tier amount cutoff (free about $30,
  pro about $120, enterprise about $500) plus holds on red flags (repeat claims,
  new accounts, prior disputes, asking for an exception). The leftover errors
  are tickets that look clearly approvable but were held for account history
  the ticket never mentions; no learnable pattern was found for them.
- *Grounding:* it catches changed numbers (durations, day counts) well, but
  hand-written drafts that invent an action ("I have put yours through") or
  narrow the scope ("only on the enterprise plan") were scored as grounded.
- The data is synthetic and templated. Unseen phrasing, other currencies or new
  tiers will degrade it, and the amount features assume USD-scale values.
- *Latency:* the ensemble is about twice as slow as one member, so a heavily
  loaded host can push p95 past 25 ms.
