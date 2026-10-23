# Model card: Decide

**What it is.** One multi-task model that answers three typed questions about a
customer-support request: `route` (5 queues), `approve` (can it be
auto-approved), `grounded` (is the draft reply supported by the KB article). It
serves on CPU as a single int8 ONNX graph (24 MB) behind `POST /v1/decide`.

**Architecture.** A shared 6-layer MiniLM cross-encoder (H384, initialised
from `cross-encoder/ms-marco-MiniLM-L6-v2`, Apache-2.0) feeds a shared mixture
of 4 expert MLPs. A soft gate conditioned on the pooled text and a task
embedding mixes the experts, and one Linear head per decision sits on top.
Each decision gets its own input framing: route reads (subject, body);
approve reads ("tier plan, $amount. subject", body) plus 75 structured
tier/amount/red-flag features; grounded reads (kb_article, draft_reply) as a
cross-encoded pair.

**Size.** About 23.07 M parameters, 99.99% shared by all three decisions; only the
final Linear heads (route 1,285, approve 257, grounded 257 weights) are
decision-specific. That is about 18x smaller than the 421 M Laya model.

**Training data.** Only `train.jsonl` (6,500 rows): 5,500 for training and
1,000 held out for calibration and the abstention thresholds. **The served model
never saw `dev.jsonl`,** so the dev scores below are honest. Training ran for 5
epochs on a Kaggle T4 (about 3 minutes), multi-task, one task per batch.

**Calibration.** Temperature scaling per head, fitted by NLL on the holdout
(route 1.18, approve 1.01, grounded 1.23).

**Public dev split** (`results.json`, full self-check, 1,500 records x 3
decisions): **83.75 / 100**. Route macro-F1 0.941, approve cost 0.190
(always-hold 0.35), grounded macro-F1 0.991, mean ECE 0.016, Brier skill 0.79.

**Consistency check (cross-validation).** To check that dev wasn't a lucky
split, separate evaluation-only models were trained on 4 folds of train + dev
(6,000 rows each). Each was scored on the 2,000 rows it never saw, using the
self-check's own metric code (`cv_analyze.py`). These fold models are not
served.

| | mean ± std over 4 folds |
|---|---|
| total (latency at full marks) | **82.56 ± 1.28** (folds 83.1 / 82.9 / 83.9 / 80.5) |
| route macro-F1 | 0.940 ± 0.005 |
| approve cost (3:1) | 0.176 ± 0.017 (always-hold: about 0.38) |
| grounded macro-F1 | 0.991 ± 0.002 |
| mean ECE | 0.023 ± 0.003 |

Dev (83.75) sits inside the fold range, so it is a fair preview of the hidden
test.

**Latency** (self-check, one request at a time, quiet 8-thread laptop i5, ONNX
Runtime with 4 threads): p50 / p95 = route 7.6 / 11.3 ms, approve 8.7 / 12.3 ms,
grounded 10.7 / 13.8 ms.

**When it abstains.** Abstained requests still return calibrated probabilities.
Thresholds live in `artifacts/policy.json`, set on the holdout to answer about 85% of
requests (the target is 80%):
- route: when the top-queue probability is under 0.90;
- grounded: when max(p, 1-p) is under 0.975;
- approve: when 0.15 < p < 0.79. The band is lopsided on purpose, because a
  wrong approval costs 3x a wrong hold. It was chosen on 8,000 out-of-fold
  predictions.

It commits to "approve" only when `p_true > 0.75` (the 3:1 Bayes threshold).

**Where it fails.**
- *Routing:* about 11% of labels disagree with the ticket's evident topic,
  mostly multi-issue tickets ("two things: ...") where the label follows the
  secondary issue. Probabilities spread out on these rather than claiming
  certainty.
- *Approval:* the learned policy is a per-tier amount cutoff (free about $30, pro
  about $120, enterprise about $500) plus holds on red flags (repeat claims, new
  accounts, prior disputes, asking for an exception). An analysis of 8,000
  out-of-fold predictions found no remaining learnable pattern. The leftover errors
  are tickets that look clearly approvable but were held for account history
  the ticket never mentions. The model reports about 0.85 on these, which is
  honest: roughly 15% of such tickets get held.
- *Grounding:* rare misses on subtle duration paraphrases and scope swaps.
- The data is synthetic and templated. Unseen phrasing, other currencies or new
  tiers will degrade it, and the amount features assume USD-scale values.
