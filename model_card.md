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

**Training data.** All 8,000 provided labelled rows (train + dev). 7,000 are
used for training and 1,000 are held out for calibration and the abstention
thresholds. Training ran for 5 epochs on a Kaggle T4 (about 4 minutes),
multi-task, one task per batch. The epoch count, backbone and policy were chosen
by cross-validation (below) before this final fit.

**Calibration.** Temperature scaling per head, fitted by NLL on the holdout
(route 1.10, approve 0.96, grounded 1.06).

**How well it works (the honest estimate).** Train and dev were pooled and split
into 4 folds. Each fold model trained on 6,000 rows and was scored on 2,000 rows
it never saw, using the self-check's own metric code (`cv_analyze.py`):

| | mean ± std over 4 folds |
|---|---|
| total (latency at full marks) | **82.44 ± 1.30** (folds 82.9 / 82.7 / 83.9 / 80.3) |
| route macro-F1 | 0.940 ± 0.005 |
| approve cost (3:1) | 0.185 ± 0.021 (always-hold: about 0.38) |
| grounded macro-F1 | 0.991 ± 0.002 |
| mean ECE | 0.023 ± 0.003 |

`results.json` comes from the self-check on the public dev split: 87.55, route
F1 0.962, approve cost 0.177, grounded F1 0.999, mean ECE 0.013. **Dev was part
of the final training data, so those numbers are optimistic.** Quote the
cross-validated figure above. Before dev was added, the same pipeline trained on
train only scored 83.76 on dev.

**Latency** (self-check, one request at a time, quiet 8-thread laptop i5, ONNX
Runtime with 4 threads): p50 / p95 = route 6.5 / 9.7 ms, approve 7.3 / 10.5 ms,
grounded 8.6 / 12.2 ms.

**When it abstains.** Abstained requests still return calibrated probabilities.
Thresholds live in `artifacts/policy.json`, set on the holdout to answer about 85% of
requests (the target is 80%):
- route: when the top-queue probability is under 0.91;
- grounded: when max(p, 1-p) is under 0.99;
- approve: when 0.15 < p < 0.79. The band is lopsided on purpose, because a
  wrong approval costs 3x a wrong hold. It was chosen on 8,000 out-of-fold
  predictions.

It commits to "approve" only when `p_true > 0.75` (the 3:1 Bayes threshold).

**Where it fails.**
- *Routing:* about 6 to 11% of labels disagree with the ticket's evident topic,
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
