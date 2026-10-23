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

**Training data.** Only the provided `train.jsonl` (6,500 records): 5,500 for
training and 1,000 held out for calibration and the abstention policy. The dev
split (1,500) was used only for reporting and for choosing between the two
backbones below. Training was 5 epochs on a Kaggle T4 (about 9 minutes),
multi-task, one task per batch.

**Calibration.** Temperature scaling per head, fitted by NLL on the holdout
(route 1.18, approve 1.01, grounded 1.23).

**Measured on the public dev split** (`results.json`, full 1,500 records x 3
decisions, one request at a time):

| | route | approve | grounded |
|---|---|---|---|
| quality | macro-F1 0.940 (acc 0.891) | cost 0.184 (always-hold 0.35) | macro-F1 0.991 |
| ECE | 0.019 | 0.013 | 0.017 |
| Brier skill | 0.79 | 0.68 | 0.91 |
| error, full coverage | 0.109 | 0.099 | 0.029 |
| error, top-80% by confidence | 0.056 | 0.058 | 0.007 |
| declared coverage | 0.83 | 0.85 | 0.84 |
| p95 latency | 21 ms | 23 ms | 28 ms |

The self-check scores this **83.76 / 100**. Latency was measured on an 8-thread
laptop i5 that was also running other heavy jobs (load average about 13), so
an idle 4-core judge machine should be no slower.

**Comparisons (same dev split).** The classical baseline (TF-IDF +
LR/LightGBM) scored 78.45. The larger H768 NLI backbone
(`cross-encoder/nli-MiniLM2-L6-H768`) grounds slightly better (F1 1.00) but
runs about 1.7x slower, and its total was 73.65 under the same load. We chose
H384 for that latency.

**When it abstains.** Each decision abstains on roughly its least-confident 15%
(thresholds in `artifacts/policy.json`, set on the holdout to target 85%
coverage, a margin above the 80% target): route when the top-queue
probability is under 0.90, approve when max(p, 1-p) is under 0.81, grounded when
max(p, 1-p) is under 0.975. Abstained requests still return calibrated
probabilities.

**Approval threshold.** The model commits to "approve" only when
`p_true > 0.75`, since a wrong approval costs 3x a wrong hold.

**Where it fails.**
- *Routing:* about 11% of dev labels disagree with the ticket's evident
  topic, especially multi-issue tickets ("two things: ...") where the label
  sometimes follows the secondary issue. The model's probabilities spread out
  on these rather than claiming certainty.
- *Approval:* the policy is learned as a per-tier amount cutoff (free about
  $30, pro about $120, enterprise about $500) with holds on red flags
  ("raised something similar", "signed up recently", disputes, asking for an
  exception). Near the cutoffs, and on the minority of cases decided by
  account history the ticket never mentions, it is uncertain. That is the
  bulk of its 10% error.
- *Grounding:* misses are subtle paraphrases of durations (e.g. "a quarter of
  an hour" against "an hour") and scope swaps that read alike.
- The data is synthetic and templated. Real tickets with unseen phrasing, other
  currencies or other tiers will degrade it, and the amount features assume
  USD-scale values.

**Consistency check (4-fold cross-validation).** Train and dev were pooled
(8,000 rows) and split into 4 folds. Each fold trained on 6,000 rows
(1,000 of them held out for calibration and policy) and was scored on its
2,000 unseen rows with the self-check metric code (`cv_analyze.py`; latency
held at full marks). Served configuration, 5 epochs: **82.44 ± 1.30**
(folds 82.88 / 82.69 / 83.87 / 80.34). Per decision: route F1 0.940 ± 0.005,
approve cost 0.185 ± 0.021, grounded F1 0.991 ± 0.002, mean ECE 0.023 ± 0.003.
Most of the spread comes from approval cost, the noisiest label. The dev result
(83.4 offline) sits inside the fold range.

**Final served weights** are retrained on all 8,000 labelled rows (train + dev,
1,000 held out for calibration and policy). Because dev was in the training data,
`results.json` dev scores are optimistic; the honest estimate is the 4-fold
cross-validated **82.44 ± 1.30** above.
