# Decide, explained from zero

A plain-language guide to what this project does, how it works and how it was
tested. Read this first if you are new to it.

## 1. The problem

Support software makes many small decisions on every ticket: which team should
handle it, can a refund be auto-approved, is the drafted reply backed by the help
article. Many products ask a large LLM these questions. That is slow (0.3 to 2 s),
costs money per call, and comes back with no usable confidence. The hackathon asks
for a small model that answers these typed questions in one pass, on a CPU, in
milliseconds, with an honest probability.

## 2. Input and output

You send one ticket and name **one** question:

```json
{ "request_id": "demo-1",
  "decision": "approve",
  "state": {
    "subject": "Refund for last month",
    "body": "Hi, I was charged $45.00 for a month I didn't use. Can you refund it?",
    "customer_tier": "pro",
    "amount": 45.0,
    "draft_reply": "Refunds can be requested within 30 days of the charge, so I have put yours through...",
    "kb_article": "Refunds may be requested within 30 days of the charge, and this applies to all plans..." } }
```

| Field | Meaning |
|---|---|
| `subject`, `body` | what the customer wrote |
| `customer_tier` | free, pro or enterprise |
| `amount` | money involved |
| `draft_reply` | a reply someone drafted |
| `kb_article` | the help article that reply is supposed to rely on (**given in the request**; we never search for it) |

The same ticket, asked all three questions (real output of the served model):

| Question | Output | Meaning |
|---|---|---|
| `route` | `value: "billing"`, `confidence: 0.906`, `probs` for all 5 queues | send to billing, 91% sure |
| `approve` | `value: true`, `p_true: 0.889`, `abstain: false` | auto-approve: pro tier, $45 is under the ~$120 limit, no red flags |
| `grounded` | `value: false`, `p_true: 0.026`, `abstain: true` | the draft says the refund is done; the article only says refunds may be requested |

Every answer has `value` (the answer), `probs` or `p_true` (the honest
probability), `abstain` (true = "let a human check") and `latency_ms`.

## 3. The rules that make it hard

1. **One model for all three questions**, sharing a backbone; not three models
   behind a switch.
2. **CPU only, no network** while answering.
3. **Fast:** p95 under 100 ms; full marks at 25 ms or less.
4. **Calibrated:** "0.8" must be right about 80% of the time.
5. **Can abstain**, but must answer at least 80%.
6. **Approve has a 3:1 cost:** a wrong approval costs 3x a wrong hold.
7. **The approve rule is secret:** it must be learned from the labels, and part of
   it depends on account history the ticket never shows.

### Point 6: why approve uses 0.75, not 0.5

At probability p that the request is fine:
- approving costs 3 x (1 - p) on average (wrong 1 - p of the time, cost 3);
- holding costs p (wrong p of the time, cost 1).

They are equal at **p = 0.75**, so the model approves only above 0.75. At p = 0.6,
approving costs 1.2 against 0.6 for holding.

### Point 7: the rule we recovered from the data

| Tier | Usually auto-approved up to |
|---|---|
| free | about $30 |
| pro | about $120 |
| enterprise | about $500 |

Red flags force a hold at any amount: "raised something similar", "signed up
recently", "disputed a charge with my bank", "outside your usual policy... an
exception". Even when everything looks fine, about 20% of such tickets are still
held because of hidden account history. The model says about 0.85 on those, which is
honest.

## 4. How it is scored

Pass/fail first: valid schema on every response, p95 under 100 ms, a model card,
no memorising dev. Then teams are ranked on a hidden test set:

| Axis | Weight | Meaning |
|---|---|---|
| Quality | 40% | right answers on the cases you answered (route and grounded macro-F1, approve 3:1 cost) |
| Calibration | 25% | half ECE (is 0.8 really 0.8?), half Brier skill (better than quoting the base rate?) |
| Latency | 20% | p95 per call; full marks at 25 ms or less |
| Abstention | 15% | does your confidence point at your own mistakes? |

## 5. The architecture, step by step

```
request -> framing -> tokenizer -> ENCODER -> experts + gate -> head -> calibration -> policy -> answer
```

1. **Framing** (`decide/inputs.py`). Each question sees what it needs: route reads
   subject + body; approve reads "pro plan, amount $45. subject" + body plus 75
   numbers (tier, amount above each cutoff, red flags); grounded reads kb_article +
   draft_reply side by side plus 3 numbers (do the durations match?).
2. **Tokenizer.** Words become IDs from a 30,000-piece dictionary:
   "refunds" becomes `ref`, `##unds`.
3. **Encoder** (MiniLM, 6 Transformer layers, 22.6 M weights). Every word looks at
   every other word (attention), so meaning comes from context. The output is
   averaged into one summary of 384 numbers. **Shared by all questions.**
4. **Mixture of experts** (4 expert layers + a gate, 0.5 M weights). The gate decides
   how much to trust each expert for this question. **Shared.**
5. **Head.** One tiny layer per question gives the raw score: route 5 scores
   (1,285 weights), approve 1 score (257), grounded 1 score (257). This is the
   only per-question part.
6. **Calibration.** Divide by a temperature (route 1.18, approve 1.01, grounded
   1.23), then softmax (route) or sigmoid (binary) to get honest probabilities.
7. **Policy** (`model.py`): commit to an answer and decide whether to abstain
   (section 7).

**Size:** about 23.07 M parameters, 99.99% shared. That is about 18x smaller than
the 421 M Laya model the brief mentions.

## 6. How `grounded` is calculated (worked example)

Same article, three drafts, real numbers from the served model:

| Draft | Raw score | p_true | Why |
|---|---|---|---|
| "...I have put yours through", 30 days, 10 business days | -4.47 | 0.03 | claims the refund is approved; the article never says so |
| faithful rewrite: 30 days, approved refunds, 10 business days | +12.98 | 1.00 | everything is supported |
| faithful rewrite but "60 days" | -2.10 | 0.15 | wrong number |

For the first draft: the raw score -4.47, divided by temperature 1.233, gives
-3.62; the sigmoid turns that into 0.026. Then value = 0.026 > 0.5, which is false.
Abstain is max(0.026, 0.974) < 0.975, which is just true.

## 7. Abstention

Abstention is the model saying "not sure, let a human decide". It works in two
separate ways:

- **The `abstain` flag** sets what you are graded on. Abstained cases leave the
  quality score, but answering under 80% scales quality down. Ours answers about
  85%.
- **The abstention score** ignores the flag. Judges sort every case by your
  confidence, keep the top 80%, and compare the error there with the error overall.

| Decision | Error, all cases | Error, most confident 80% |
|---|---|---|
| route | 10.9% | 5.6% |
| approve | 9.9% | 5.7% |
| grounded | 2.9% | 0.7% |

Our abstention score is 0.56. It is capped by noisy labels and hidden account
history: some mistakes look certain because nothing in the ticket reveals them.

Thresholds (`artifacts/policy.json`): route abstains when its top probability is
under 0.90; grounded when it is under 97.5% sure either way; approve when
0.15 < p < 0.79. That band is lopsided because risky approvals are the costly
mistakes.

## 8. Training

1. 6,500 tickets from `train.jsonl`: 5,500 to train, 1,000 held out. **Dev is
   never trained on.**
2. Each ticket becomes 3 training items (one per question), 16,500 in total.
3. Batches of 32 items from one question pass through the model. The error
   (loss) is measured and all 23 M weights are nudged to reduce it. Because the
   encoder is shared, every question improves the same brain.
4. 5 passes over the data, about 3 minutes on a free Kaggle T4 GPU.
5. Temperatures and abstain thresholds are fitted on the 1,000 held-out tickets.
6. Export to ONNX and compress to int8.

## 9. fp32 against int8

| | fp32 (32-bit) | int8 (8-bit, quantized) |
|---|---|---|
| bytes per weight | 4 | 1 |
| file size | about 92 MB | **24 MB** |
| used for | training | **serving** (what the judges run) |

The fp32 export is deleted after compression; only int8 ships. Which tests used which:

| Test | Precision |
|---|---|
| official dev score 83.75, latency, schema check, demo | **int8** (the served model) |
| cross-validation, second split, backbone and policy experiments | fp32 (predictions saved during GPU training) |

On the same dev split the two agree within about 0.4 points (83.38 fp32 against
83.75 int8). Quality and calibration are essentially identical; abstention moves
about 0.03.

## 10. Was it tested on other splits?

| Test | Result |
|---|---|
| dev split (served model, never trained on dev) | **83.75** |
| 4-fold cross-validation, shuffle A (evaluation-only models) | 82.56 +- 1.28 (folds 83.1 / 82.9 / 83.9 / 80.5) |
| 4-fold, shuffle B (stopped early; accuracy only) | route 0.885 to 0.895, approve 0.894 to 0.909, grounded 0.973 to 0.977 |

Dev sits inside the fold range, and a second shuffle gives the same accuracy. The
expected hidden-test score is about 82 to 84. Most of the spread comes from approve,
the noisiest label.

## 11. Key decisions

| Decision | Why |
|---|---|
| small H384 encoder over the larger H768 | H768 grounds slightly better but is about 1.7x slower, which risks the latency score |
| never train on dev | keeps dev scores honest and respects the "no memorising dev" rule |
| lopsided approve abstention band | a wrong approval costs 3x a wrong hold |
| 5 epochs, 85% coverage target | best and safest settings in cross-validation |
