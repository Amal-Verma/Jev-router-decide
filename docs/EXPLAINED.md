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
    "draft_reply": "You can request a refund within 60 days of the charge on any plan. Once approved, the money goes back to your original payment method within 10 business days.",
    "kb_article": "Refunds may be requested within 30 days of the charge, and this applies to all plans. Approved refunds are returned to the original payment method within 10 business days." } }
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
| `route` | `value: "billing"`, `confidence: 0.888`, `abstain: true`, `probs` for all 5 queues | billing, 89% sure; just under the 0.892 bar, so a human confirms |
| `approve` | `value: true`, `p_true: 0.883`, `abstain: false` | auto-approve: pro tier, $45 is under the ~$120 limit, no red flags |
| `grounded` | `value: false`, `p_true: 0.055`, `abstain: true` | the draft says 60 days; the article says 30 |

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

The served model is an **ensemble**: two complete models that each answer all
three questions, packed into **one** ONNX file and computed in **one** call.

```
request -> framing -> 2 tokenizers -> [ MiniLM member | Ettin member ] -> average -> calibration -> policy -> answer
                                       (each: ENCODER -> experts + gate -> head)
```

1. **Framing** (`decide/inputs.py`). Each question sees what it needs: route reads
   subject + body; approve reads "pro plan, amount $45. subject" + body plus
   numbers for tier, amount above each cutoff and red flags; grounded reads
   kb_article + draft_reply side by side plus numbers for whether the durations
   match.
2. **Tokenizers.** Words become IDs. Each member has its own dictionary
   (`artifacts/tok_m`, `artifacts/tok_e`).
3. **Encoders.** Every word looks at every other word (attention), so meaning
   comes from context. The output is averaged into one summary. Member 1 is
   MiniLM (6 layers, 384 numbers wide, 22.6 M weights, stored as int8). Member 2
   is Ettin-17m (7 layers, 256 wide, 16.8 M weights, kept at full precision).
   **Each encoder is shared by all three questions.**
4. **Mixture of experts**, one per member (4 expert layers + a gate). The gate
   decides how much to trust each expert for this question. **Shared.**
5. **Heads.** One tiny layer per question per member gives the raw score: route 5
   scores (1,285 weights), approve 1 score (257), grounded 1 score (257). This is
   the only per-question part.
6. **Averaging and calibration.** Each member's raw score is divided by its own
   temperature, the two are averaged, and the average is divided by a final
   temperature (route 1.00, approve 0.96, grounded 0.84). Softmax (route) or
   sigmoid (binary) then gives honest probabilities.
7. **Policy** (`model.py`): commit to an answer and decide whether to abstain
   (section 7).

**Size:** about 40.2 M parameters (23.07 M MiniLM member + 17.17 M Ettin member),
over 99.99% shared across the three questions. That is about 10x smaller than the
421 M Laya model the brief mentions.

**Why two members?** They make different mistakes, so averaging helps. In
4-fold cross-validation the pair scored 83.13 ± 0.53 against 82.57 ± 1.29 for
MiniLM alone. The price is about twice the latency.

## 6. How `grounded` is calculated (worked example)

Same article as in section 2, three drafts, real numbers from the served model:

| Draft | MiniLM raw | Ettin raw | p_true | Why |
|---|---|---|---|---|
| faithful rewrite: 30 days, any plan, 10 business days | +11.91 | +13.10 | 1.00 | everything is supported |
| same, but "60 days" | -1.90 | -3.70 | 0.06 | wrong number |
| same, but "3 business days" | -6.96 | -13.58 | 0.00003 | wrong number |

For the "60 days" draft: MiniLM -1.90 / 1.216 = -1.56 and Ettin -3.70 / 1.149 =
-3.22; their average is -2.39; divided by the final temperature 0.842 that is
-2.84, and the sigmoid turns it into 0.055. Then value = 0.055 > 0.5, which is
false. Abstain is max(0.055, 0.945) < 0.9957, which is true.

What it misses: drafts that invent an action ("so I have put yours through")
or narrow the scope ("but only on the enterprise plan") were both scored as
grounded (p_true about 1.00). The model learned the kinds of mismatch in the
training data, which are mostly changed numbers.

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
| route | 10.7% | 5.1% |
| approve | 10.5% | 6.0% |
| grounded | 1.2% | 0.08% |

Our abstention score is 0.63. It is capped by noisy labels and hidden account
history: some mistakes look certain because nothing in the ticket reveals them.

Thresholds (`artifacts/policy.json`): route abstains when its top probability is
under 0.892; grounded when it is under 99.57% sure either way; approve when
0.15 < p < 0.79. That band is lopsided because risky approvals are the costly
mistakes.

## 8. Training

1. 6,500 tickets from `train.jsonl`: 5,500 to train, 1,000 held out (the same
   rows for both members). **Dev is never used for training or fitting.**
2. Each ticket becomes 3 training items (one per question), 16,500 in total.
3. Batches of 32 items from one question pass through the model. The error
   (loss) is measured and all the member's weights are nudged to reduce it.
   Because the encoder is shared, every question improves the same brain.
4. 5 passes over the data, seed 13, on a free Kaggle T4 GPU. This is done
   twice, once per member. The Ettin member is a deterministic rerun of an
   earlier run; three reruns reproduced it exactly.
5. The temperatures and abstain thresholds are fitted on the 1,000 held-out
   tickets (`fit_ensemble.py`).
6. Export to ONNX. MiniLM is compressed to int8; Ettin stays fp32. The two are
   merged into one graph.

## 9. fp32 against int8

| | fp32 (32-bit) | int8 (8-bit, quantized) |
|---|---|---|
| bytes per weight | 4 | 1 |
| used for | training, and the served Ettin member | the served MiniLM member |

int8 makes a model about 4x smaller and usually faster, at a small cost in
accuracy. For MiniLM the cost is tiny: int8 and fp32 give the same answer on
99.7 to 100% of cases. For Ettin it is not: they agree on only 91% of route,
92% of approve and 81% of grounded cases. An ensemble with Ettin in int8 scored
82.24 on dev; with Ettin at fp32 it scored 85.44. So Ettin ships at fp32, and
the merged file is 92 MB.

| Test | Precision |
|---|---|
| official dev score 85.44, latency, schema check, demo | the served ensemble (MiniLM int8 + Ettin fp32) |
| cross-validation, second split, backbone and policy experiments | fp32 (predictions saved during GPU training) |

## 10. Was it tested on other splits?

| Test | Result |
|---|---|
| dev split (served ensemble, never trained on dev) | **85.44** |
| dev split, MiniLM member alone | 83.75 |
| 4-fold cross-validation of the ensemble (fresh fp32 models per fold) | 83.13 +- 0.53 |
| 4-fold cross-validation, MiniLM alone (same folds) | 82.57 +- 1.29 |
| 4-fold, shuffle B (stopped early; accuracy only) | route 0.885 to 0.895, approve 0.894 to 0.909, grounded 0.973 to 0.977 |

The ensemble's dev gain (+1.69) is larger than its cross-validated gain (+0.56),
so part of the dev 85.44 is a favourable Ettin run. The honest expected
hidden-test score is roughly 83 to 85. Most of the spread comes from approve,
the noisiest label.

## 11. Key decisions

| Decision | Why |
|---|---|
| ensemble of MiniLM and Ettin-17m | best and steadiest in cross-validation (83.13 ± 0.53); costs about 2x latency, still under 25 ms p95 |
| Ettin kept at fp32 | int8 changed up to 19% of its answers |
| small H384 encoder over the larger H768 | H768 grounds slightly better but is about 1.7x slower, which risks the latency score |
| never train on dev | keeps dev scores honest and respects the "no memorising dev" rule |
| lopsided approve abstention band | a wrong approval costs 3x a wrong hold |
| 5 epochs, 85% coverage target | best and safest settings in cross-validation |
