# Three-minute demo script

Start: `uvicorn app:app --host 0.0.0.0 --port 8000`

## 1. It works (about 45 s)

```bash
curl -s localhost:8000/v1/decide -H 'content-type: application/json' -d '{
 "request_id":"d1","decision":"approve",
 "state":{"subject":"Refund please","body":"Please refund $12.","customer_tier":"pro","amount":12.0}}'
```
This auto-approves (`p_true` about 0.87, `value: true`). Then run the same
state with `"decision":"route"` to show that the one model answers a different
question.

## 2. It refuses when it should (about 60 s)

A Pro customer at $112, just under the learned Pro cutoff (about $120), with a
ticket that is not even about money:

```bash
curl -s localhost:8000/v1/decide -H 'content-type: application/json' -d '{
 "request_id":"d2","decision":"approve",
 "state":{"subject":"bulk upload takes minutes to load",
          "body":"Hi team, Loading a normal report now takes several minutes. Is something wrong on your side? Any help would be great.",
          "customer_tier":"pro","amount":112.37}}'
```
Returns `p_true` about 0.31 and `abstain: true`. That is near the threshold, the
model knows it, and it hands the case to a human instead of guessing. (In the
labelled data the answer was "hold".)

## 3. One trade-off (about 60 s)

We trained two backbones on the same multi-task setup. The larger H768 NLI
encoder grounds perfectly on dev (F1 1.00 vs 0.99) but is about 1.7x slower.
Latency is 20% of the score, and full marks need p95 of 25 ms or less, so we
serve the H384 model: 83.75 total vs 73.65 on dev (cross-validated 82.6 ± 1.3).

Backup line if asked about sharing: every request runs the same encoder and
the same 4 mixture-of-experts MLPs; only the final Linear layer differs per
decision.
