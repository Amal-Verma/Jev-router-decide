# Dataset audit

The supplied `data/train.jsonl` and `data/dev.jsonl` provide 8,000 support requests with all three required labels. No public substitute or generated examples are needed. This audit is descriptive; it does not select model features, thresholds, or hyperparameters using the development labels.

Run the audit from the repository root:

```bash
python audit_data.py
```

The script uses only Python's standard library and writes `evaluation/data_audit.json`. Use `--train`, `--dev`, and `--output` to override paths. Hashes in the report identify the exact input files. Default paths resolve relative to the script, so an absolute script path also works from other directories.

## Schema and integrity

| Check | Training | Development |
|---|---:|---:|
| Records | 6,500 | 1,500 |
| Valid JSON records | 6,500 | 1,500 |
| Schema errors | 0 | 0 |
| Missing required fields | 0 | 0 |
| Null required fields | 0 | 0 |
| Blank text fields | 0 | 0 |
| Duplicate request IDs | 0 | 0 |
| Duplicate record contents, excluding request ID | 0 | 0 |
| Duplicate complete input states | 0 | 0 |

Each input state includes `subject`, `body`, `customer_tier`, `amount`, `draft_reply`, and `kb_article`. Every record has a valid five-class `route` label and Boolean `approve` and `grounded` labels. Amounts are finite numbers; the training range is 0–1,996.72 and the development range is 0–1,998.96.

There are **zero shared request IDs, zero shared complete states, and zero shared record contents across the two splits**. Comparisons canonicalize JSON key order and exclude `request_id` for content checks. Exact content duplicates and same-state conflicting labels were not found. These checks do not establish semantic independence: paraphrases or shared ticket/article templates may still occur across splits.

## Class distributions

| Label | Training count | Development count |
|---|---:|---:|
| Route: billing | 1,829 | 395 |
| Route: technical | 1,541 | 354 |
| Route: account | 1,324 | 298 |
| Route: shipping | 998 | 233 |
| Route: general | 808 | 220 |
| Approve: true | 2,453 | 619 |
| Approve: false | 4,047 | 881 |
| Grounded: true | 3,887 | 935 |
| Grounded: false | 2,613 | 565 |

Customer tiers in training are free: 2,945, pro: 2,468, enterprise: 1,087. Development contains free: 654, pro: 594, enterprise: 252. Class imbalance differs between tasks; raw accuracy alone is insufficient. Use the supplied scoring harness for macro-F1, asymmetric approval cost, calibration, coverage, and latency.

## Interpretation and limits

The original problem statement explicitly warns that some routing labels are inconsistent with a careful reading. It also says a minority of approval labels depend on account history absent from the inputs. These are organizer-provided limitations, not noise rates measured by this audit. Neither a zero-duplicate result nor schema validity proves the labels are correct.

Approval must use structured information such as customer tier and amount alongside ticket text. Grounding must compare the draft reply with the article; ticket routing and whether the reply is supported are separate questions. No model should assume perfect confidence merely because a training template is familiar.

Keep the supplied development split for final evaluation. Create any early-stopping, calibration, and threshold-selection partitions from the training file only. Record their random seed and proportions. The audit does not inspect, reconstruct, or access the hidden test set, and it cannot verify whether the public development distribution matches it.
