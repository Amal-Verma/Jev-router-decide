# Applicability to real support workflows

The current system is a strong benchmark prototype and a candidate for a supervised pilot. Its real-world accuracy and calibration have not been established. Synthetic, templated training/dev tickets and a high local dev score do not establish that it can autonomously decide real customer outcomes.

## Current use

| Decision | Plausible initial use | Remaining business limitation |
|---|---|---|
| Route | Suggest a queue to a support agent, with correction and fallback | Real taxonomies, multiple issues, conversation history and unfamiliar language are not validated |
| Approve | Suggest review priority and possible eligibility | The supplied fields cannot establish account history, payment/order status, previous refunds or current company rules |
| Grounded | Flag a draft for agent review | Article support is different from proof of a completed action; claim-level correctness is not established |

The 85.36 local rubric score is not an 85.36% real-world success rate. Grounding F1 0.9983 applies to answered cases on this public dev set. It must not be presented as 99.83% reliability on arbitrary customer replies. Confidence calibrated on one dataset can degrade after a distribution shift; see [Ovadia et al., NeurIPS 2019](https://proceedings.neurips.cc/paper/2019/hash/8558cb408c1d76621371888657d2eb1d-Abstract.html).

## Illustrative challenge findings

Four hand-written cases were evaluated after consolidation and were not used for fitting or policy tuning. Their inputs, expected labels and rationale are in `evaluation/real_world_challenges.json`. This small set exposes examples, not a representative failure rate.

- A refund policy stated eligibility and settlement time. The draft said **“I have processed your refund.”** The model returned supported at **0.998804**, without abstaining, although no transaction evidence existed. This is an unsafe answer for autonomous use.
- A draft incorrectly restricted an all-plan policy to enterprise customers. The model's tentative answer was supported at 0.630405, but it abstained. Human review prevented commitment in this example.
- A changed deadline (14 vs 30 days) was tentatively rejected, with abstention.
- A supported “14 days” / “two weeks” paraphrase was tentatively accepted but also abstained, illustrating review workload outside the training templates.

## Priority additions

1. **Representative real evaluation.** Obtain appropriately de-identified, expert-labelled tickets from a chosen business. Include multi-issue conversations, policy exceptions, unsupported actions, contradictory documents, unfamiliar phrasing and incomplete inputs. Hold out future time periods and other customers/organizations; do not just randomly split near-identical templates. Measure false approvals, false supported claims, routing corrections and human-review load by task and subgroup. Recalibrate confidence on this distribution.
2. **Authoritative business context.** Supply current/versioned company policies, order and payment state, account history, previous claims, dates and currencies. Retrieve the relevant policy rather than accepting an arbitrary article as complete authority. Ask for missing information instead of inferring it.
3. **Explicit approval eligibility.** Check amount limits, time windows, payment/order status, repeat claims and company rules through an explicit policy decision. Use ML to interpret intent and flag ambiguity. A learned label and the hackathon's fixed 3:1 cost ratio are insufficient authorization for an actual refund.
4. **Claim-level grounding and action verification.** Identify claims and supporting passages; check names, amounts, dates, quantities, conditions, negation and scope. Claims that something happened need tool/transaction evidence. A useful result distinguishes supported, contradicted and insufficient evidence, with the relevant source or missing fact. [FActScore](https://aclanthology.org/2023.emnlp-main.741/) is related work on evaluating individual factual claims; it is not proof this model implements that approach.
5. **Complete human handoff.** Return a useful reason, evidence, suggested next action or clarification question. Preserve the request context when escalating. Evaluate whether abstention catches costly mistakes, including confident failures; confidence alone does not establish safety.
6. **A narrow rollout with feedback.** Start in shadow mode, then agent-assist mode. Allow automation only for a limited, explicit set of low-risk cases after meeting business-defined error tolerances. Review overrides and distribution changes over time. More languages and domains should be added as separately evaluated scope extensions.

There is no meaningful “perfect” version for arbitrary support requests. A dependable product has a clear supported domain, verifies the business facts it needs, and refers unsupported situations to a person. The next highest-value work is domain evidence and real evaluation, not simply a larger encoder or a higher synthetic benchmark score.
