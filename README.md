# Decide: a CPU-only typed decision engine

One shared model answers three typed questions about a support ticket
(`route`, `approve`, `grounded`) behind `POST /v1/decide`, with calibrated
probabilities and a deliberate abstention policy. No network calls at serving
time; inference is one ONNX graph on CPU holding a two-member ensemble
(MiniLM int8 + Ettin-17m fp32), computed in a single call.

## Run it (clean 4-core Linux box)

```bash
pip install -r requirements.txt && uvicorn app:app --host 0.0.0.0 --port 8000 --workers 1
```

The trained weights are committed in `artifacts/` (`decide.onnx` is 92 MB,
and `artifacts/` holds only the served ensemble, about 93 MB), so there is nothing to download. Then, with `data/dev.jsonl` in place:

```bash
python selfcheck.py --url http://localhost:8000/v1/decide
```

## Running each model (optional CLI)

The judges' command above serves the **ensemble** (the submission). The two
single models are also committed under `artifacts/`, and either can be run:

| Model | Weights | Dev score (full selfcheck) |
|---|---|---|
| **ensemble** (default, submitted) | `artifacts/` | 85.44 |
| MiniLM + MoE | `artifacts/minilm/` | 83.75 |
| Ettin-17m + MoE | `artifacts/ettin/` | 84.68 |

```bash
python serve.py --model ensemble     # same as: uvicorn app:app
python serve.py --model minilm --port 8001
python serve.py --model ettin  --port 8002
# or, without serve.py:  ARTIFACTS_DIR=artifacts/minilm uvicorn app:app --port 8001

# answer one ticket in the terminal (no server); prints one JSON line per decision
python predict.py --model ettin --decision approve --ticket ticket.json
echo '{"subject":"Refund please","body":"Please refund $12.","customer_tier":"pro","amount":12}' | python predict.py --model minilm
```

All three use the same `POST /v1/decide` schema.

### Scoring a model on the dev split

`selfcheck.py` (the judges' scoring code) needs `data/dev.jsonl`, which is not
committed: put the provided `train.jsonl` and `dev.jsonl` in `data/`. Then, one
model at a time:

```bash
# terminal 1: start the model (ensemble, minilm or ettin) and leave it running
python serve.py --model ensemble --port 8000

# terminal 2: score it on the full dev split (1,500 tickets x 3 decisions, about 2 minutes)
python selfcheck.py --url http://localhost:8000/v1/decide --json /tmp/dev_ensemble.json

# stop the server with Ctrl+C, then repeat with --model minilm and --model ettin
```

Write to a scratch path as above, so the submitted `results.json` (the ensemble)
is not overwritten. Latency depends on how busy the machine is.

## Architecture: where the sharing happens

```mermaid
flowchart TD
    R["POST /v1/decide<br/>decision + state"] --> F{"frame input<br/>per decision"}
    F -->|route| I1["(subject, body)"]
    F -->|approve| I2["(tier plan, $amount. subject, body)<br/>+ tier/amount/flag features"]
    F -->|grounded| I3["(kb_article, draft_reply)"]
    I1 & I2 & I3 --> ONE["ONE ONNX graph, ONE session.run"]
    subgraph M1["Member 1 (int8)"]
        E1["MiniLM-L6-H384 encoder"] --> X1["gate + 4 shared experts"]
        X1 --> H1["route / approve / grounded heads"]
    end
    subgraph M2["Member 2 (fp32)"]
        E2["Ettin-17m encoder"] --> X2["gate + 4 shared experts"]
        X2 --> H2["route / approve / grounded heads"]
    end
    ONE --> E1
    ONE --> E2
    H1 --> T1["member temperature"]
    H2 --> T2["member temperature"]
    T1 & T2 --> AVG["mean of the two logits<br/>then a final temperature"]
    AVG --> P["Policy: approve if p > 0.75,<br/>abstain below the confidence threshold"]
    P --> O["typed JSON response"]

    classDef shared fill:#dbeafe,stroke:#1d4ed8,color:#0b1b3f
    class E1,X1,E2,X2 shared
```

Blue = shared by every decision type, inside each member. More diagrams
(request lifecycle, training pipeline, approve and abstention logic) are in
[`docs/ARCHITECTURE.md`](docs/ARCHITECTURE.md).

- **The served model is an ensemble of two complete shared models**, merged
  into one ONNX graph (`artifacts/decide.onnx`). Each member is the full
  `DecideNet` from `train_encoder.py`: one encoder and one mixture of 4 experts
  shared by all three decisions, plus one small Linear head per decision.
  Neither member is specialised for one decision; each answers all three.
- **One call computes both members.** `EnsembleBackend` in `decide/runtime.py`
  runs one `session.run` per request. The decision's logit is the mean of the
  two members' temperature-scaled logits, followed by a final temperature
  (`artifacts/ensemble.json`, `artifacts/calibration.json`, fitted by
  `fit_ensemble.py`).
- **Member 1:** MiniLM-L6-H384 (from `cross-encoder/ms-marco-MiniLM-L6-v2`),
  int8. **Member 2:** Ettin-17m (from `jhu-clsp/ettin-encoder-17m`), kept at
  full precision (fp32), because int8 dynamic quantisation broke it: int8 and
  fp32 Ettin gave the same answer on only 91% of route, 92% of approve and 81%
  of grounded cases, while MiniLM int8 agrees with its fp32 version on 99.7 to
  100%.
- **The questions are not the same shape**, so each one is framed for what it
  compares (`decide/inputs.py`): routing reads the ticket, approval reads the
  ticket plus structured features (the policy is mostly a per-tier amount
  threshold with red-flag overrides), and grounding is a cross-encoded
  premise/hypothesis pair, which is why we start from encoders that read text
  pairs.
- **Parameter count** (counted from `artifacts/decide.onnx`): about 40.2 M in
  total, 23.07 M in the MiniLM member and 17.17 M in the Ettin member. Only
  1,799 weights per member (3,598 in all) are decision-specific.

| Component | MiniLM member | Ettin-17m member | Shared by all decisions? |
|---|---|---|---|
| encoder | 22,565,376 (6 layers, 384 wide) | 16,797,440 (7 layers, 256 wide) | yes |
| 4 expert MLPs (mixture of experts) | 492,544 | 361,472 | yes |
| tabular-feature projection | 5,056 | 5,056 | yes |
| gate | 1,668 | 1,156 | yes |
| task embedding (3 x 32) | 96 | 96 | yes (one row per decision) |
| route head | 1,285 | 1,285 | route only |
| approve head | 257 | 257 | approve only |
| grounded head | 257 | 257 | grounded only |
| **member total** | **about 23.07 M** | **17,167,019** | **99.99% shared** |

- **Adding a fourth decision** is one new task-embedding row and one Linear head
  per member, plus its training examples. The encoders and experts are reused,
  and inference cost does not change.

## Calibration and abstention

- Training holds out 1,000 train records (the same rows for both members).
  On them, `fit_ensemble.py` fits by NLL (Guo et al. 2017) one temperature per
  member and decision, then one final temperature per decision on the averaged
  logits.
- `approve` commits to "yes" only when `p_true > 0.75`: a wrong approval costs
  3x a wrong hold, so the Bayes threshold is 3/(3+1).
- Route and grounded abstain on their least-confident ~15% of the holdout
  (route: top probability under 0.892; grounded: max(p, 1-p) under 0.9957);
  approve abstains in the lopsided band 0.15 < p < 0.79 (wrong approvals cost
  3x). Thresholds are in `artifacts/policy.json` (written by `fit_ensemble.py`),
  so declared coverage on dev sits at 0.83 to 0.84, above the 80% target.

## Training

GPU training ran on a Kaggle T4 (`kaggle/decide_train.ipynb` runs
`train_encoder.py`): multi-task fine-tuning, one task per batch, 5 epochs,
AdamW + one-cycle schedule, fp16, seed 13. Each member is trained this way on
its own backbone and exported to ONNX; the MiniLM member gets dynamic int8
quantisation, the Ettin member stays fp32. The two graphs are merged into one,
and `python fit_ensemble.py <dir> data/train.jsonl` fits the temperatures and
writes the abstention policy.

The Ettin member is a deterministic rerun of an earlier Ettin training run
(same code, seed 13); three reruns reproduced it exactly.

Both members are trained on `train.jsonl` only, and calibrated on the same
1,000-row train holdout. Dev is never used for training or fitting. 4-fold
cross-validation (`kaggle/decide_cv.ipynb`, `cv_analyze.py`) was used to check
consistency and choose epochs and the approve abstention band; its fold models
are not served.

`train_baseline.py` is the gate-1 classical baseline (TF-IDF + logistic
regression / LightGBM), saved to `baseline/baseline.pkl`. It is kept for
comparison only; the service does not load it.

## Pretrained components and licences

| Component | Use | Licence |
|---|---|---|
| `cross-encoder/ms-marco-MiniLM-L6-v2` (Hugging Face) | encoder of served member 1 | Apache-2.0 |
| `jhu-clsp/ettin-encoder-17m` (Hugging Face) | encoder of served member 2 | MIT |
| `cross-encoder/nli-MiniLM2-L6-H768` (Hugging Face) | compared backbone, not served | Apache-2.0 |

Libraries: PyTorch (BSD-3), Transformers (Apache-2.0), ONNX Runtime (MIT),
tokenizers (Apache-2.0), FastAPI (MIT), Uvicorn (BSD-3), scikit-learn (BSD-3),
LightGBM (MIT).

## Results (public dev split, full self-check)

| | score |
|---|---|
| classical baseline (`train_baseline.py`) | 78.45 |
| shared encoder, H768 NLI backbone, machine busy | 73.65 (more accurate grounding, too slow) |
| MiniLM + Ettin-17m ensemble, Ettin in int8 | 82.24 (int8 breaks Ettin) |
| MiniLM + Ettin-17m ensemble, a different Ettin training run | 82.78 |
| MiniLM H384 + MoE alone (previously served) | 83.75 (mean p95 10.7 ms) |
| Ettin-17m + MoE alone, fp32 | 84.68 (mean p95 12.7 ms, but approve coverage 0.790, under 80%) |
| **MiniLM + Ettin-17m ensemble, Ettin fp32 (served, trained on train only)** | **85.44** |

Served model, in detail (`results.json`): quality 0.858, calibration 0.869,
latency 1.000, abstention 0.627. Route macro-F1 0.9471, approve cost 0.1786,
grounded macro-F1 0.9983, mean ECE 0.0141, Brier skill 0.8086. Coverage: route
0.838, approve 0.829, grounded 0.837. 4,500 of 4,500 responses valid.

**How much of 85.44 to expect on the hidden test.** 4-fold cross-validation of
the same ensemble idea (fresh models per fold, fp32) gave **83.13 ± 0.53**,
against 82.57 ± 1.29 for MiniLM alone; Ettin alone averaged 81.85. So the
ensemble is better and steadier, but the dev 85.44 is partly a favourable
Ettin run. An honest hidden-test expectation is roughly 83 to 85.

**Latency.** p95 in `results.json` is route 18.1, approve 19.8, grounded 23.3 ms
(mean 20.4 ms), measured on a busy laptop (load average 3.5 to 4.5). A
fresh-venv run of `scripts/verify_submission.sh` gave PASS, 85.44, mean p95
23.0 ms under similar load, and a fresh-clone run (`--clone`) gave PASS, 85.44,
4,500 of 4,500 valid, mean p95 23.8 ms. That is under the 25 ms full-marks line, but with
less headroom than the single MiniLM model had. Full numbers are in
`results.json`; accuracy, calibration and failure modes are in `model_card.md`.

## Trade-offs we made

- **Ensemble over a single model.** It costs about twice the latency of
  MiniLM alone, but it scored +1.69 on dev and +0.56 in cross-validation, with
  a smaller spread across folds. Latency still gets full marks on a busy laptop.
- **Ettin kept at fp32.** int8 would make it smaller and faster, but it changed
  up to 19% of Ettin's grounded answers, and the int8 ensemble scored 82.24
  against 85.44.
- **H384 MiniLM over the H768 NLI backbone.** H768 grounds perfectly on dev
  (F1 1.00 vs 0.99) but costs about 1.7x the latency, and it scored 73.65 when
  the machine was busy.

## Docs

- `docs/EXPLAINED.md` (also as `docs/report/explained.pdf`): **start here**, a plain-language guide from the problem to every number
- `docs/EXPERIMENTS.md` (also `docs/report/experiments.pdf`): every approach tried and its measured score
- `docs/presentation/decide.pptx`: 8-slide plain deck for the 3-minute demo, with speaker notes
- `docs/report/report.pdf`: 4-page technical report (LaTeX source alongside)
- `docs/ARCHITECTURE.md`: diagrams of the model, request path, training and policy
- `docs/API.md`: request and response reference
- `docs/TRAINING.md`: how to retrain (Kaggle GPU) and rebuild the policy
- `docs/DEMO.md`: three-minute demo script, including a refusal case
- `docs/PLAN.md`: plan, status and untried ideas
- `docs/SUBMISSION.md`: submission checklist; `scripts/verify_submission.sh` re-checks it from a fresh venv

## Layout

```
app.py              HTTP layer (starter, unchanged)
model.py            DecisionModel: policy (thresholds, abstention) over a backend
serve.py            start the API with --model ensemble | minilm | ettin
predict.py          answer a ticket in the terminal with any of the three models
artifacts/          served ensemble (root), plus artifacts/minilm/ and artifacts/ettin/
merge_ensemble.py   merge two member models into one ONNX graph
decide/inputs.py    per-decision input framing for the shared encoder
decide/tabular.py   approve features (tier x amount cuts, red flags)
decide/features.py  duration normalisation ("two weeks" == "14 days")
decide/runtime.py   ONNX Runtime CPU backends (EnsembleBackend is the served one)
train_encoder.py    multi-task training + ONNX export (GPU)
fit_ensemble.py     ensemble temperatures + abstention policy (holdout only)
tune_policy.py      abstention thresholds for a single-model run
train_baseline.py   gate-1 classical baseline (needs requirements-train.txt)
kaggle/decide_train.ipynb  the Kaggle notebook used for training
```
