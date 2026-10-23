# Decide: a CPU-only typed decision engine

One shared model answers three typed questions about a support ticket
(`route`, `approve`, `grounded`) behind `POST /v1/decide`, with calibrated
probabilities and a deliberate abstention policy. No network calls at serving
time; inference is a single int8 ONNX graph on CPU.

## Run it (clean 4-core Linux box)

```bash
pip install -r requirements.txt && uvicorn app:app --host 0.0.0.0 --port 8000 --workers 1
```

The trained weights are committed in `artifacts/` (int8 ONNX, 24 MB), so
there is nothing to download. Then, with `data/dev.jsonl` in place:

```bash
python selfcheck.py --url http://localhost:8000/v1/decide
```

## Architecture: where the sharing happens

```
             route: (subject, body)
  request -> approve: ("<tier> plan, $<amount>. subject", body) + tier/amount/flag features
             grounded: (kb_article, draft_reply)          [NLI premise / hypothesis]
                 |
                 v
     shared Transformer encoder (MiniLM-L6-H384)         <- shared by all decisions
                 |  mean-pooled
                 v
     shared Mixture of Experts: 4 expert MLPs,           <- shared by all decisions
     soft gate conditioned on (pooled text, task embedding)
                 |
        +--------+---------+
        v        v         v
     route    approve   grounded                         <- one Linear head each
     (5-way)  (binary)  (binary)
                 |
     per-head temperature scaling -> policy (threshold, abstention)
```

- **Encoder and experts are shared** (`DecideNet` in `train_encoder.py`). Every
  request, whatever its decision type, runs through the same weights in the same
  ONNX graph (`artifacts/decide.onnx`). Only the last Linear layer differs.
- **The questions are not the same shape**, so each one is framed for what it
  compares (`decide/inputs.py`): routing reads the ticket, approval reads the
  ticket plus structured features (the policy is mostly a per-tier amount
  threshold with red-flag overrides), and grounding is a cross-encoded
  premise/hypothesis pair, which is why we start from an encoder pre-trained
  as a cross-encoder over text pairs.
- **Adding a fourth decision** is one new task-embedding row, one Linear head
  and its training examples. The encoder and experts are reused, and inference cost
  does not change.

## Calibration and abstention

- Training holds out 1,000 train records. One temperature per head is fitted on
  them by NLL (Guo et al. 2017). The dev split is never used for fitting.
- `approve` commits to "yes" only when `p_true > 0.75`: a wrong approval costs
  3x a wrong hold, so the Bayes threshold is 3/(3+1).
- Each decision abstains on its least-confident ~18% (thresholds in
  `artifacts/policy.json`, chosen on the holdout by `tune_policy.py`), so the
  declared coverage sits just above the 80% target.

## Training

GPU training ran on a Kaggle T4 (`kaggle/decide_train.ipynb` runs
`train_encoder.py`): multi-task fine-tuning, one task per batch, 5 epochs,
AdamW + one-cycle schedule, fp16. It exports to ONNX and applies dynamic int8
quantisation. Then `python tune_policy.py artifacts` writes the abstention
policy.

`train_baseline.py` is the gate-1 classical baseline (TF-IDF + logistic
regression / LightGBM). `model.py` falls back to it if `artifacts/decide.onnx`
is absent.

## Pretrained components and licences

| Component | Use | Licence |
|---|---|---|
| `cross-encoder/ms-marco-MiniLM-L6-v2` (Hugging Face) | encoder initialisation (served model) | Apache-2.0 |
| `cross-encoder/nli-MiniLM2-L6-H768` (Hugging Face) | compared backbone, not served | Apache-2.0 |

Libraries: PyTorch (BSD-3), Transformers (Apache-2.0), ONNX Runtime (MIT),
tokenizers (Apache-2.0), FastAPI (MIT), Uvicorn (BSD-3), scikit-learn (BSD-3),
LightGBM (MIT).

## Results (public dev split, full self-check)

| | score |
|---|---|
| classical baseline (`train_baseline.py`) | 78.45 |
| shared encoder, H768 NLI backbone | 73.65 (more accurate grounding, too slow) |
| **shared encoder + MoE, H384 (served)** | **83.76** |

Mean p95 is about 24 ms per call. Full numbers are in `results.json`; accuracy,
calibration and failure modes are in `model_card.md`.

## One trade-off we made

We served the smaller H384 backbone over the H768 NLI one. H768 grounds
perfectly on dev (F1 1.00 vs 0.99) but costs about 1.7x the latency, and
latency is 20% of the score with full marks only at 25 ms. A 0.01 F1 gain on
one decision does not pay for that.

## Docs

- `docs/API.md`: request and response reference
- `docs/TRAINING.md`: how to retrain (Kaggle GPU) and rebuild the policy
- `docs/DEMO.md`: three-minute demo script, including a refusal case
- `docs/PLAN.md`: plan, status and untried ideas

## Layout

```
app.py              HTTP layer (starter, unchanged)
model.py            DecisionModel: policy (thresholds, abstention) over a backend
decide/inputs.py    per-decision input framing for the shared encoder
decide/tabular.py   approve features (tier x amount cuts, red flags)
decide/features.py  duration normalisation ("two weeks" == "14 days")
decide/runtime.py   ONNX Runtime CPU backend
train_encoder.py    multi-task training + ONNX export (GPU)
tune_policy.py      abstention thresholds from the calibration holdout
train_baseline.py   gate-1 classical baseline (needs requirements-train.txt)
kaggle/decide_train.ipynb  the Kaggle notebook used for training
```
