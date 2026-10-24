# Plan and status

Intent: a CPU-only service answering three typed decisions (route / approve /
grounded) from one shared model, with calibrated probabilities, a deliberate
abstention policy and p95 latency of 25 ms or less.

| Phase | Done when | Status |
|---|---|---|
| 1. Baseline (gate 1) | classical model serving, schema-valid, scored | done: 78.45 |
| 2. Live endpoint (gate 2) | `uvicorn app:app` serves the schema | done |
| 3. Shared encoder | multi-task MiniLM + MoE trained on GPU, exported int8 ONNX | done: Kaggle T4 |
| 4. Calibration + abstention | temperatures and thresholds from the holdout | done: mean ECE 0.016, coverage 83-85% |
| 5. Deliverables | README, model card, results.json, docs | done: 83.75 on dev (train-only model), 82.56 ± 1.28 cross-validated |
| 6. Ensemble | MiniLM (int8) + Ettin-17m (fp32) in one ONNX graph, calibrated on the train holdout | done: served, 85.44 on dev, 83.13 ± 0.53 cross-validated |

Open:
- Run the service on the judging host and give the judges its host and port.
- Ideas not tried: train on train+holdout with cross-fitted calibration; tune
  the approve threshold on the holdout instead of the fixed 0.75; distil H768
  grounding into H384.
