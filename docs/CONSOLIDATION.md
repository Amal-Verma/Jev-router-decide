# Decide implementation comparison and consolidation

## Decision

Use `Amal-Verma/Jev-router-decide` as the canonical project and submit the integration through a fork pull request. Retain its complete trained ONNX ensemble and single-model alternatives, training/export/calibration code, CLI, experiment history, presentation and reports. Add the offline workbench, defensive API checks, missing-evidence policy, reproducible dataset audit and regression tests from the smaller implementation.

The weaker classifier is not inserted into the prediction path: that would add a new model-selection/calibration problem without evidence of a quality gain. The original lightweight implementation remains recoverable from its local checkout and Git bundle backup. The historical comparison report is retained in `evaluation/lightweight-2026-10-26.json`.

## Source comparison

| Area | Existing Jev implementation | Lightweight implementation | Combined choice |
|---|---|---|---|
| Predictor | Shared MiniLM/Ettin models in one CPU ONNX graph | Shared sparse encoder and 96-unit MLP | Retain ONNX model and its task-specific input framing |
| Recorded dev score | 85.44 | 75.78 | Retain stronger quality/calibration model |
| Grounding macro-F1 on answered cases | 0.9983 | 0.9086 | Retain cross-encoded article/reply model |
| Mean calibration error | 0.0141 | 0.0368 | Retain existing calibration |
| Historical mean HTTP p95 | 20.40 ms | 8.89 ms | Retain fast CPU serving; measure again locally |
| Artifacts | About 193 MB including all three options | About 3.80 MB including baseline | Retain existing artifacts, below 500 MB |
| Training/experiments | GPU fine-tuning, exports, calibration, CV, reports | Reproducible CPU baseline/training and data audit | Retain existing training; add audit and preserve comparison evidence |
| Interface | API and CLI | Offline interactive workbench | Add workbench, fonts and response inspection |
| Service checks | Supplied scorer and clean-install verifier | API/model regressions and invalid-input handling | Retain scorer/verifier, add targeted regressions and validation |
| Missing input | Encoder predictions even when evidence absent | Explicit abstention with uninformative probabilities | Add uniform routing / 0.5 binary abstention |

Historical scores above come from the saved source reports, not a controlled same-host speed comparison. The existing ensemble reference is `evaluation/ensemble-recorded.json`. PDFs and experiment/deck documents retain their original experimental measurements; current integration evidence is below.

## Current validation, 2026-11-16

The pre-integration checkout was `1559b331a2cffb5b94adf6484daa1b093d38dbfb`. Workbench source was `97ace9e4a8ba8b18762613276feaab9f8b0f70a3`. Both self-checks used the same public dev file, 1,500 tickets and 4,500 sequential HTTP requests after 40 discarded warm-up calls. Neither run fits weights, calibration or policy on dev.

| Measure | Existing service, current environment | Combined service |
|---|---:|---:|
| Valid schema responses | 4,500 / 4,500 | 4,500 / 4,500 |
| Reachability / schema / latency gates | All pass | All pass |
| Local rubric score | 85.1279 | 85.3567 |
| Mean per-decision HTTP p95 | 26.38 ms | 25.52 ms |
| Route macro-F1, answered cases | 0.947124 | 0.947124 |
| Approval cost, answered cases | 0.176329 | 0.176329 |
| Grounded macro-F1, answered cases | 0.998265 | 0.998265 |
| Mean ECE, all cases | 0.014311 | 0.014311 |

Every non-latency result section (`gates`, `schema`, `quality`, `calibration`, `abstention`) is exactly identical before and after integration. The score difference is entirely timing-related, not an accuracy improvement. Latest p95 values are route 22.81 ms, approve 24.71 ms, grounded 29.04 ms. Mean p95 is slightly above the 25 ms full-marks line, comfortably below the 100 ms gate. These are local-host measurements, not the judges' four-core hardware or hidden test.

All 18 artifact-file SHA-256 hashes match the upstream revision; no model, tokenizer, temperature or learned threshold was changed. `evaluation/integration_verification.json` records hashes, versions and report comparison; `evaluation/ensemble-before-integration.json` preserves the fresh before-run and root `results.json` the combined run. The newer runtime environment produces minor numerical differences from the original historical report, so its exact historical score is not claimed as freshly reproduced.

Other verification:

- 21 model/policy/API regression tests pass, including CPU-only provider, no network during inference, complete-input parity with the original policy, threshold boundaries, missing evidence, negative amounts, numerical overflow protection and schema validation.
- All three bundled model choices load and answer through the existing CLI.
- The full dataset audit reports 6,500 train and 1,500 dev rows, no schema/missing-field errors, and no exact duplicate IDs, complete states or record contents across splits. This does not prove semantic/template independence or correct labels.
- Browser checks against the real ensemble cover Refund submission, Ctrl+Enter, live probabilities/timing, missing-article abstention at `p_true=0.5`, selected-model identification and raw JSON disclosure. Desktop document width fits the viewport and no external scripts are loaded. Responsive styles and reduced-motion support are retained; physical mobile-device testing is not claimed.
- Shell and JavaScript syntax checks and `pip check` pass. The environment was created fresh from serving requirements plus test dependencies.

## Runtime additions

The fixed `POST /v1/decide` request/response contract and the original `selfcheck.py` remain intact. `/` serves the optional workbench; `/health` adds public loaded-model metadata; `/v1/metrics` selects the actual report for the loaded model and returns 404 if none is available. It never substitutes the ensemble score while a different model is active.

Finite input and output checks prevent malformed probabilities or non-serializable errors. Model artifacts resolve relative to the source rather than the shell directory. Empty ticket text defers routing and approval; absent reply/article defers grounding; negative approval amounts defer before tabular logarithms. Complete supported requests keep the same learned behavior. Binary probability conversion uses a stable sigmoid for extreme logits.

Use `./run.sh` for local setup, or retain the existing judging command. `MODEL=minilm` or `MODEL=ettin` selects a single model. Dependencies for tests are separate in `requirements-dev.txt`. New implementation rationale is documented here and in the service/UI documents instead of added source comments. The interface contains no emojis.

![Consolidated workbench](screenshots/workbench.jpg)
