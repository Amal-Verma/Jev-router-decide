# Local service

FastAPI loads one `DecisionModel` instance at startup and uses it for all three decisions. Startup fails with the model's error when artifacts cannot be loaded; serving does not silently fall back to placeholder predictions.

The default artifacts, static interface, and evaluation report paths resolve relative to `app.py`, so changing the shell's working directory does not change which files are loaded. `ARTIFACTS_DIR` overrides the model artifact directory.

## Endpoints

| Method | Path | Behavior |
| --- | --- | --- |
| GET | `/` | Serves the local interactive interface. |
| GET | `/static/...` | Serves its bundled styles and scripts. |
| GET | `/health` | Reports model availability and public model metadata. |
| POST | `/v1/decide` | Preserves the supplied judging contract for `route`, `approve`, and `grounded`. |
| GET | `/v1/metrics` | Returns the selected model’s actual report, or 404 when it does not exist. |

Text fields containing explicit JSON null become empty strings, and a null amount becomes zero. Unknown state fields are preserved. Non-finite amounts are rejected with a JSON 422 response. Validation errors omit the original input values so even malformed non-finite JSON inputs receive a serializable error response.

Routing probabilities are checked for finite, nonnegative values and a positive total. Distributions outside the supplied normalization tolerance are normalized; confidence always matches the returned queue's probability. Binary probabilities must be finite and between zero and one. Invalid model output produces an error instead of a misleading probability.

The health response exposes a fixed allowlist of metadata fields and omits arbitrary private fields.

## Contract checks

From the repository root after installing `requirements-dev.txt`, run:

```bash
.venv/bin/python -m unittest test_service -v
```

The suite isolates the HTTP contract with a deterministic model double. It checks all three decisions against the unchanged judging validator, abstention preservation, null normalization, state extensions, invalid decisions, non-finite amounts, safe health metadata, evaluation report availability, UI serving, unavailable models, and invalid model outputs. The supplied self-check separately measures the trained model through the real HTTP endpoint.

The ensemble report is `results.json`; single-model reports are `evaluation/minilm.json` and `evaluation/ettin.json`. This prevents displaying the ensemble benchmark while serving a single encoder. `artifact_bytes` in public metadata is the selected ONNX graph size.
