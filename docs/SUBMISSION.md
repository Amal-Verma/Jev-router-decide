# Submission checklist

Everything the brief asks for, and who has to do it. "Done" items are in the
repo now; "Amal" items can only be done by the account owner.

## Deliverables in the repo

| # | Item | Where | Status |
|---|---|---|---|
| 1 | Single start command | `README.md`, "Run it": `pip install -r requirements.txt && uvicorn app:app --host 0.0.0.0 --port 8000 --workers 1` | done |
| 2 | Trained weights committed, nothing to download | `artifacts/` (one merged ONNX graph `decide.onnx` 92 MB: MiniLM member int8 + Ettin-17m member fp32; tokenizers `tok_m/`, `tok_e/`; `ensemble.json`, calibration, policy; `artifacts/` holds only the served ensemble, about 93 MB) | done |
| 3 | Self-check results on dev | `results.json` (full run, 1,500 records x 3 decisions): 85.44 | done |
| 4 | Model card | `model_card.md` | done |
| 5 | Licences of pretrained components and libraries | `README.md`, "Pretrained components and licences" | done |
| 6 | One shared model for all three decisions | served model is a MiniLM + Ettin-17m ensemble; each member is a full shared model answering all three decisions, both in one ONNX graph, one `session.run` per request (`README.md`, "Architecture") | done |
| 7 | No training on, or memorising of, the dev split | both members trained on `train.jsonl` only and calibrated on a train holdout (`model_card.md`, "Training data") | done |
| 8 | Demo script (3 minutes, includes a refusal case) | `docs/DEMO.md`, deck `docs/presentation/decide.pptx` | done |
| 9 | Clean-environment proof | `scripts/verify_submission.sh` (fresh venv, README command, full self-check) | done |
| 10 | Release tag | `v1.0-final` (annotated, local) | done locally |

## Pass/fail floors (checked by `scripts/verify_submission.sh`)

- Valid schema on every response: 4,500 / 4,500 valid in `results.json`.
- p95 under 100 ms: see `latency_ms.p95` in `results.json` (mean p95 20.4 ms on a busy laptop; a fresh-venv run gave PASS, 85.44, mean p95 23.0 ms under similar load; a fresh-clone run with `--clone` gave PASS, 85.44, 4,500 / 4,500 valid, mean p95 23.8 ms). This is under the 25 ms full-marks line, but with less headroom than the earlier single model, so keep the judging host idle.
- One shared model, model card present, dev never used for training: see items 6, 4 and 7.

To re-check from scratch: `scripts/verify_submission.sh` (working tree) or
`scripts/verify_submission.sh --clone` (fresh clone of HEAD, proves the committed
files are enough). Both need `data/dev.jsonl`, which is not committed.

## Only Amal can do these

| # | Action | Command or note |
|---|---|---|
| A | Push the code and the tag to the public GitHub repo | `git push origin master --tags` (the tag was moved locally, so if `v1.0-final` was pushed before: `git push -f origin v1.0-final`) |
| B | Make sure the repo is public | GitHub settings of `Amal-Verma/Jev-router-decide` |
| C | Start the service on the judging host and keep it live at the deadline | `uvicorn app:app --host 0.0.0.0 --port 8000 --workers 1`; keep the machine otherwise idle, latency is 20% of the score |
| D | Submit host and port to the judges | e.g. `http://<host>:8000/v1/decide`; check from another machine with `curl http://<host>:8000/health` |
| E | Give the 3-minute demo | follow `docs/DEMO.md`; slides in `docs/presentation/decide.pptx` |
| F | Revoke the Kaggle API token `decide-cli` | Kaggle, Settings, API tokens |
| G | Delete the temporary token file | `rm ~/.kaggle_token_tmp` |
