# Reproducing the model

The served weights are committed in `artifacts/`. You only need this to retrain.

## 1. Data

Put the provided `train.jsonl` and `dev.jsonl` in `data/` (not committed).

## 2. Train on a GPU (we used a free Kaggle T4)

1. Create a **private** Kaggle dataset with these files at the top level:
   `train.jsonl`, `dev.jsonl`, `train_encoder.py`, `decide/features.py`,
   `decide/tabular.py`, `decide/inputs.py` (flat; the notebook rebuilds the
   `decide/` package).
2. New notebook, then File > Import Notebook > `kaggle/decide_train.ipynb`.
3. Add the dataset as input, set Accelerator to GPU T4, and turn Internet on
   (needed to fetch the pretrained encoder).
4. Save Version > Save & Run All. It trains two backbones (H768 NLI and H384
   ms-marco), each 5 epochs, about 9 minutes in total, and writes
   `art_h384.zip` and `art_h768.zip` to the output.

Or on any CUDA machine:

```bash
pip install -r requirements-train.txt
python train_encoder.py --data data --out artifacts --base cross-encoder/ms-marco-MiniLM-L6-v2 --epochs 5
```

Each run writes `decide.onnx` (int8), `tok/`, `calibration.json` (one
temperature per head), and `cal_logits.npz` / `dev_logits.npz`.

### The served ensemble

The served model has two members, each trained as above on `train.jsonl` with
seed 13: `--base cross-encoder/ms-marco-MiniLM-L6-v2` and
`--base jhu-clsp/ettin-encoder-17m`. The MiniLM member is served int8. The
Ettin member must be kept at fp32 (the fp32 export before `quantize_dynamic`),
because int8 changes up to 19% of its answers. The two graphs are merged into
one `decide.onnx` with input and output names prefixed `m_` and `e_`, the
tokenizers go to `tok_m/` and `tok_e/`, and then:

```bash
python fit_ensemble.py <ensemble dir> data/train.jsonl   # ensemble.json, calibration.json, policy.json
```

This fits the member and final temperatures and the abstention thresholds on
the 1,000-row train holdout (`cal_ids.json`), never dev. Note: the committed
`train_encoder.py` deletes its fp32 export after quantising, and the merge step
is not a committed script.

## 3. Abstention policy

```bash
python tune_policy.py <artifact dir>      # writes policy.json
```

This picks each decision's confidence threshold on the 1,000-record
calibration holdout (never dev), targeting 85% coverage.

## 4. Install and score

Copy `decide.onnx`, the tokenizer folders, `ensemble.json` (ensemble only),
`calibration.json` and `policy.json` into `artifacts/`, start the server and run `python selfcheck.py --url
http://localhost:8000/v1/decide`.

## Baseline

`python train_baseline.py` builds the gate-1 TF-IDF + LR/LightGBM model
(`baseline/baseline.pkl`). It is kept for comparison only; the service does not
load it.
