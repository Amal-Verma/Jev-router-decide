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

## 3. Abstention policy

```bash
python tune_policy.py <artifact dir>      # writes policy.json
```

This picks each decision's confidence threshold on the 1,000-record
calibration holdout (never dev), targeting 85% coverage.

## 4. Install and score

Copy `decide.onnx`, `tok/`, `calibration.json` and `policy.json` into
`artifacts/`, start the server and run `python selfcheck.py --url
http://localhost:8000/v1/decide`.

## Baseline

`python train_baseline.py` builds the gate-1 TF-IDF + LR/LightGBM model
(`artifacts/baseline.pkl`, not committed). `model.py` uses it only when
`artifacts/decide.onnx` is missing.
