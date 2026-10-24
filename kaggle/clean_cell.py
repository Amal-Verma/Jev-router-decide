import glob, json, os, random, shutil, subprocess, sys
import numpy as np
src = os.path.dirname(glob.glob("/kaggle/input/**/train_encoder.py", recursive=True)[0])
W = "/kaggle/working/src"
shutil.copytree(src, W, dirs_exist_ok=True)
os.makedirs(W + "/decide", exist_ok=True); os.makedirs(W + "/data", exist_ok=True)
open(W + "/decide/__init__.py", "w").close()
for m in ("features.py", "tabular.py", "inputs.py"):
    shutil.move(os.path.join(W, m), os.path.join(W, "decide", m))
for f in ("train.jsonl", "dev.jsonl"):
    shutil.move(os.path.join(W, f), os.path.join(W, "data", f))
E = "/kaggle/working/exp"
B = "cross-encoder/ms-marco-MiniLM-L6-v2"
QUEUES = ["billing", "technical", "account", "shipping", "general"]
def run(*a):
    subprocess.run([sys.executable, "train_encoder.py", "--epochs", "5", "--no_export", "--base", B, *a], cwd=W, check=True)
# 1) out-of-fold predictions inside train.jsonl only (dev never involved)
rows = [l for l in open(W + "/data/train.jsonl") if l.strip()]
random.Random(99).shuffle(rows)
K = 4; n = len(rows) // K + 1
oof = {}
for k in range(K):
    ev, tr = rows[k*n:(k+1)*n], rows[:k*n] + rows[(k+1)*n:]
    open(f"{W}/oof{k}_tr.jsonl", "w").writelines(tr); open(f"{W}/oof{k}_ev.jsonl", "w").writelines(ev)
    run("--train_file", f"oof{k}_tr.jsonl", "--dev_file", f"oof{k}_ev.jsonl", "--out", f"{E}/oof{k}", "--seed", str(300 + k), "--holdout", "500")
    lg = np.load(f"{E}/oof{k}/dev_logits.npz")
    for i, l in enumerate(ev):
        r = json.loads(l)
        z = lg["route"][i]; p = np.exp(z - z.max()); p /= p.sum()
        oof[r["request_id"]] = (r["labels"], p, 1 / (1 + np.exp(-lg["approve"][i])), 1 / (1 + np.exp(-lg["grounded"][i])))
# 2) flag confident disagreements
drop, relabel = {}, {}
for rid, (lab, pr, pa, pg) in oof.items():
    d = {}
    if pr[QUEUES.index(lab["route"])] < 0.1 and pr.max() > 0.8:
        d["route"] = QUEUES[int(pr.argmax())]
    for t, p in (("approve", pa), ("grounded", pg)):
        if (lab[t] and p < 0.1) or ((not lab[t]) and p > 0.9):
            d[t] = bool(p > 0.5)
    if d:
        relabel[rid] = d; drop[rid] = {t: None for t in d}
cnt = {t: sum(t in d for d in relabel.values()) for t in ("route", "approve", "grounded")}
print("flagged", len(relabel), "rows; per task", cnt, flush=True)
json.dump(drop, open(W + "/clean_drop.json", "w")); json.dump(relabel, open(W + "/clean_relabel.json", "w"))
# 3) cleaned training, same seeds as the served MoE comparison
for s in ("13", "14"):
    for v in ("drop", "relabel"):
        run("--data", "data", "--out", f"{E}/{v}_s{s}", "--seed", s, "--clean_file", f"clean_{v}.json")
for k in range(K): shutil.rmtree(f"{E}/oof{k}")
shutil.copy(W + "/clean_relabel.json", E + "/clean_relabel.json")
shutil.make_archive("/kaggle/working/clean_logits", "zip", E)
shutil.rmtree(E); shutil.rmtree(W)
print(os.listdir("/kaggle/working"))
