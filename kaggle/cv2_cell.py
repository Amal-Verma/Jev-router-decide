import glob, os, shutil, subprocess, sys
src = os.path.dirname(glob.glob("/kaggle/input/**/fold0_train.jsonl", recursive=True)[0])
W = "/kaggle/working/src"
shutil.copytree(src, W, dirs_exist_ok=True)
os.makedirs(W + "/decide", exist_ok=True)
open(W + "/decide/__init__.py", "w").close()
for m in ("features.py", "tabular.py", "inputs.py"):
    shutil.move(os.path.join(W, m), os.path.join(W, "decide", m))
for name, base in [("h384", "cross-encoder/ms-marco-MiniLM-L6-v2"), ("h768", "cross-encoder/nli-MiniLM2-L6-H768")]:
    for k in range(4):
        subprocess.run([sys.executable, "train_encoder.py", "--train_file", f"fold{k}_train.jsonl",
                        "--dev_file", f"fold{k}_eval.jsonl", "--out", f"/kaggle/working/cv2/{name}/fold{k}",
                        "--epochs", "5", "--seed", str(200 + k), "--base", base,
                        "--no_export", "--every_epoch"], cwd=W, check=True)
shutil.make_archive("/kaggle/working/cv2_logits", "zip", "/kaggle/working/cv2")
shutil.rmtree("/kaggle/working/cv2"); shutil.rmtree(W)
print(os.listdir("/kaggle/working"))
