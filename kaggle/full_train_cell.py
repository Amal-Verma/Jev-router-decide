import glob, os, shutil, subprocess, sys
src = os.path.dirname(glob.glob("/kaggle/input/**/all.jsonl", recursive=True)[0])
W = "/kaggle/working/src"
shutil.copytree(src, W, dirs_exist_ok=True)
os.makedirs(W + "/decide", exist_ok=True)
open(W + "/decide/__init__.py", "w").close()
for m in ("features.py", "tabular.py", "inputs.py"):
    shutil.move(os.path.join(W, m), os.path.join(W, "decide", m))
subprocess.run([sys.executable, "-m", "pip", "install", "-q", "onnx", "onnxruntime", "onnxscript"], check=True)
subprocess.run([sys.executable, "train_encoder.py", "--train_file", "all.jsonl", "--dev_file", "dev.jsonl",
                "--out", "/kaggle/working/art_full", "--epochs", "5", "--seed", "13",
                "--base", "cross-encoder/ms-marco-MiniLM-L6-v2"], cwd=W, check=True)
shutil.make_archive("/kaggle/working/art_full", "zip", "/kaggle/working/art_full")
shutil.rmtree("/kaggle/working/art_full"); shutil.rmtree(W)
print(os.listdir("/kaggle/working"))
