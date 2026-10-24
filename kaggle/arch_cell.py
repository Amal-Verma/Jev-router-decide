import glob, os, shutil, subprocess, sys
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
def run(out, *extra):
    subprocess.run([sys.executable, "train_encoder.py", "--data", "data", "--out", f"{E}/{out}",
                    "--epochs", "5", "--no_export", *extra], cwd=W, check=True)
run("teacher", "--base", "cross-encoder/nli-MiniLM2-L6-H768", "--seed", "13", "--save_all_logits")
for s in ("13", "14"):
    for arch in ("moe", "trunk", "linear"):
        run(f"{arch}_s{s}", "--base", "cross-encoder/ms-marco-MiniLM-L6-v2", "--arch", arch, "--seed", s)
    run(f"distill_s{s}", "--base", "cross-encoder/ms-marco-MiniLM-L6-v2", "--arch", "moe", "--seed", s,
        "--teacher", f"{E}/teacher")
for d in glob.glob(E + "/*/all_logits.npz"): os.remove(d)
shutil.make_archive("/kaggle/working/arch_logits", "zip", E)
shutil.rmtree(E); shutil.rmtree(W)
print(os.listdir("/kaggle/working"))
