"""Train the shared multi-task decision model and export it to ONNX.

One encoder (cross-encoder/nli-MiniLM2-L6-H768) is fine-tuned on all three
decisions at once. Every request goes through the same encoder; only a small
head per decision type sits on top (see DecideNet). Adding a fourth decision
means adding one more head and its examples, not a new model.

    python train_encoder.py --data data --out artifacts [--epochs 4]

Runs on GPU when available (Kaggle T4), otherwise CPU. Writes:
    artifacts/decide.onnx       int8 dynamic-quantised graph (all three heads)
    artifacts/tokenizer.json    fast tokenizer
    artifacts/calibration.json  per-head temperatures
    artifacts/policy.json       abstention thresholds
"""
import argparse
import re
import json
import math
import os
import random
import sys

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from transformers import AutoModel, AutoTokenizer

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from decide.inputs import N_TAB, QUEUES, encode_inputs  # noqa: E402

BASE = "cross-encoder/nli-MiniLM2-L6-H768"
TASKS = ["route", "approve", "grounded"]


class DecideNet(nn.Module):
    """Shared encoder -> shared mixture of experts -> one tiny head per decision.

    The encoder and the experts are shared by every decision type. A gate that
    sees the pooled text and the task embedding picks a soft mix of experts, so
    related decisions can reuse experts while differently shaped ones (grounding
    compares two texts; approval leans on amount/tier) can specialise. Adding a
    fourth decision = one more task embedding row and one more output head.
    Every forward returns all three logits; the runtime reads the one it needs.
    """

    def __init__(self, base, n_experts=4, d_expert=256, arch="moe"):
        super().__init__()
        self.arch = arch
        self.encoder = AutoModel.from_pretrained(base)
        h = self.encoder.config.hidden_size
        # ModernBERT-style encoders (e.g. Ettin) have no segment embeddings.
        self.use_tt = getattr(self.encoder.config, "type_vocab_size", 0) > 1
        self.task_emb = nn.Embedding(len(TASKS), 32)
        self.tab = nn.Sequential(nn.Linear(N_TAB, 64), nn.GELU())
        if arch == "moe":
            self.experts = nn.ModuleList([
                nn.Sequential(nn.Dropout(0.1), nn.Linear(h + 64 + 32, d_expert), nn.GELU())
                for _ in range(n_experts)])
            self.gate = nn.Linear(h + 32, n_experts)
            d = d_expert
        elif arch == "trunk":      # one shared MLP layer, no experts or gate
            self.trunk = nn.Sequential(nn.Dropout(0.1), nn.Linear(h + 64 + 32, d_expert), nn.GELU())
            d = d_expert
        elif arch == "linear":     # heads read the encoder summary directly
            self.drop = nn.Dropout(0.1)
            d = h + 64
        else:
            raise ValueError(arch)
        self.route_head = nn.Linear(d, len(QUEUES))
        self.approve_head = nn.Linear(d, 1)
        self.grounded_head = nn.Linear(d, 1)

    def encode(self, input_ids, attention_mask, token_type_ids):
        kw = {"token_type_ids": token_type_ids} if self.use_tt else {}
        out = self.encoder(input_ids=input_ids, attention_mask=attention_mask, **kw).last_hidden_state
        m = attention_mask.unsqueeze(-1).to(out.dtype)
        return (out * m).sum(1) / m.sum(1).clamp(min=1.0)

    def forward(self, input_ids, attention_mask, token_type_ids, tab, task):
        pooled = self.encode(input_ids, attention_mask, token_type_ids)
        return self.post(pooled, tab, task)[1:]

    def post(self, pooled, tab, task):
        """Everything after the encoder. Returns (z, route, approve, grounded),
        where z is the representation the heads read."""
        te = self.task_emb(task)
        if self.arch == "linear":
            z = self.drop(torch.cat([pooled, self.tab(tab)], -1))
        elif self.arch == "trunk":
            z = self.trunk(torch.cat([pooled, self.tab(tab), te], -1))
        else:
            x = torch.cat([pooled, self.tab(tab), te], -1)
            g = torch.softmax(self.gate(torch.cat([pooled, te], -1)), -1)      # (b, E)
            mix = torch.stack([e(x) for e in self.experts], 1)                 # (b, E, d)
            z = (g.unsqueeze(-1) * mix).sum(1)
        return z, self.route_head(z), self.approve_head(z).squeeze(-1), self.grounded_head(z).squeeze(-1)


def load(p):
    return [json.loads(l) for l in open(p, encoding="utf-8") if l.strip()]


def label_of(task, r):
    y = r["labels"][task]
    return QUEUES.index(y) if task == "route" else float(y)


# ---- label-safe data augmentation ------------------------------------------------
FILLER = re.compile(r"^(hi|hello|hey|thanks|thank you|regards|cheers|best|please advise|let me know|appreciate the help|"
                    r"any help would be great|sent from my phone|good morning|hi team|hi there)\b", re.I)
DUR_FORMS = {7: ["7 days", "a week", "one week", "seven days"], 14: ["14 days", "two weeks", "fourteen days"],
             30: ["30 days", "a month", "thirty days"], 2: ["2 days", "48 hours", "two days"], 1: ["1 day", "24 hours", "one day"],
             3: ["3 days", "three days"], 5: ["5 days", "five days"], 10: ["10 days", "ten days"], 21: ["21 days", "three weeks"],
             60: ["60 days", "two months"], 90: ["90 days", "three months"]}
DUR_RE = re.compile(r"\b(\d+|a|one|two|three|five|seven|ten|fourteen|thirty|forty-eight)\s+(days?|weeks?|months?|hours?)\b", re.I)
WORDNUM = {"a": 1, "one": 1, "two": 2, "three": 3, "five": 5, "seven": 7, "ten": 10, "fourteen": 14, "thirty": 30, "forty-eight": 48}
UNIT = {"day": 1, "week": 7, "month": 30, "hour": 1 / 24}


def _days(m):
    n = m.group(1).lower()
    n = WORDNUM.get(n) if not n.isdigit() else int(n)
    u = m.group(2).lower().rstrip("s")
    return None if n is None else n * UNIT[u]


def _sentences(t):
    return [x for x in re.split(r"(?<=[.!?])\s+", t.strip()) if x]


def augment(task, r, rng):
    """Return a new (task, row) with a label-safe change, or None."""
    st, lab = dict(r["state"]), dict(r["labels"])
    if task in ("route", "approve"):
        sents = _sentences(st.get("body") or "")
        if len(sents) < 2:
            return None
        if task == "route":
            sents = [x for x in sents if not (FILLER.match(x) and rng.random() < 0.5)] or sents
        rng.shuffle(sents)
        st["body"] = " ".join(sents)
    else:
        draft = st.get("draft_reply") or ""
        ms = [m for m in DUR_RE.finditer(draft) if _days(m) is not None]
        if not ms:
            return None
        m = rng.choice(ms)
        d = _days(m)
        if lab["grounded"] and rng.random() < 0.5 and d in DUR_FORMS:
            new = rng.choice([f for f in DUR_FORMS[d] if f.lower() != m.group(0).lower()] or [m.group(0)])
        elif lab["grounded"]:
            alt = [k for k in DUR_FORMS if k != d and abs(k - d) >= 2]
            if not alt:
                return None
            new = rng.choice(DUR_FORMS[rng.choice(alt)])
            lab["grounded"] = False
        else:
            return None
        st["draft_reply"] = draft[:m.start()] + new + draft[m.end():]
    return task, dict(r, state=st, labels=lab, request_id=r["request_id"] + "#aug")


def batches(tok, items, bs, shuffle, max_len):
    idx = list(range(len(items)))
    if shuffle:
        random.shuffle(idx)
    # group by task so each batch is one task (one loss, cleaner padding)
    by_task = {t: [i for i in idx if items[i][0] == t] for t in TASKS}
    chunks = [(t, ids[k:k + bs]) for t, ids in by_task.items() for k in range(0, len(ids), bs)]
    if shuffle:
        random.shuffle(chunks)
    for task, ids in chunks:
        a, b, tab = zip(*(encode_inputs(task, items[i][1]["state"]) for i in ids))
        enc = tok(list(a), list(b), truncation=True, max_length=max_len, padding=True, return_tensors="pt")
        if "token_type_ids" not in enc:
            enc["token_type_ids"] = torch.zeros_like(enc["input_ids"])
        y = torch.tensor([label_of(task, items[i][1]) for i in ids])
        enc["task"] = torch.full((len(ids),), TASKS.index(task), dtype=torch.long)
        enc["rid"] = [items[i][1]["request_id"] for i in ids]
        yield task, enc, torch.tensor(np.stack(tab)), y


@torch.no_grad()
def bayes_outputs(model, tok, rows, dev, max_len, n_mc=20, bs=128):
    """Per task: head inputs z (for last-layer Laplace) and MC-dropout mean
    probabilities (encoder run once, dropout on in the post-encoder part only)."""
    model.eval()
    post_dropouts = [m for n, m in model.named_modules()
                     if isinstance(m, nn.Dropout) and not n.startswith("encoder.")]
    out = {}
    for task in TASKS:
        ti = TASKS.index(task)
        zs, mcs = [], []
        for _, enc, tab, _ in batches(tok, [(task, r) for r in rows], bs, False, max_len):
            args = [enc[k].to(dev) for k in ("input_ids", "attention_mask", "token_type_ids")]
            pooled = model.encode(*args)
            tab_, task_ = tab.to(dev), enc["task"].to(dev)
            zs.append(model.post(pooled, tab_, task_)[0].float().cpu())
            for m in post_dropouts:
                m.train()
            acc = 0
            for _ in range(n_mc):
                lg = model.post(pooled, tab_, task_)[1 + ti].float()
                acc = acc + (torch.softmax(lg, -1) if task == "route" else torch.sigmoid(lg))
            for m in post_dropouts:
                m.eval()
            mcs.append((acc / n_mc).cpu())
        out[task + "_z"] = torch.cat(zs).numpy()
        out[task + "_mc"] = torch.cat(mcs).numpy()
    return out


@torch.no_grad()
def predict(model, tok, rows, dev, max_len, bs=128):
    """Raw logits per task, in row order."""
    model.eval()
    out = {}
    for task in TASKS:
        items = [(task, r) for r in rows]
        logits = []
        for _, enc, tab, _ in batches(tok, items, bs, False, max_len):
            res = model(enc["input_ids"].to(dev), enc["attention_mask"].to(dev),
                        enc["token_type_ids"].to(dev), tab.to(dev), enc["task"].to(dev))
            logits.append(res[TASKS.index(task)].float().cpu())
        out[task] = torch.cat(logits).numpy()
    return out


def task_loss(task, z, y, args):
    """Cross-entropy, optionally with label smoothing, focal weighting
    (Mukhoti et al. 2020) and an entropy-minimisation term (Feng et al. 2022)."""
    eps, g, lam = args.label_smoothing, args.focal_gamma, args.ent_reg
    if task == "route":
        logp = F.log_softmax(z, -1)
        tgt = F.one_hot(y.long(), z.shape[-1]).float()
        if eps:
            tgt = tgt * (1 - eps) + eps / z.shape[-1]
        ce = -(tgt * logp).sum(-1)
        if g:
            pt = logp.gather(1, y.long()[:, None]).squeeze(1).exp()
            ce = (1 - pt) ** g * ce
        loss = ce.mean()
        if lam:
            loss = loss + lam * -(logp.exp() * logp).sum(-1).mean()
    else:
        yt = y.float() * (1 - eps) + eps / 2 if eps else y.float()
        ce = F.binary_cross_entropy_with_logits(z, yt, reduction="none")
        if g:
            p = torch.sigmoid(z)
            pt = torch.where(y > 0.5, p, 1 - p)
            ce = (1 - pt) ** g * ce
        loss = ce.mean()
        if lam:
            p = torch.sigmoid(z).clamp(1e-6, 1 - 1e-6)
            loss = loss + lam * -(p * p.log() + (1 - p) * (1 - p).log()).mean()
    return loss


def fit_temperature(logits, y, multiclass):
    """Single-parameter temperature scaling (Guo et al. 2017) by grid search on NLL."""
    best, best_t = 1e9, 1.0
    for t in np.exp(np.linspace(math.log(0.3), math.log(5.0), 200)):
        z = torch.tensor(logits / t)
        if multiclass:
            nll = F.cross_entropy(z, torch.tensor(y, dtype=torch.long)).item()
        else:
            nll = F.binary_cross_entropy_with_logits(z, torch.tensor(y, dtype=torch.float32)).item()
        if nll < best:
            best, best_t = nll, float(t)
    return best_t


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", default="data")
    ap.add_argument("--out", default="artifacts")
    ap.add_argument("--epochs", type=int, default=4)
    ap.add_argument("--lr", type=float, default=4e-5)
    ap.add_argument("--bs", type=int, default=32)
    ap.add_argument("--max_len", type=int, default=192)
    ap.add_argument("--holdout", type=int, default=1000)
    ap.add_argument("--seed", type=int, default=13)
    ap.add_argument("--base", default=BASE)
    ap.add_argument("--train_file", default=None, help="overrides <data>/train.jsonl")
    ap.add_argument("--dev_file", default=None, help="overrides <data>/dev.jsonl")
    ap.add_argument("--no_export", action="store_true", help="skip ONNX export (CV runs)")
    ap.add_argument("--arch", default="moe", choices=["moe", "trunk", "linear"])
    ap.add_argument("--n_experts", type=int, default=4)
    ap.add_argument("--keep_fp32", action="store_true",
                    help="also keep decide_fp32.onnx (needed for Ettin, which int8 quantisation breaks)")
    ap.add_argument("--save_all_logits", action="store_true",
                    help="teacher mode: save logits for every training-file row (all_logits.npz, all_ids.json)")
    ap.add_argument("--teacher", default=None, help="dir with a teacher's all_logits.npz / all_ids.json")
    ap.add_argument("--distill_alpha", type=float, default=0.5)
    ap.add_argument("--distill_T", type=float, default=2.0)
    ap.add_argument("--clean_file", default=None,
                    help="JSON {request_id: {task: new_label | null}} applied to the TRAINING part only "
                         "(null drops that task's example); the calibration holdout keeps original labels")
    ap.add_argument("--augment", type=float, default=0.0,
                    help="fraction of training items to add as label-safe augmented copies each epoch")
    ap.add_argument("--label_smoothing", type=float, default=0.0)
    ap.add_argument("--focal_gamma", type=float, default=0.0)
    ap.add_argument("--ent_reg", type=float, default=0.0, help="entropy-minimisation weight")
    ap.add_argument("--bayes", action="store_true",
                    help="save Laplace features (z) for train/cal/dev and MC-dropout probs for cal/dev")
    ap.add_argument("--every_epoch", action="store_true",
                    help="also save holdout and dev logits after each epoch (e<N>_cal.npz, e<N>_dev.npz)")
    args = ap.parse_args()

    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    dev = "cuda" if torch.cuda.is_available() else "cpu"
    os.makedirs(args.out, exist_ok=True)

    rows = load(args.train_file or os.path.join(args.data, "train.jsonl"))
    random.shuffle(rows)
    cal, train = rows[:args.holdout], rows[args.holdout:]
    devset = load(args.dev_file or os.path.join(args.data, "dev.jsonl"))
    print(f"train {len(train)}  calibration {len(cal)}  dev {len(devset)}  device {dev}", flush=True)

    tok = AutoTokenizer.from_pretrained(args.base)
    model = DecideNet(args.base, n_experts=args.n_experts, arch=args.arch).to(dev)
    teacher = None
    if args.teacher:
        tl = np.load(os.path.join(args.teacher, "all_logits.npz"))
        tid = {r: i for i, r in enumerate(json.load(open(os.path.join(args.teacher, "all_ids.json"))))}
        teacher = (tl, tid)
    items = [(t, r) for r in train for t in TASKS]
    if args.clean_file:
        fix = json.load(open(args.clean_file))
        cleaned, n_drop, n_relabel = [], 0, 0
        for t, r in items:
            ch = fix.get(r["request_id"], {})
            if t not in ch:
                cleaned.append((t, r))
            elif ch[t] is None:
                n_drop += 1
            else:
                r2 = dict(r, labels=dict(r["labels"], **{t: ch[t]}))
                cleaned.append((t, r2)); n_relabel += 1
        items = cleaned
        print(f"cleaning: dropped {n_drop}, relabelled {n_relabel} task examples", flush=True)
    # batches are grouped per task, so each task rounds up on its own
    aug_extra = 1 + args.augment
    steps = args.epochs * sum(math.ceil(aug_extra * sum(1 for t, _ in items if t == k) / args.bs) + 1 for k in TASKS)
    head_params = [p for n, p in model.named_parameters() if not n.startswith("encoder.")]
    opt = torch.optim.AdamW([
        {"params": model.encoder.parameters(), "lr": args.lr},
        {"params": head_params, "lr": args.lr * 10},
    ], weight_decay=0.01)
    sched = torch.optim.lr_scheduler.OneCycleLR(opt, max_lr=[args.lr, args.lr * 10], total_steps=steps,
                                                pct_start=0.1, anneal_strategy="linear")
    scaler = torch.amp.GradScaler(enabled=dev == "cuda")

    step = 0
    for ep in range(args.epochs):
        model.train()
        tot = {t: [0.0, 0] for t in TASKS}
        ep_items = items
        if args.augment:
            rng = random.Random(args.seed * 1000 + ep)
            extra = [a for a in (augment(t, r, rng) for t, r in rng.sample(items, int(args.augment * len(items)))) if a]
            ep_items = items + extra
            print(f"  epoch {ep + 1}: +{len(extra)} augmented items", flush=True)
        for task, enc, tab, y in batches(tok, ep_items, args.bs, True, args.max_len):
            with torch.autocast(device_type=dev, dtype=torch.float16, enabled=dev == "cuda"):
                res = model(enc["input_ids"].to(dev), enc["attention_mask"].to(dev),
                            enc["token_type_ids"].to(dev), tab.to(dev), enc["task"].to(dev))
                z = res[TASKS.index(task)].float()
                y = y.to(dev)
                loss = task_loss(task, z, y, args)
                if teacher is not None:
                    T, a = args.distill_T, args.distill_alpha
                    t = torch.tensor(teacher[0][task][[teacher[1][i] for i in enc["rid"]]], device=dev).float()
                    if task == "route":
                        soft = F.kl_div(F.log_softmax(z / T, -1), F.softmax(t / T, -1), reduction="batchmean")
                    else:
                        soft = F.binary_cross_entropy_with_logits(z / T, torch.sigmoid(t / T))
                    loss = (1 - a) * loss + a * soft * T * T
            opt.zero_grad(set_to_none=True)
            scaler.scale(loss).backward()
            scaler.unscale_(opt)
            nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            scaler.step(opt)
            scaler.update()
            sched.step()
            step += 1
            tot[task][0] += loss.item()
            tot[task][1] += 1
            if step % 100 == 0:
                print(f"  step {step}/{steps}", {t: round(v[0] / max(1, v[1]), 4) for t, v in tot.items()}, flush=True)
        lg = predict(model, tok, devset, dev, args.max_len)
        acc_r = np.mean(lg["route"].argmax(1) == [QUEUES.index(r["labels"]["route"]) for r in devset])
        acc_a = np.mean((lg["approve"] > 0) == [r["labels"]["approve"] for r in devset])
        acc_g = np.mean((lg["grounded"] > 0) == [r["labels"]["grounded"] for r in devset])
        print(f"epoch {ep + 1}: dev acc route {acc_r:.4f} approve {acc_a:.4f} grounded {acc_g:.4f}", flush=True)
        if args.every_epoch:
            np.savez(os.path.join(args.out, f"e{ep + 1}_dev.npz"), **lg)
            np.savez(os.path.join(args.out, f"e{ep + 1}_cal.npz"), **predict(model, tok, cal, dev, args.max_len))

    # ---- calibration on the held-out slice of train --------------------------
    lc = predict(model, tok, cal, dev, args.max_len)
    temps = {
        "route": fit_temperature(lc["route"], [QUEUES.index(r["labels"]["route"]) for r in cal], True),
        "approve": fit_temperature(lc["approve"], [float(r["labels"]["approve"]) for r in cal], False),
        "grounded": fit_temperature(lc["grounded"], [float(r["labels"]["grounded"]) for r in cal], False),
    }
    print("temperatures", temps, flush=True)
    with open(os.path.join(args.out, "calibration.json"), "w") as fh:
        json.dump(temps, fh, indent=2)
    np.savez(os.path.join(args.out, "cal_logits.npz"), **{k: v for k, v in lc.items()})
    ld = predict(model, tok, devset, dev, args.max_len)
    np.savez(os.path.join(args.out, "dev_logits.npz"), **{k: v for k, v in ld.items()})
    with open(os.path.join(args.out, "cal_ids.json"), "w") as fh:
        json.dump([r["request_id"] for r in cal], fh)
    if args.bayes:
        for name, rs in (("train", train), ("cal", cal), ("dev", devset)):
            np.savez(os.path.join(args.out, f"bayes_{name}.npz"), **bayes_outputs(model, tok, rs, dev, args.max_len))
        np.savez(os.path.join(args.out, "heads.npz"),
                 **{n.replace(".", "_"): p.detach().float().cpu().numpy()
                    for n, p in model.named_parameters() if "_head" in n})
        with open(os.path.join(args.out, "train_ids.json"), "w") as fh:
            json.dump([r["request_id"] for r in train], fh)
    if args.save_all_logits:
        la = predict(model, tok, rows, dev, args.max_len)
        np.savez(os.path.join(args.out, "all_logits.npz"), **la)
        with open(os.path.join(args.out, "all_ids.json"), "w") as fh:
            json.dump([r["request_id"] for r in rows], fh)

    # ---- export ----------------------------------------------------------------
    if args.no_export:
        return
    model = model.float().cpu().eval()
    tok.save_pretrained(os.path.join(args.out, "tok"))
    try:
        export_onnx(model, tok, args.out, args.max_len, keep_fp32=args.keep_fp32)
    except Exception as e:                       # keep the logits even if export fails
        print("ONNX export failed:", type(e).__name__, str(e)[:300], flush=True)


def export_onnx(model, tok, out, max_len, keep_fp32=False):
    from onnxruntime.quantization import QuantType, quantize_dynamic

    enc = tok(["a b c"], ["d e f"], return_tensors="pt")
    if "token_type_ids" not in enc:
        enc["token_type_ids"] = torch.zeros_like(enc["input_ids"])
    tab = torch.zeros(1, N_TAB)
    task = torch.zeros(1, dtype=torch.long)
    fp32 = os.path.join(out, "decide_fp32.onnx")
    torch.onnx.export(
        model, (enc["input_ids"], enc["attention_mask"], enc["token_type_ids"], tab, task), fp32,
        input_names=["input_ids", "attention_mask", "token_type_ids", "tab", "task"],
        output_names=["route", "approve", "grounded"],
        dynamic_axes={"input_ids": {0: "b", 1: "s"}, "attention_mask": {0: "b", 1: "s"},
                      "token_type_ids": {0: "b", 1: "s"}, "tab": {0: "b"}, "task": {0: "b"},
                      "route": {0: "b"}, "approve": {0: "b"}, "grounded": {0: "b"}},
        opset_version=17, **({"dynamo": False} if "dynamo" in torch.onnx.export.__code__.co_varnames else {}))
    quantize_dynamic(fp32, os.path.join(out, "decide.onnx"), weight_type=QuantType.QInt8)
    if not keep_fp32:
        os.remove(fp32)
    print("exported", os.path.join(out, "decide.onnx"), flush=True)


if __name__ == "__main__":
    main()
