"""CPU serving for the shared encoder: one ONNX session, three heads."""
import json
import math
import os

import numpy as np
import onnxruntime as ort
from tokenizers import Tokenizer

from .inputs import encode_inputs

TASK_IDS = {"route": 0, "approve": 1, "grounded": 2}

MAX_LEN = 192


class EncoderBackend:
    def __init__(self, artifacts_dir, threads=None):
        so = ort.SessionOptions()
        so.intra_op_num_threads = threads or int(os.environ.get("DECIDE_THREADS", "4"))
        so.inter_op_num_threads = 1
        so.execution_mode = ort.ExecutionMode.ORT_SEQUENTIAL
        so.graph_optimization_level = ort.GraphOptimizationLevel.ORT_ENABLE_ALL
        self.sess = ort.InferenceSession(os.path.join(artifacts_dir, "decide.onnx"), so,
                                         providers=["CPUExecutionProvider"])
        self.tok = Tokenizer.from_file(os.path.join(artifacts_dir, "tok", "tokenizer.json"))
        self.tok.enable_truncation(MAX_LEN)
        self.tok.no_padding()
        with open(os.path.join(artifacts_dir, "calibration.json"), encoding="utf-8") as fh:
            self.temp = json.load(fh)
        # Optional post-hoc maps (fit_posthoc.py): isotonic curves for the binary
        # heads, and an isotonic map from the route top-2 logit margin to the
        # top-queue probability. Absent maps fall back to temperature only.
        self.maps = {}
        mp = os.path.join(artifacts_dir, "posthoc.json")
        if os.path.exists(mp):
            with open(mp, encoding="utf-8") as fh:
                self.maps = {k: (np.asarray(v["x"]), np.asarray(v["y"])) for k, v in json.load(fh).items()}
        self.outputs = {"route": ["route"], "approve": ["approve"], "grounded": ["grounded"]}
        # Feed only the inputs this graph declares (an architecture that ignores
        # the task id exports without a "task" input).
        self.inputs = {i.name for i in self.sess.get_inputs()}

    def _logits(self, task, s):
        a, b, tab = encode_inputs(task, s)
        e = self.tok.encode(a, b)
        feed = {
            "input_ids": np.asarray([e.ids], dtype=np.int64),
            "attention_mask": np.asarray([e.attention_mask], dtype=np.int64),
            "token_type_ids": np.asarray([e.type_ids], dtype=np.int64),
            "tab": tab[None],
            "task": np.asarray([TASK_IDS[task]], dtype=np.int64),
        }
        feed = {k: v for k, v in feed.items() if k in self.inputs}
        return self.sess.run(self.outputs[task], feed)[0][0]

    def route_probs(self, s):
        raw = self._logits("route", s)
        z = raw / self.temp["route"]
        z = np.exp(z - z.max())
        p = z / z.sum()
        if "route_margin" in self.maps:
            top2 = np.sort(raw)[-2:]
            i = int(p.argmax())
            top = float(np.interp(top2[1] - top2[0], *self.maps["route_margin"]))
            rest = p.copy()
            rest[i] = 0.0
            rest = rest / rest.sum() * (1.0 - top)
            rest[i] = top
            p = rest
        return p

    def _binary(self, task, s):
        z = float(self._logits(task, s)) / self.temp[task]
        p = 1.0 / (1.0 + math.exp(-z))
        if task in self.maps:
            p = float(np.interp(p, *self.maps[task]))
        return p

    def approve_prob(self, s):
        return self._binary("approve", s)

    def grounded_prob(self, s):
        return self._binary("grounded", s)


class EnsembleBackend(EncoderBackend):
    """Several member networks merged into ONE ONNX graph (fit_ensemble.py).

    Each member is the full shared-encoder + mixture-of-experts model and
    answers all three decisions; a single session.run computes every member,
    and the decision's logit is the mean of the members' temperature-scaled
    logits. ensemble.json lists the members (input/output prefix, tokenizer).
    """

    def __init__(self, artifacts_dir, threads=None):
        with open(os.path.join(artifacts_dir, "ensemble.json"), encoding="utf-8") as fh:
            self.cfg = json.load(fh)
        so = ort.SessionOptions()
        so.intra_op_num_threads = threads or int(os.environ.get("DECIDE_THREADS", "4"))
        so.inter_op_num_threads = 1
        so.execution_mode = ort.ExecutionMode.ORT_SEQUENTIAL
        so.graph_optimization_level = ort.GraphOptimizationLevel.ORT_ENABLE_ALL
        self.sess = ort.InferenceSession(os.path.join(artifacts_dir, "decide.onnx"), so,
                                         providers=["CPUExecutionProvider"])
        self.inputs = {i.name for i in self.sess.get_inputs()}
        self.toks = []
        for m in self.cfg["members"]:
            tk = Tokenizer.from_file(os.path.join(artifacts_dir, m["tok"], "tokenizer.json"))
            tk.enable_truncation(MAX_LEN)
            tk.no_padding()
            self.toks.append(tk)
        with open(os.path.join(artifacts_dir, "calibration.json"), encoding="utf-8") as fh:
            self.temp = json.load(fh)
        self.maps = {}

    def member_logits(self, task, s):
        """Raw logits of every member for one decision (one session.run)."""
        a, b, tab = encode_inputs(task, s)
        feed = {}
        for m, tk in zip(self.cfg["members"], self.toks):
            e = tk.encode(a, b)
            p = m["prefix"]
            feed.update({
                p + "input_ids": np.asarray([e.ids], dtype=np.int64),
                p + "attention_mask": np.asarray([e.attention_mask], dtype=np.int64),
                p + "token_type_ids": np.asarray([e.type_ids], dtype=np.int64),
                p + "tab": tab[None],
                p + "task": np.asarray([TASK_IDS[task]], dtype=np.int64),
            })
        feed = {k: v for k, v in feed.items() if k in self.inputs}
        outs = self.sess.run([m["prefix"] + task for m in self.cfg["members"]], feed)
        return [o[0] for o in outs]

    def _logits(self, task, s):
        zs = self.member_logits(task, s)
        mt = self.cfg["member_temp"]
        return np.mean([z / mt[i][task] for i, z in enumerate(zs)], axis=0)
