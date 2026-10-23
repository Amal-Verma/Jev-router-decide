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
        self.outputs = {"route": ["route"], "approve": ["approve"], "grounded": ["grounded"]}

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
        return self.sess.run(self.outputs[task], feed)[0][0]

    def route_probs(self, s):
        z = self._logits("route", s) / self.temp["route"]
        z = np.exp(z - z.max())
        return z / z.sum()

    def _binary(self, task, s):
        z = float(self._logits(task, s)) / self.temp[task]
        return 1.0 / (1.0 + math.exp(-z))

    def approve_prob(self, s):
        return self._binary("approve", s)

    def grounded_prob(self, s):
        return self._binary("grounded", s)
