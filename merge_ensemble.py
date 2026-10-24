"""Merge two trained member models into ONE ONNX graph (the served ensemble).

Each member keeps its own inputs/outputs under a prefix (m_ and e_), and its
dynamic dimensions are renamed per member so the two tokenisers' different
sequence lengths do not clash. Then run fit_ensemble.py on the output folder.

    python merge_ensemble.py <minilm dir> <ettin dir> <out dir>
    python fit_ensemble.py <out dir> data/train.jsonl

The served artifacts were built from the MiniLM int8 model and the Ettin-17m
model at full precision (int8 breaks Ettin; see docs/EXPERIMENTS.md).
"""
import os
import shutil
import sys

import onnx
from onnx import compose


def rename_dims(model, prefix):
    for vi in list(model.graph.input) + list(model.graph.output) + list(model.graph.value_info):
        for d in vi.type.tensor_type.shape.dim:
            if d.dim_param:
                d.dim_param = prefix + d.dim_param
    return model


def main(minilm_dir, ettin_dir, out):
    os.makedirs(out, exist_ok=True)
    a = rename_dims(compose.add_prefix(onnx.load(os.path.join(minilm_dir, "decide.onnx")), prefix="m_"), "m_")
    b = rename_dims(compose.add_prefix(onnx.load(os.path.join(ettin_dir, "decide.onnx")), prefix="e_"), "e_")
    merged = compose.merge_models(a, b, io_map=[])
    onnx.checker.check_model(merged)
    onnx.save(merged, os.path.join(out, "decide.onnx"))
    shutil.copytree(os.path.join(minilm_dir, "tok"), os.path.join(out, "tok_m"), dirs_exist_ok=True)
    shutil.copytree(os.path.join(ettin_dir, "tok"), os.path.join(out, "tok_e"), dirs_exist_ok=True)
    shutil.copy(os.path.join(minilm_dir, "cal_ids.json"), os.path.join(out, "cal_ids.json"))
    print("wrote", os.path.join(out, "decide.onnx"))


if __name__ == "__main__":
    main(*sys.argv[1:])
