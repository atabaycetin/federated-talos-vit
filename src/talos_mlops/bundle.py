"""On-disk layout of one task: everything needed to serve it on top of the public backbone.

    <bundle>/
      meta.json          task, classes, backbone id, metrics, drift reference
      head.npz           linear head (weight, bias)
      task_vector.npz    sparse TaLoS task vector for the backbone (empty for linear_probe)
      model.onnx         merged backbone + tau + head, fp32       (after `talos export`)
      model.int8.onnx    dynamic INT8 quantization of the above   (after `talos export`)
      benchmark.json                                              (after `talos benchmark`)

No full checkpoints, optimizer states or Fisher scores are written.
"""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np

META = "meta.json"
HEAD = "head.npz"
TASK_VECTOR = "task_vector.npz"
ONNX_FP32 = "model.onnx"
ONNX_INT8 = "model.int8.onnx"
BENCHMARK = "benchmark.json"


def write_meta(bundle: Path, meta: dict) -> None:
    bundle.mkdir(parents=True, exist_ok=True)
    (bundle / META).write_text(json.dumps(meta, indent=2))


def read_meta(bundle: Path) -> dict:
    return json.loads((Path(bundle) / META).read_text())


def update_meta(bundle: Path, **fields) -> dict:
    meta = read_meta(bundle)
    meta.update(fields)
    write_meta(bundle, meta)
    return meta


def save_head(bundle: Path, weight: np.ndarray, bias: np.ndarray) -> None:
    np.savez(bundle / HEAD, weight=weight.astype(np.float32), bias=bias.astype(np.float32))


def load_head(bundle: Path) -> tuple[np.ndarray, np.ndarray]:
    with np.load(Path(bundle) / HEAD, allow_pickle=False) as data:
        return data["weight"], data["bias"]


def size_mb(path: Path) -> float:
    return path.stat().st_size / 2**20
