"""Inference backends. Both read a model store laid out as `<store>/<task>/` bundles.

- OnnxPredictor: one INT8 ONNX Runtime session per task. Default; no torch in the image.
- TaskVectorPredictor (task_vectors.py): one shared PyTorch backbone, with the task's
  sparse TaLoS vector swapped in per request.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Protocol

import numpy as np
import onnxruntime as ort

from talos_mlops import bundle as B


@dataclass
class TaskInfo:
    name: str
    meta: dict

    @property
    def classes(self) -> list[str]:
        return self.meta["classes"]

    @property
    def version(self) -> str:
        reg = self.meta.get("registered_model")
        return f"{reg['name']}/{reg['version']}" if reg else self.meta.get("run_name", "unknown")

    def summary(self) -> dict:
        m = self.meta
        return {
            "task": self.name,
            "version": self.version,
            "method": m.get("method"),
            "num_classes": m.get("num_classes"),
            "sparsity": m.get("sparsity"),
            "task_vector_params": m.get("task_vector_params"),
            "metrics": m.get("metrics", {}),
        }


class Predictor(Protocol):
    backend: str
    tasks: dict[str, TaskInfo]

    def predict(self, batch: np.ndarray, task: str) -> np.ndarray:
        """(N,3,224,224) float32 -> (N,num_classes) logits."""


def discover_tasks(store: Path, required: tuple[str, ...]) -> dict[str, TaskInfo]:
    tasks = {}
    for d in sorted(p for p in Path(store).iterdir() if p.is_dir()):
        if (d / B.META).exists() and all((d / f).exists() for f in required):
            tasks[d.name] = TaskInfo(d.name, B.read_meta(d))
    return tasks


class OnnxPredictor:
    backend = "onnx"

    def __init__(self, store: Path, threads: int = 0, prefer_int8: bool = True):
        self.tasks: dict[str, TaskInfo] = {}
        self._sessions: dict[str, ort.InferenceSession] = {}
        opts = ort.SessionOptions()
        opts.intra_op_num_threads = threads
        order = (B.ONNX_INT8, B.ONNX_FP32) if prefer_int8 else (B.ONNX_FP32, B.ONNX_INT8)
        for name, info in discover_tasks(store, ()).items():
            path = next((store / name / f for f in order if (store / name / f).exists()), None)
            if path is None:
                continue
            self._sessions[name] = ort.InferenceSession(str(path), opts, providers=["CPUExecutionProvider"])
            info.meta["served_file"] = path.name
            self.tasks[name] = info

    def predict(self, batch: np.ndarray, task: str) -> np.ndarray:
        return self._sessions[task].run(None, {"input": batch})[0]
