"""Multi-task serving on one shared backbone by swapping sparse TaLoS task vectors.

Memory is one backbone plus each task's ~40%-dense delta and head, instead of a full
model per task. Swapping touches only the masked positions: restore the previous
task's positions from the pristine copy, then add the new task's tau. Requests are
serialized on a lock, so throughput is best when traffic for one task arrives together.
"""

from __future__ import annotations

import threading
import time
from pathlib import Path

import numpy as np
import torch
from torch import nn

from talos_mlops import bundle as B
from talos_mlops.model import build_backbone
from talos_mlops.serving.predictors import TaskInfo, discover_tasks
from talos_mlops.sparse import SparseTaskVector


class _Task:
    def __init__(self, bundle_dir: Path, params: dict[str, torch.Tensor]):
        tv = SparseTaskVector.load(bundle_dir / B.TASK_VECTOR)
        self.deltas = {
            n: (torch.from_numpy(e.indices), torch.from_numpy(e.values).to(params[n].dtype))
            for n, e in tv.entries.items()
        }
        weight, bias = B.load_head(bundle_dir)
        self.head = nn.Linear(weight.shape[1], weight.shape[0])
        with torch.no_grad():
            self.head.weight.copy_(torch.from_numpy(weight))
            self.head.bias.copy_(torch.from_numpy(bias))
        self.head.eval()


class TaskVectorPredictor:
    backend = "taskvector"

    def __init__(self, store: Path, threads: int = 0, on_swap=None):
        if threads:
            torch.set_num_threads(threads)
        self.tasks: dict[str, TaskInfo] = discover_tasks(store, (B.HEAD, B.TASK_VECTOR))
        if not self.tasks:
            self._tasks: dict[str, _Task] = {}
            return
        backbones = {
            (t.meta["backbone"], str(t.meta.get("backbone_kwargs", {}))) for t in self.tasks.values()
        }
        if len(backbones) != 1:
            raise ValueError(f"all tasks must share one backbone, found {backbones}")
        first = next(iter(self.tasks.values())).meta
        if not first.get("pretrained", True):
            torch.manual_seed(first["seed"])
        self.backbone = build_backbone(
            first["backbone"], first.get("pretrained", True), **first.get("backbone_kwargs", {})
        ).eval()
        self.backbone.requires_grad_(False)
        self._params = dict(self.backbone.named_parameters())
        self._pristine = {n: p.detach().clone() for n, p in self._params.items()}
        self._tasks = {name: _Task(store / name, self._params) for name in self.tasks}
        self._active: str | None = None
        self._lock = threading.Lock()
        self._on_swap = on_swap

    @torch.no_grad()
    def _activate(self, task: str) -> None:
        t0 = time.perf_counter()
        if self._active is not None:
            for n, (idx, _) in self._tasks[self._active].deltas.items():
                self._params[n].view(-1)[idx] = self._pristine[n].view(-1)[idx]
        for n, (idx, vals) in self._tasks[task].deltas.items():
            self._params[n].view(-1).index_add_(0, idx, vals)
        if self._on_swap:
            self._on_swap(self._active, task, time.perf_counter() - t0)
        self._active = task

    def predict(self, batch: np.ndarray, task: str) -> np.ndarray:
        x = torch.from_numpy(batch)
        with self._lock, torch.inference_mode():
            if self._active != task:
                self._activate(task)
            return self._tasks[task].head(self.backbone(x)).numpy()
