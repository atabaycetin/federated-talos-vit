"""Benchmark stage: size, CPU latency (batch 1, p50/p95) and accuracy for each variant."""

from __future__ import annotations

import json
import time
from pathlib import Path

import numpy as np
import onnxruntime as ort
import torch

from talos_mlops import bundle as B
from talos_mlops.config import PipelineConfig
from talos_mlops.data import build_splits
from talos_mlops.model import load_task_model


def _latency_ms(fn, x: np.ndarray, iters: int, warmup: int) -> dict:
    for _ in range(warmup):
        fn(x)
    times = []
    for _ in range(iters):
        t0 = time.perf_counter()
        fn(x)
        times.append((time.perf_counter() - t0) * 1000)
    return {"p50_ms": float(np.percentile(times, 50)), "p95_ms": float(np.percentile(times, 95))}


def _predict_all(fn, images: np.ndarray, batch: int = 64) -> np.ndarray:
    return np.concatenate([fn(images[i : i + batch]).argmax(1) for i in range(0, len(images), batch)])


def run_benchmark(cfg: PipelineConfig, bundle_dir: Path) -> dict:
    b = cfg.benchmark
    test = build_splits(cfg.data.model_copy(update={"limit_eval": None}), 0.0, cfg.seed).test
    n = min(b.eval_samples, len(test))
    rng = np.random.default_rng(cfg.seed)
    idx = rng.choice(len(test), n, replace=False)
    images = np.stack([test[i][0].numpy() for i in idx])
    labels = np.array([test[i][1] for i in idx])
    single = images[:1]

    torch.set_num_threads(b.threads)
    model = load_task_model(bundle_dir).eval()

    def torch_fn(x):
        with torch.inference_mode():
            return model(torch.from_numpy(x)).numpy()

    n_params = sum(p.numel() for p in model.parameters())
    variants = {"pytorch_fp32": (torch_fn, n_params * 4 / 2**20)}

    opts = ort.SessionOptions()
    opts.intra_op_num_threads = b.threads
    for name, file in (("onnx_fp32", B.ONNX_FP32), ("onnx_int8", B.ONNX_INT8)):
        path = bundle_dir / file
        if path.exists():
            sess = ort.InferenceSession(str(path), opts, providers=["CPUExecutionProvider"])
            variants[name] = (lambda x, s=sess: s.run(None, {"input": x})[0], B.size_mb(path))

    results, reference_pred = {}, None
    for name, (fn, size) in variants.items():
        pred = _predict_all(fn, images)
        reference_pred = pred if reference_pred is None else reference_pred
        results[name] = {
            "size_mb": round(size, 2),
            "top1": float((pred == labels).mean()),
            "agreement_with_fp32": float((pred == reference_pred).mean()),
            **_latency_ms(fn, single, b.latency_iters, b.warmup_iters),
        }

    report = {
        "eval_samples": n,
        "threads": b.threads,
        "batch_size_latency": 1,
        "cpu": _cpu_name(),
        # What has to be stored per task on top of the public backbone.
        "task_bundle_mb": round(B.size_mb(bundle_dir / B.TASK_VECTOR) + B.size_mb(bundle_dir / B.HEAD), 2),
        "variants": results,
    }
    (bundle_dir / B.BENCHMARK).write_text(json.dumps(report, indent=2))
    print(format_table(report))
    return report


def format_table(report: dict) -> str:
    rows = [
        "| variant | size (MB) | top-1 | agree w/ fp32 | p50 (ms) | p95 (ms) |",
        "|---|---|---|---|---|---|",
    ]
    for name, r in report["variants"].items():
        rows.append(
            f"| {name} | {r['size_mb']:.1f} | {r['top1']:.4f} | {r['agreement_with_fp32']:.4f} "
            f"| {r['p50_ms']:.1f} | {r['p95_ms']:.1f} |"
        )
    return "\n".join(rows)


def _cpu_name() -> str:
    import platform

    return platform.processor() or platform.machine()
