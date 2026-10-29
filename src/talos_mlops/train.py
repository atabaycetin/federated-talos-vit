"""Training stage: linear-probe warmup -> Fisher mask -> sparse fine-tuning -> bundle.

`method: linear_probe` stops after the warmup; it is the fast fallback model and
produces a bundle the serving code treats exactly like a TaLoS one.
"""

from __future__ import annotations

import math
import random
import time
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import torch
from torch import nn
from tqdm import tqdm

from talos_mlops import bundle as B
from talos_mlops.config import PipelineConfig
from talos_mlops.data import build_splits, make_loader
from talos_mlops.drift import build_reference, drift_features
from talos_mlops.model import build_model, resolve_device
from talos_mlops.preprocessing import IMAGE_SIZE, MEAN, STD
from talos_mlops.sparse import (
    SparseTaskVector,
    calibrate_mask,
    fisher_scores,
    mask_gradients,
    max_frozen_drift,
)
from talos_mlops.tracking import Tracker


def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)


def warmup_cosine(total_steps: int, warmup_frac: float):
    warmup = max(1, int(total_steps * warmup_frac))

    def factor(step: int) -> float:
        if step < warmup:
            return (step + 1) / warmup
        progress = (step - warmup) / max(1, total_steps - warmup)
        return 0.5 * (1 + math.cos(math.pi * min(progress, 1.0)))

    return factor


@torch.no_grad()
def evaluate(model, loader, device, collect_drift: bool = False) -> tuple[dict, np.ndarray | None]:
    model.eval()
    criterion = nn.CrossEntropyLoss(reduction="sum")
    loss = top1 = top5 = n = 0.0
    feats = []
    for inputs, labels in loader:
        inputs, labels = inputs.to(device), labels.to(device)
        logits = model(inputs)
        loss += criterion(logits, labels).item()
        top = logits.topk(min(5, logits.shape[1]), dim=1).indices
        hits = top.eq(labels[:, None])
        top1 += hits[:, 0].sum().item()
        top5 += hits.any(dim=1).sum().item()
        n += labels.numel()
        if collect_drift:
            probs = logits.softmax(dim=1).float().cpu().numpy()
            feats.append(drift_features(inputs.float().cpu().numpy(), probs))
    metrics = {"loss": loss / n, "top1": top1 / n, "top5": top5 / n}
    return metrics, (np.concatenate(feats) if collect_drift else None)


def _train_epoch(model, loader, optimizer, scheduler, device, desc, step_hook=None) -> dict:
    model.train()
    criterion = nn.CrossEntropyLoss()
    total_loss = torch.zeros((), device=device)
    correct = torch.zeros((), device=device)
    n = 0
    start = time.perf_counter()
    for inputs, labels in tqdm(loader, desc=desc, leave=False):
        inputs, labels = inputs.to(device), labels.to(device)
        optimizer.zero_grad(set_to_none=True)
        logits = model(inputs)
        loss = criterion(logits, labels)
        loss.backward()
        if step_hook:
            step_hook()
        optimizer.step()
        scheduler.step()
        total_loss += loss.detach() * labels.numel()
        correct += logits.argmax(dim=1).eq(labels).sum()
        n += labels.numel()
    seconds = time.perf_counter() - start
    return {
        "loss": total_loss.item() / n,
        "top1": correct.item() / n,
        "epoch_seconds": seconds,
        "images_per_second": n / seconds,
    }


def run_training(cfg: PipelineConfig, tracker: Tracker | None = None) -> Path:
    tracker = tracker or Tracker(cfg)
    set_seed(cfg.seed)
    device = resolve_device(cfg.device)
    print(f"[train] device={device} method={cfg.method} run={cfg.run_name}")

    splits = build_splits(cfg.data, cfg.talos.calib_fraction, cfg.seed)
    d = cfg.data
    train_loader = make_loader(splits.train, d.batch_size, d.num_workers, shuffle=True)
    val_loader = make_loader(splits.val, d.eval_batch_size, d.num_workers)
    test_loader = make_loader(splits.test, d.eval_batch_size, d.num_workers)

    m = cfg.model
    model = build_model(m.backbone, len(splits.classes), m.pretrained, m.dropout, **m.backbone_kwargs)
    model.to(device)
    backbone = dict(model.backbone.named_parameters())
    base_state = {n: p.detach().cpu().clone() for n, p in backbone.items()}
    o = cfg.optim

    # Phase 1: linear probe on the frozen backbone. Fisher scores need a trained head,
    # and for `linear_probe` this is the whole training run.
    for p in backbone.values():
        p.requires_grad_(False)
    probe_epochs = cfg.head_warmup_epochs if cfg.method == "talos" else cfg.epochs
    opt = torch.optim.SGD(
        model.head.parameters(), lr=o.head_lr, momentum=o.momentum, weight_decay=o.weight_decay
    )
    sched = torch.optim.lr_scheduler.LambdaLR(
        opt, warmup_cosine(probe_epochs * len(train_loader), o.warmup_frac)
    )

    bundle_dir = cfg.bundle_dir
    best = {"top1": -1.0}
    masks: dict[str, torch.Tensor] = {}
    step = 0

    def checkpoint_if_best(val: dict) -> None:
        nonlocal best
        if val["top1"] <= best["top1"]:
            return
        best = {**val, "epoch": step}
        tau = SparseTaskVector.extract(base_state, backbone, masks) if masks else SparseTaskVector.empty()
        bundle_dir.mkdir(parents=True, exist_ok=True)
        tau.save(bundle_dir / B.TASK_VECTOR)
        B.save_head(
            bundle_dir, model.head.weight.detach().cpu().numpy(), model.head.bias.detach().cpu().numpy()
        )
        print(f"[train] new best val top1={val['top1']:.4f} -> {bundle_dir}")

    for epoch in range(probe_epochs):
        tr = _train_epoch(model, train_loader, opt, sched, device, f"probe {epoch + 1}/{probe_epochs}")
        val, _ = evaluate(model, val_loader, device)
        step += 1
        tracker.metrics({**_prefix("train", tr), **_prefix("val", val)}, step=step)
        print(f"[probe {epoch + 1}] train={_fmt(tr)} val={_fmt(val)}")
        checkpoint_if_best(val)

    if cfg.method == "talos":
        # Phase 2: TaLoS mask from Fisher sensitivity on the calibration split.
        t0 = time.perf_counter()
        calib_loader = make_loader(splits.calib, cfg.talos.calib_batch_size, d.num_workers)
        scores = fisher_scores(model, calib_loader, backbone, device)
        masks = calibrate_mask(scores, cfg.talos.sparsity)
        del scores
        trainable = sum(int(m.sum()) for m in masks.values())
        total = sum(m.numel() for m in masks.values())
        print(f"[talos] mask: {trainable:,}/{total:,} backbone weights trainable ({trainable / total:.1%})")
        tracker.metrics(
            {"mask_trainable_fraction": trainable / total, "mask_seconds": time.perf_counter() - t0}
        )
        # The best probe checkpoint has no tau; reset so the first sparse epoch is compared fairly.
        best = {"top1": -1.0}

        # Phase 3: sparse fine-tuning; masked weights get zero gradient (and zero decay).
        for p in backbone.values():
            p.requires_grad_(True)
        dev_masks = {n: m.to(device, dtype=torch.float32) for n, m in masks.items()}
        opt = torch.optim.SGD(
            [
                {"params": list(backbone.values()), "lr": o.lr, "weight_decay": 0.0},
                {"params": list(model.head.parameters()), "lr": o.head_lr, "weight_decay": o.weight_decay},
            ],
            momentum=o.momentum,
        )
        sched = torch.optim.lr_scheduler.LambdaLR(
            opt, warmup_cosine(cfg.epochs * len(train_loader), o.warmup_frac)
        )

        def hook() -> None:
            mask_gradients(backbone, dev_masks, o.weight_decay)

        for epoch in range(cfg.epochs):
            tr = _train_epoch(
                model, train_loader, opt, sched, device, f"talos {epoch + 1}/{cfg.epochs}", hook
            )
            val, _ = evaluate(model, val_loader, device)
            step += 1
            tracker.metrics({**_prefix("train", tr), **_prefix("val", val)}, step=step)
            print(f"[talos {epoch + 1}] train={_fmt(tr)} val={_fmt(val)}")
            checkpoint_if_best(val)

        drift = max_frozen_drift(base_state, backbone, masks)
        tracker.metrics({"frozen_weight_max_drift": drift})
        if drift != 0.0:
            raise RuntimeError(f"frozen weights moved by {drift}; the mask was not respected")

    # Restore the best weights (pretrained + best tau + best head), then final eval.
    best_tau = SparseTaskVector.load(bundle_dir / B.TASK_VECTOR)
    weight, bias = B.load_head(bundle_dir)
    with torch.no_grad():
        for n, p in backbone.items():
            p.copy_(base_state[n].to(device))
        best_tau.apply_(backbone)
        model.head.weight.copy_(torch.from_numpy(weight))
        model.head.bias.copy_(torch.from_numpy(bias))

    val, ref_feats = evaluate(model, val_loader, device, collect_drift=True)
    test, _ = evaluate(model, test_loader, device)
    tracker.metrics({**_prefix("best_val", val), **_prefix("test", test)})
    print(f"[train] best val={_fmt(val)} test={_fmt(test)}")

    B.write_meta(
        bundle_dir,
        {
            "task": cfg.task,
            "run_name": cfg.run_name,
            "dataset": cfg.data.dataset,
            "method": cfg.method,
            "backbone": cfg.model.backbone,
            "backbone_kwargs": cfg.model.backbone_kwargs,
            "pretrained": cfg.model.pretrained,
            "seed": cfg.seed,
            "num_classes": len(splits.classes),
            "classes": splits.classes,
            "sparsity": cfg.talos.sparsity if cfg.method == "talos" else 1.0,
            "task_vector_params": best_tau.n_params,
            "backbone_params": sum(p.numel() for p in backbone.values()),
            "epochs": {"head_warmup": probe_epochs, "sparse": cfg.epochs if cfg.method == "talos" else 0},
            "metrics": {
                "val_top1": val["top1"],
                "val_top5": val["top5"],
                "test_top1": test["top1"],
                "test_top5": test["top5"],
            },
            "preprocessing": {"image_size": IMAGE_SIZE, "mean": MEAN.tolist(), "std": STD.tolist()},
            "drift_reference": build_reference(ref_feats),
            "created_at": datetime.now(timezone.utc).isoformat(timespec="seconds"),
        },
    )
    for name in (B.META, B.HEAD, B.TASK_VECTOR):
        tracker.artifact(bundle_dir / name, "bundle")
    return bundle_dir


def _prefix(prefix: str, d: dict) -> dict:
    return {f"{prefix}_{k}": v for k, v in d.items()}


def _fmt(d: dict) -> str:
    return " ".join(f"{k}={v:.4f}" for k, v in d.items() if k in ("loss", "top1", "top5"))
