"""TaLoS: Fisher scoring, least-sensitive mask, masked updates and sparse task vectors.

Following Iurada et al. (ICLR 2025), only the backbone weights with the *lowest*
Fisher sensitivity are updated. The research `calibrate_mask` kept the highest-scoring
ones; this module uses the paper's rule, which keeps the task vectors from interfering
when several are applied to the same backbone.
"""

from __future__ import annotations

from collections.abc import Iterable, Mapping
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import torch
from torch import nn


def fisher_scores(
    model: nn.Module,
    loader: Iterable,
    params: Mapping[str, nn.Parameter],
    device: torch.device,
) -> dict[str, torch.Tensor]:
    """Empirical diagonal Fisher (mean squared gradient) for `params`, on CPU."""
    was_training = model.training
    model.eval()
    grad_flags = {n: p.requires_grad for n, p in params.items()}
    for p in params.values():
        p.requires_grad_(True)

    scores = {n: torch.zeros_like(p, device="cpu") for n, p in params.items()}
    criterion = nn.CrossEntropyLoss()
    n_batches = 0
    for inputs, labels in loader:
        model.zero_grad(set_to_none=True)
        loss = criterion(model(inputs.to(device)), labels.to(device))
        loss.backward()
        for n, p in params.items():
            if p.grad is not None:
                scores[n] += p.grad.detach().pow(2).cpu()
        n_batches += 1

    model.zero_grad(set_to_none=True)
    for n, p in params.items():
        p.requires_grad_(grad_flags[n])
    model.train(was_training)
    return {n: s / max(n_batches, 1) for n, s in scores.items()}


def calibrate_mask(scores: Mapping[str, torch.Tensor], sparsity: float) -> dict[str, torch.Tensor]:
    """Global mask that marks the `1 - sparsity` least-sensitive weights as trainable.

    Returns bool tensors (True = updated during fine-tuning). Exactly
    round((1 - sparsity) * N) entries are True; ties are broken by position.
    """
    names = list(scores)
    flat = torch.cat([scores[n].reshape(-1).float() for n in names])
    keep = round((1.0 - sparsity) * flat.numel())
    alive = torch.zeros(flat.numel(), dtype=torch.bool)
    if keep > 0:
        alive[torch.topk(flat, keep, largest=False, sorted=False).indices] = True

    masks, offset = {}, 0
    for n in names:
        numel = scores[n].numel()
        masks[n] = alive[offset : offset + numel].view_as(scores[n])
        offset += numel
    return masks


def mask_gradients(
    params: Mapping[str, nn.Parameter], masks: Mapping[str, torch.Tensor], weight_decay: float = 0.0
) -> None:
    """Call between backward() and optimizer.step(), with the optimizer's own weight decay off.

    Weight decay is folded in here so it is masked too; with zero gradient (and hence a
    zero momentum buffer) the frozen entries stay bit-identical to the pretrained weights.
    """
    for n, p in params.items():
        if p.grad is None:
            continue
        if weight_decay:
            p.grad.add_(p.detach(), alpha=weight_decay)
        p.grad.mul_(masks[n])


def max_frozen_drift(
    base: Mapping[str, torch.Tensor], tuned: Mapping[str, torch.Tensor], masks: Mapping[str, torch.Tensor]
) -> float:
    """Largest change at a frozen (masked-out) position. Should be exactly 0."""
    drift = 0.0
    for n, m in masks.items():
        delta = (tuned[n].detach().cpu() - base[n].detach().cpu()).abs()
        frozen = delta[~m.cpu()]
        if frozen.numel():
            drift = max(drift, frozen.max().item())
    return drift


@dataclass
class _Entry:
    shape: tuple[int, ...]
    indices: np.ndarray  # flat positions, int64
    values: np.ndarray  # float32 deltas at those positions


class SparseTaskVector:
    """tau = (theta_finetuned - theta_pretrained) restricted to the TaLoS mask.

    Stored as a bit-packed mask plus the nonzero values in a pickle-free .npz.
    """

    def __init__(self, entries: dict[str, _Entry]):
        self.entries = entries

    @classmethod
    def extract(
        cls,
        base: Mapping[str, torch.Tensor],
        tuned: Mapping[str, torch.Tensor],
        masks: Mapping[str, torch.Tensor],
    ) -> SparseTaskVector:
        entries = {}
        for n, m in masks.items():
            idx = torch.nonzero(m.reshape(-1).cpu(), as_tuple=True)[0]
            delta = (tuned[n].detach().cpu().float() - base[n].detach().cpu().float()).reshape(-1)
            entries[n] = _Entry(tuple(m.shape), idx.numpy().astype(np.int64), delta[idx].numpy())
        return cls(entries)

    @classmethod
    def empty(cls) -> SparseTaskVector:
        return cls({})

    @property
    def n_params(self) -> int:
        return sum(e.indices.size for e in self.entries.values())

    @property
    def n_total(self) -> int:
        return sum(int(np.prod(e.shape)) for e in self.entries.values())

    def save(self, path: str | Path) -> None:
        arrays: dict[str, np.ndarray] = {}
        for n, e in self.entries.items():
            numel = int(np.prod(e.shape))
            bits = np.zeros(numel, dtype=bool)
            bits[e.indices] = True
            arrays[f"{n}::shape"] = np.asarray(e.shape, dtype=np.int64)
            arrays[f"{n}::mask"] = np.packbits(bits)
            arrays[f"{n}::values"] = e.values.astype(np.float32)
        np.savez(path, **arrays)

    @classmethod
    def load(cls, path: str | Path) -> SparseTaskVector:
        entries = {}
        with np.load(path, allow_pickle=False) as data:
            names = {k.rsplit("::", 1)[0] for k in data.files}
            for n in sorted(names):
                shape = tuple(int(s) for s in data[f"{n}::shape"])
                numel = int(np.prod(shape))
                bits = np.unpackbits(data[f"{n}::mask"], count=numel).astype(bool)
                entries[n] = _Entry(shape, np.flatnonzero(bits), data[f"{n}::values"])
        return cls(entries)

    @torch.no_grad()
    def apply_(self, params: Mapping[str, torch.Tensor], scale: float = 1.0) -> None:
        """In place: params[n] += scale * tau[n]."""
        for n, e in self.entries.items():
            p = params[n]
            if tuple(p.shape) != e.shape:
                raise ValueError(f"shape mismatch for {n}: {tuple(p.shape)} vs {e.shape}")
            idx = torch.from_numpy(e.indices).to(p.device)
            vals = torch.from_numpy(e.values).to(p.device, p.dtype)
            p.view(-1).index_add_(0, idx, vals, alpha=scale)
