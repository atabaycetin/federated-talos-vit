"""Input/prediction drift check with the Population Stability Index (PSI).

At training time we summarise the validation set into per-feature histograms; the
service keeps a rolling window of the same features for live traffic and compares.
"""

from __future__ import annotations

import threading
from collections import deque

import numpy as np

FEATURES = ("mean_r", "mean_g", "mean_b", "contrast", "confidence")
N_BINS = 10
PSI_ALERT = 0.2  # common rule of thumb: <0.1 stable, 0.1-0.2 moderate, >0.2 significant
_EPS = 1e-4


def drift_features(batch: np.ndarray, probs: np.ndarray) -> np.ndarray:
    """batch: (N,3,H,W) preprocessed images; probs: (N,C) softmax -> (N, len(FEATURES))."""
    channel_means = batch.mean(axis=(2, 3))
    contrast = batch.std(axis=(1, 2, 3))[:, None]
    confidence = probs.max(axis=1)[:, None]
    return np.concatenate([channel_means, contrast, confidence], axis=1).astype(np.float32)


def build_reference(features: np.ndarray) -> dict:
    """Quantile bins from reference features, stored as plain JSON-able lists."""
    ref = {}
    for i, name in enumerate(FEATURES):
        col = features[:, i]
        inner = np.unique(np.quantile(col, np.linspace(0, 1, N_BINS + 1)[1:-1]))
        counts = _bin_counts(col, inner)
        ref[name] = {"edges": inner.tolist(), "expected": (counts / counts.sum()).tolist()}
    return ref


def _bin_counts(values: np.ndarray, inner_edges: np.ndarray) -> np.ndarray:
    idx = np.searchsorted(inner_edges, values, side="right")
    return np.bincount(idx, minlength=len(inner_edges) + 1).astype(np.float64)


def psi(expected: np.ndarray, actual: np.ndarray) -> float:
    e = np.clip(expected, _EPS, None)
    a = np.clip(actual, _EPS, None)
    return float(np.sum((a - e) * np.log(a / e)))


class DriftMonitor:
    def __init__(self, reference: dict, window: int = 500, min_samples: int = 100):
        self.reference = reference
        self.min_samples = min_samples
        self._window: deque[np.ndarray] = deque(maxlen=window)
        self._lock = threading.Lock()

    def update(self, features: np.ndarray) -> None:
        with self._lock:
            self._window.extend(features)

    @property
    def n_samples(self) -> int:
        return len(self._window)

    def scores(self) -> dict[str, float] | None:
        """PSI per feature, or None until the window has `min_samples` observations."""
        if self.n_samples < self.min_samples:
            return None
        with self._lock:
            live = np.stack(self._window)
        out = {}
        for i, name in enumerate(FEATURES):
            ref = self.reference[name]
            counts = _bin_counts(live[:, i], np.asarray(ref["edges"]))
            out[name] = psi(np.asarray(ref["expected"]), counts / counts.sum())
        return out
