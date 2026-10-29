"""CIFAR datasets wired to the shared preprocessing in `preprocessing.py`."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import numpy as np
import torch
from PIL import Image
from torch.utils.data import DataLoader, Dataset
from torchvision import datasets, transforms

from talos_mlops.config import DataConfig
from talos_mlops.preprocessing import preprocess, resize_center_crop, to_array

_DATASETS = {"cifar10": datasets.CIFAR10, "cifar100": datasets.CIFAR100}


class EvalTransform:
    """Exactly what the service runs on an uploaded image."""

    def __call__(self, img: Image.Image) -> torch.Tensor:
        return torch.from_numpy(preprocess(img))


class TrainTransform:
    """Augment at native resolution (cheap on 32x32), then the shared resize + normalize."""

    def __init__(self):
        self.augment = transforms.Compose(
            [transforms.RandomHorizontalFlip(), transforms.ColorJitter(0.4, 0.4, 0.4, 0.1)]
        )

    def __call__(self, img: Image.Image) -> torch.Tensor:
        return torch.from_numpy(to_array(resize_center_crop(self.augment(img))))


class _Subset(Dataset):
    def __init__(self, base, indices, transform):
        self.base, self.indices, self.transform = base, list(indices), transform

    def __len__(self) -> int:
        return len(self.indices)

    def __getitem__(self, i):
        img, label = self.base[self.indices[i]]
        return self.transform(img), label


@dataclass
class Splits:
    train: Dataset
    val: Dataset
    test: Dataset
    calib: Dataset
    classes: list[str]


def build_splits(cfg: DataConfig, calib_fraction: float, seed: int) -> Splits:
    root = Path(cfg.root).expanduser()
    cls = _DATASETS[cfg.dataset]
    train_full = cls(root, train=True, download=True)
    test_full = cls(root, train=False, download=True)

    rng = np.random.default_rng(seed)
    perm = rng.permutation(len(train_full))
    n_val = int(len(perm) * cfg.val_split)
    val_idx, train_idx = perm[:n_val], perm[n_val:]
    test_idx = np.arange(len(test_full))
    if cfg.limit_train:
        train_idx = train_idx[: cfg.limit_train]
    if cfg.limit_eval:
        val_idx, test_idx = val_idx[: cfg.limit_eval], test_idx[: cfg.limit_eval]
    # Calibration samples come from the train split, without augmentation.
    calib_idx = train_idx[: max(1, int(len(train_idx) * calib_fraction))]

    return Splits(
        train=_Subset(train_full, train_idx, TrainTransform()),
        val=_Subset(train_full, val_idx, EvalTransform()),  # val is not augmented
        test=_Subset(test_full, test_idx, EvalTransform()),
        calib=_Subset(train_full, calib_idx, EvalTransform()),
        classes=list(train_full.classes),
    )


def make_loader(ds: Dataset, batch_size: int, num_workers: int, shuffle: bool = False) -> DataLoader:
    return DataLoader(
        ds,
        batch_size=batch_size,
        shuffle=shuffle,
        num_workers=num_workers,
        persistent_workers=num_workers > 0,
        pin_memory=torch.cuda.is_available(),
    )
