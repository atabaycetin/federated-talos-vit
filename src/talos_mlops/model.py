"""DINO ViT-S/16 backbone + linear head, built from timm so it exports cleanly to ONNX."""

from __future__ import annotations

from pathlib import Path

import timm
import torch
from torch import nn

from talos_mlops.bundle import TASK_VECTOR, load_head, read_meta
from talos_mlops.sparse import SparseTaskVector


class TaskClassifier(nn.Module):
    """`backbone.*` is shared across tasks; `head.*` is per task."""

    def __init__(self, backbone: nn.Module, num_classes: int, dropout: float = 0.0):
        super().__init__()
        self.backbone = backbone
        self.dropout = nn.Dropout(dropout)
        self.head = nn.Linear(backbone.num_features, num_classes)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.head(self.dropout(self.backbone(x)))


def build_backbone(name: str, pretrained: bool = True, **kwargs) -> nn.Module:
    # num_classes=0 -> timm returns the pooled CLS embedding (384-d for ViT-S),
    # same features as torch.hub's dino_vits16 used in the research code.
    return timm.create_model(name, pretrained=pretrained, num_classes=0, **kwargs)


def build_model(
    backbone: str, num_classes: int, pretrained: bool = True, dropout: float = 0.0, **kwargs
) -> TaskClassifier:
    return TaskClassifier(build_backbone(backbone, pretrained, **kwargs), num_classes, dropout)


def resolve_device(name: str = "auto") -> torch.device:
    if name != "auto":
        return torch.device(name)
    if torch.cuda.is_available():
        return torch.device("cuda")
    if torch.backends.mps.is_available():
        return torch.device("mps")
    return torch.device("cpu")


def load_task_model(bundle_dir) -> TaskClassifier:
    """Rebuild a task's model from its bundle: public backbone + sparse tau + head."""
    meta = read_meta(bundle_dir)
    pretrained = meta.get("pretrained", True)
    if not pretrained:
        # Random-init backbones (tests) are reproducible only from the training seed.
        torch.manual_seed(meta["seed"])
    model = build_model(
        meta["backbone"], meta["num_classes"], pretrained=pretrained, **meta.get("backbone_kwargs", {})
    )
    weight, bias = load_head(bundle_dir)
    with torch.no_grad():
        model.head.weight.copy_(torch.from_numpy(weight))
        model.head.bias.copy_(torch.from_numpy(bias))
    SparseTaskVector.load(Path(bundle_dir) / TASK_VECTOR).apply_(dict(model.backbone.named_parameters()))
    return model.eval()
