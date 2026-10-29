"""Typed pipeline config, loaded from a single YAML file."""

from __future__ import annotations

from pathlib import Path
from typing import Literal

import yaml
from pydantic import BaseModel, ConfigDict, Field


class _Strict(BaseModel):
    model_config = ConfigDict(extra="forbid")


class ModelConfig(_Strict):
    backbone: str = "vit_small_patch16_224.dino"  # timm id; DINO ViT-S/16
    pretrained: bool = True
    dropout: float = 0.3
    backbone_kwargs: dict = {}  # extra timm kwargs (tests use a tiny ViT)


class DataConfig(_Strict):
    dataset: Literal["cifar10", "cifar100"] = "cifar100"
    root: str = "~/.cache/talos-mlops/datasets"
    val_split: float = 0.1
    batch_size: int = 64
    eval_batch_size: int = 128
    num_workers: int = 4
    limit_train: int | None = None  # cap sample counts for smoke runs
    limit_eval: int | None = None


class OptimConfig(_Strict):
    lr: float = 1e-3  # sparse backbone updates
    head_lr: float = 1e-2
    momentum: float = 0.9
    weight_decay: float = 4e-4
    warmup_frac: float = 0.05  # fraction of steps with linear LR warmup


class TalosConfig(_Strict):
    sparsity: float = Field(0.6, ge=0.0, lt=1.0)  # fraction of backbone weights left frozen
    calib_fraction: float = 0.05  # share of the train split used for Fisher scores
    calib_batch_size: int = 16


class MlflowConfig(_Strict):
    enabled: bool = True
    tracking_uri: str = "sqlite:///mlflow.db"
    experiment: str = "talos-production"
    registered_model: str | None = None  # defaults to "talos-<task>"


class ExportConfig(_Strict):
    opset: int = 18  # minimum for the torch.export-based exporter
    quantize: bool = True


class BenchmarkConfig(_Strict):
    eval_samples: int = 2000
    latency_iters: int = 200
    warmup_iters: int = 20
    threads: int = 4


class PipelineConfig(_Strict):
    run_name: str
    task: str  # name the API exposes; also the model-store directory name
    method: Literal["talos", "linear_probe"] = "talos"
    seed: int = 42
    device: Literal["auto", "cpu", "cuda", "mps"] = "auto"
    output_dir: str = "artifacts"
    head_warmup_epochs: int = 1  # linear probe before Fisher scoring, so the head isn't random
    epochs: int = 4
    model: ModelConfig = ModelConfig()
    data: DataConfig = DataConfig()
    optim: OptimConfig = OptimConfig()
    talos: TalosConfig = TalosConfig()
    mlflow: MlflowConfig = MlflowConfig()
    export: ExportConfig = ExportConfig()
    benchmark: BenchmarkConfig = BenchmarkConfig()

    @property
    def bundle_dir(self) -> Path:
        return Path(self.output_dir) / self.run_name

    @property
    def registered_model_name(self) -> str:
        return self.mlflow.registered_model or f"talos-{self.task}"


def load_config(path: str | Path, overrides: list[str] | None = None) -> PipelineConfig:
    """Load YAML, then apply dotted `key=value` overrides (e.g. `epochs=1`, `data.limit_train=512`)."""
    raw = yaml.safe_load(Path(path).read_text()) or {}
    for item in overrides or []:
        key, _, value = item.partition("=")
        node = raw
        *parents, leaf = key.split(".")
        for part in parents:
            node = node.setdefault(part, {})
        node[leaf] = yaml.safe_load(value)
    return PipelineConfig.model_validate(raw)
